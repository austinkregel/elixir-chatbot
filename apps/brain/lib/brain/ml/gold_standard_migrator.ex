defmodule Brain.ML.GoldStandardMigrator do
  @moduledoc """
  Migrates intent training data from multiple sources into the gold standard
  evaluation file. Supports destructive mode to delete source files after migration.

  ## Data Sources

  1. `data/intents/*_usersays_en.json` - Dialogflow usersays files
  2. `data/training/intents/*.json` - Enriched training data (tokens, POS, entities)
  3. `data/legacy/intents/*.json` - Legacy Dialogflow definition files

  ## Features

  - Context variant normalization (e.g., "intent - context_ foo" → "intent")
  - Deduplication by text+intent, favoring entries with richer metadata
  - Destructive mode: deletes source files after successful migration

  ## Usage

      # List available intents
      GoldStandardMigrator.list_available_intents()

      # Preview migration for specific intents
      GoldStandardMigrator.preview(["smarthome.lights.switch.off"])

      # Migrate all intents (non-destructive)
      GoldStandardMigrator.migrate_intents(:all)

      # Migrate and delete source files
      GoldStandardMigrator.migrate_intents(:all, destructive: true)
  """

  require Logger

  alias Brain.ML.EvaluationStore

  @data_intents_dir "data/intents"
  @training_intents_dir "data/training/intents"
  @legacy_intents_dir "data/legacy/intents"

  @doc """
  Lists all available intent names found across all data sources.

  Returns a list of maps: `[%{name: "smarthome.lights.switch.off", example_count: 42, sources: [...]}, ...]`
  """
  def list_available_intents do
    # Collect from all sources
    usersays_intents = list_usersays_intents()
    training_intents = list_training_intents()

    # Merge by intent name
    all_intents =
      (usersays_intents ++ training_intents)
      |> Enum.group_by(& &1.name)
      |> Enum.map(fn {name, entries} ->
        total_count = Enum.sum(Enum.map(entries, & &1.example_count))
        sources = Enum.flat_map(entries, & &1.sources) |> Enum.uniq()
        paths = Enum.flat_map(entries, & &1.paths)
        %{name: name, example_count: total_count, sources: sources, paths: paths}
      end)
      |> Enum.sort_by(& &1.name)

    all_intents
  end

  defp list_usersays_intents do
    intents_dir()
    |> list_usersays_files()
    |> Enum.map(fn path ->
      intent_name = extract_intent_name_from_usersays(path)
      count = count_examples(path)
      %{name: intent_name, example_count: count, sources: [:usersays], paths: [path]}
    end)
  end

  defp list_training_intents do
    training_dir = Path.join(project_root(), @training_intents_dir)

    if File.dir?(training_dir) do
      training_dir
      |> File.ls!()
      |> Enum.filter(&String.ends_with?(&1, ".json"))
      |> Enum.map(fn filename ->
        path = Path.join(training_dir, filename)
        intent_name = Path.basename(filename, ".json")
        count = count_examples(path)
        %{name: intent_name, example_count: count, sources: [:training], paths: [path]}
      end)
    else
      []
    end
  end

  @doc """
  Groups available intents by their top-level category (first two segments).

  Returns a map: `%{"smarthome.lights" => [%{name: ..., example_count: ...}, ...], ...}`
  """
  def list_available_intents_grouped do
    list_available_intents()
    |> Enum.group_by(fn intent ->
      intent.name
      |> String.split(".")
      |> Enum.take(2)
      |> Enum.join(".")
    end)
    |> Enum.sort_by(fn {group, _} -> group end)
    |> Enum.into(%{})
  end

  @doc """
  Preview what would be migrated for the given intent names.

  Options:
  - `:limit` - max examples per intent (default: all)
  - `:merge_context_variants` - if true, merge context variants into base intents (default: false)

  Returns `{intent_examples, entity_examples, source_files}` where:
  - `intent_examples` is a list of `%{"text" => "...", "intent" => "..."}`
  - `entity_examples` is a list of `%{"text" => "...", "expected" => [...]}`
  - `source_files` is a list of paths that were read (for destructive cleanup)
  """
  def preview(intent_names, opts \\ []) do
    limit = Keyword.get(opts, :limit, :all)
    merge_contexts? = Keyword.get(opts, :merge_context_variants, false)

    intents = list_available_intents()

    selected =
      case intent_names do
        :all -> intents
        names when is_list(names) -> Enum.filter(intents, fn i -> i.name in names end)
      end

    # Collect all examples from all source paths
    {intent_examples, entity_examples, source_files} =
      Enum.reduce(selected, {[], [], []}, fn intent, {ie_acc, ee_acc, files_acc} ->
        # Parse each source path for this intent
        {ie, ee} =
          Enum.reduce(intent.paths, {[], []}, fn path, {ie_inner, ee_inner} ->
            {new_ie, new_ee} = parse_intent_file(path, intent.name, limit)
            {ie_inner ++ new_ie, ee_inner ++ new_ee}
          end)

        {ie_acc ++ ie, ee_acc ++ ee, files_acc ++ intent.paths}
      end)

    # Optionally normalize context variants (default: keep them separate)
    normalized_examples = normalize_context_variants(intent_examples, merge_context_variants: merge_contexts?)

    # Deduplicate by text+intent, keeping richest metadata
    deduplicated = deduplicate_by_richness(normalized_examples)

    # Also process entity examples
    normalized_entity = normalize_context_variants_for_entities(entity_examples)
    deduplicated_entity = deduplicate_entity_examples(normalized_entity)

    {deduplicated, deduplicated_entity, Enum.uniq(source_files)}
  end

  @doc """
  Migrate intents from all data sources into the gold standard files.

  Options:
  - `:limit` - max examples per intent (default: all)
  - `:append` - if true, append to existing gold standard (default: false for destructive)
  - `:include_ner` - if true, also populate NER gold standard (default: true)
  - `:destructive` - if true, delete source files after successful migration (default: false)
  - `:merge_context_variants` - if true, merge context variants into base intents (default: false)
    Context variants are follow-up utterances (e.g., "how much") that require prior
    conversational context. By default they're kept as separate intents.

  Returns `{:ok, %{intent_count: N, ner_count: M, deleted_files: [...]}}`.
  """
  def migrate_intents(intent_names, opts \\ []) do
    destructive? = Keyword.get(opts, :destructive, false)
    # Default to replace mode (not append) for destructive migrations
    append? = Keyword.get(opts, :append, not destructive?)
    include_ner? = Keyword.get(opts, :include_ner, true)

    # Pass through merge_context_variants option to preview
    preview_opts = Keyword.take(opts, [:limit, :merge_context_variants])
    {intent_examples, entity_examples, source_files} = preview(intent_names, preview_opts)

    intent_count = save_to_gold_standard("intent", intent_examples, append?)

    ner_count =
      if include_ner? do
        non_empty = Enum.filter(entity_examples, fn e -> e["expected"] != [] end)
        save_to_gold_standard("ner", non_empty, append?)
      else
        0
      end

    Logger.info(
      "GoldStandardMigrator: Migrated #{intent_count} intent examples, #{ner_count} NER examples"
    )

    # Delete source files if destructive mode
    deleted_files =
      if destructive? do
        delete_source_files(source_files)
      else
        []
      end

    {:ok, %{intent_count: intent_count, ner_count: ner_count, deleted_files: deleted_files}}
  end

  @doc """
  Delete all source directories (destructive cleanup).
  Call this after all migrations are complete.
  """
  def delete_source_directories do
    dirs_to_delete = [
      Path.join(project_root(), @data_intents_dir),
      Path.join(project_root(), @training_intents_dir),
      Path.join(project_root(), @legacy_intents_dir)
    ]

    custom_smalltalk = Path.join(project_root(), "data/customSmalltalkResponses_en.json")

    deleted =
      Enum.reduce(dirs_to_delete, [], fn dir, acc ->
        if File.dir?(dir) do
          case File.rm_rf(dir) do
            {:ok, files} ->
              Logger.info("Deleted directory: #{dir} (#{length(files)} files)")
              acc ++ files

            {:error, reason, _} ->
              Logger.warning("Failed to delete #{dir}: #{inspect(reason)}")
              acc
          end
        else
          acc
        end
      end)

    # Also delete custom smalltalk file if it exists
    deleted =
      if File.exists?(custom_smalltalk) do
        case File.rm(custom_smalltalk) do
          :ok ->
            Logger.info("Deleted: #{custom_smalltalk}")
            [custom_smalltalk | deleted]

          {:error, reason} ->
            Logger.warning("Failed to delete #{custom_smalltalk}: #{inspect(reason)}")
            deleted
        end
      else
        deleted
      end

    {:ok, deleted}
  end

  @doc """
  Returns summary stats about current gold standard data.
  """
  def gold_standard_stats do
    Enum.into(~w(intent ner sentiment speech_act), %{}, fn task ->
      examples = EvaluationStore.load_gold_standard(task)
      {task, length(examples)}
    end)
  end

  # ============================================================================
  # Intent Metadata Extraction (for intent_registry.json)
  # ============================================================================

  @doc """
  Extract intent metadata from Dialogflow definition files.

  Returns a map of intent_name => metadata where metadata includes:
  - required_slots, optional_slots, entity_mappings
  - clarification_templates (from parameter prompts)
  - input_contexts (prerequisite contexts)
  - output_contexts (contexts this intent sets)
  """
  def extract_intent_metadata do
    definition_files = list_definition_files()

    Enum.reduce(definition_files, %{}, fn path, acc ->
      case extract_metadata_from_file(path) do
        {:ok, intent_name, metadata} ->
          Map.put(acc, intent_name, metadata)

        {:error, _} ->
          acc
      end
    end)
  end

  @doc """
  Merge extracted metadata into the existing intent_registry.json.
  Preserves manually-added entries and updates only extracted fields.
  """
  def merge_into_intent_registry(extracted_metadata, opts \\ []) do
    registry_path = intent_registry_path()
    write? = Keyword.get(opts, :write, false)

    existing =
      if File.exists?(registry_path) do
        case File.read(registry_path) do
          {:ok, content} ->
            case Jason.decode(content) do
              {:ok, data} -> data
              _ -> %{}
            end

          _ ->
            %{}
        end
      else
        %{}
      end

    # Merge: extracted values fill in missing fields, existing values preserved
    merged =
      Map.merge(extracted_metadata, existing, fn _key, extracted, manual ->
        # Manual entries take precedence, but fill in missing fields from extracted
        Map.merge(extracted, manual)
      end)

    if write? do
      File.mkdir_p!(Path.dirname(registry_path))
      File.write!(registry_path, Jason.encode!(merged, pretty: true))
      Logger.info("Updated intent registry with #{map_size(extracted_metadata)} extracted intents")
    end

    {:ok, merged}
  end

  defp intent_registry_path do
    Application.app_dir(:brain)
    |> Path.join("priv/analysis/intent_registry.json")
  end

  defp list_definition_files do
    dir = intents_dir()

    if File.dir?(dir) do
      dir
      |> File.ls!()
      |> Enum.filter(fn f ->
        String.ends_with?(f, ".json") and not String.contains?(f, "_usersays_")
      end)
      |> Enum.map(&Path.join(dir, &1))
      |> Enum.sort()
    else
      []
    end
  end

  defp extract_metadata_from_file(path) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} when is_map(data) ->
            intent_name = Map.get(data, "name", Path.basename(path, ".json"))
            metadata = build_metadata(data)
            {:ok, intent_name, metadata}

          _ ->
            {:error, :invalid_json}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp build_metadata(data) do
    responses = Map.get(data, "responses", [])
    first_response = List.first(responses) || %{}
    parameters = Map.get(first_response, "parameters", [])

    # Extract required/optional slots
    {required, optional} = partition_parameters(parameters)

    # Build entity mappings from dataType
    entity_mappings = build_entity_mappings(parameters)

    # Extract clarification templates from prompts
    clarification_templates = extract_clarification_templates(parameters)

    # Extract contexts
    input_contexts = Map.get(data, "contexts", [])
    output_contexts = extract_output_contexts(first_response)

    # Infer domain and category from intent name
    intent_name = Map.get(data, "name", "")
    {domain, category} = infer_domain_and_category(intent_name)

    %{
      "domain" => domain,
      "category" => category,
      "required" => required,
      "optional" => optional,
      "entity_mappings" => entity_mappings,
      "clarification_templates" => clarification_templates,
      "input_contexts" => input_contexts,
      "output_contexts" => output_contexts
    }
  end

  defp partition_parameters(parameters) do
    {required, optional} =
      Enum.split_with(parameters, fn p -> Map.get(p, "required", false) end)

    {
      Enum.map(required, &Map.get(&1, "name")),
      Enum.map(optional, &Map.get(&1, "name"))
    }
  end

  defp build_entity_mappings(parameters) do
    Enum.reduce(parameters, %{}, fn param, acc ->
      name = Map.get(param, "name")
      data_type = Map.get(param, "dataType", "")

      # Clean the data type (remove @ prefix)
      entity_type = String.replace_prefix(data_type, "@", "")

      if name && entity_type != "" do
        Map.put(acc, name, [entity_type])
      else
        acc
      end
    end)
  end

  defp extract_clarification_templates(parameters) do
    Enum.reduce(parameters, %{}, fn param, acc ->
      name = Map.get(param, "name")
      prompts = Map.get(param, "prompts", [])

      # Get first English prompt
      prompt =
        prompts
        |> Enum.find(fn p -> Map.get(p, "lang") == "en" end)
        |> case do
          nil -> nil
          p -> Map.get(p, "value")
        end

      if name && prompt do
        Map.put(acc, name, prompt)
      else
        acc
      end
    end)
  end

  defp extract_output_contexts(response) do
    response
    |> Map.get("affectedContexts", [])
    |> Enum.map(fn ctx ->
      %{
        "name" => Map.get(ctx, "name"),
        "lifespan" => Map.get(ctx, "lifespan", 5)
      }
    end)
  end

  defp infer_domain_and_category(intent_name) do
    parts = String.split(intent_name, ".")

    domain =
      case parts do
        [first | _] -> first
        _ -> "unknown"
      end

    # Infer category from common patterns
    category =
      cond do
        String.contains?(intent_name, "smalltalk") -> "expressive"
        String.contains?(intent_name, "greeting") -> "expressive"
        String.contains?(intent_name, "query") -> "directive"
        String.contains?(intent_name, "check") -> "directive"
        String.contains?(intent_name, "set") -> "directive"
        String.contains?(intent_name, "control") -> "directive"
        true -> "assertive"
      end

    {domain, category}
  end

  # ============================================================================
  # Response Template Extraction (for templates.json)
  # ============================================================================

  @doc """
  Extract response templates from Dialogflow definition files and custom smalltalk.

  Returns a map of intent_name => %{templates: [...]} where each template has:
  - text: The response text
  - condition: Optional condition for when to use this template
  - source: "dialogflow" or "custom"
  """
  def extract_response_templates do
    # Extract from Dialogflow definition files
    dialogflow_templates = extract_dialogflow_templates()

    # Extract from custom smalltalk file
    custom_templates = extract_custom_smalltalk_templates()

    # Merge, keeping both sources
    merge_template_maps(dialogflow_templates, custom_templates)
  end

  @doc """
  Write extracted templates to the consolidated templates.json file.
  """
  def write_templates_json(templates, opts \\ []) do
    write? = Keyword.get(opts, :write, false)
    path = templates_json_path()

    if write? do
      File.mkdir_p!(Path.dirname(path))
      File.write!(path, Jason.encode!(templates, pretty: true))
      Logger.info("Wrote templates.json with #{map_size(templates)} intents")
      {:ok, path}
    else
      {:ok, templates}
    end
  end

  defp templates_json_path do
    Application.app_dir(:brain)
    |> Path.join("priv/response/templates.json")
  end

  defp extract_dialogflow_templates do
    definition_files = list_definition_files()

    Enum.reduce(definition_files, %{}, fn path, acc ->
      case extract_templates_from_file(path) do
        {:ok, intent_name, templates} when templates != [] ->
          Map.put(acc, intent_name, %{"templates" => templates})

        _ ->
          acc
      end
    end)
  end

  defp extract_templates_from_file(path) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} when is_map(data) ->
            intent_name = Map.get(data, "name", Path.basename(path, ".json"))
            templates = extract_speech_templates(data)
            {:ok, intent_name, templates}

          _ ->
            {:error, :invalid_json}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp extract_speech_templates(data) do
    responses = Map.get(data, "responses", [])

    # Extract from main responses
    main_templates =
      Enum.flat_map(responses, fn response ->
        messages = Map.get(response, "messages", [])

        Enum.flat_map(messages, fn msg ->
          speech_list = Map.get(msg, "speech", [])
          condition = Map.get(msg, "condition", "")

          speech_list
          |> Enum.filter(&(is_binary(&1) and String.trim(&1) != ""))
          |> Enum.map(fn text ->
            %{
              "text" => text,
              "condition" => if(condition == "", do: nil, else: condition),
              "source" => "dialogflow"
            }
          end)
        end)
      end)

    # Extract from conditional responses
    conditional_templates =
      data
      |> Map.get("conditionalResponses", [])
      |> Enum.flat_map(fn cond_response ->
        condition = Map.get(cond_response, "condition", "")
        messages = Map.get(cond_response, "messages", [])

        Enum.flat_map(messages, fn msg ->
          speech_list = Map.get(msg, "speech", [])

          speech_list
          |> Enum.filter(&(is_binary(&1) and String.trim(&1) != ""))
          |> Enum.map(fn text ->
            %{
              "text" => text,
              "condition" => if(condition == "", do: nil, else: condition),
              "source" => "dialogflow"
            }
          end)
        end)
      end)

    main_templates ++ conditional_templates
  end

  defp extract_custom_smalltalk_templates do
    custom_path = Path.join(project_root(), "data/customSmalltalkResponses_en.json")

    case File.read(custom_path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} when is_list(data) ->
            Enum.reduce(data, %{}, fn item, acc ->
              action = Map.get(item, "action")
              answers = Map.get(item, "customAnswers", [])

              if is_binary(action) and is_list(answers) and length(answers) > 0 do
                templates =
                  Enum.map(answers, fn text ->
                    %{
                      "text" => text,
                      "condition" => nil,
                      "source" => "custom"
                    }
                  end)

                Map.put(acc, action, %{"templates" => templates})
              else
                acc
              end
            end)

          _ ->
            %{}
        end

      {:error, _} ->
        %{}
    end
  end

  defp merge_template_maps(dialogflow, custom) do
    Map.merge(dialogflow, custom, fn _intent, df_entry, custom_entry ->
      df_templates = Map.get(df_entry, "templates", [])
      custom_templates = Map.get(custom_entry, "templates", [])

      # Custom templates come first (higher priority)
      %{"templates" => custom_templates ++ df_templates}
    end)
  end

  # ============================================================================
  # Private: File Operations
  # ============================================================================

  defp intents_dir do
    Path.join(project_root(), @data_intents_dir)
  end

  defp project_root do
    # Navigate from brain app dir (_build/dev/lib/brain) to the umbrella root
    Application.app_dir(:brain)
    |> Path.join("../../../..")
    |> Path.expand()
  end

  defp list_usersays_files(dir) do
    if File.dir?(dir) do
      dir
      |> File.ls!()
      |> Enum.filter(&String.ends_with?(&1, "_usersays_en.json"))
      |> Enum.map(&Path.join(dir, &1))
      |> Enum.sort()
    else
      []
    end
  end

  defp extract_intent_name_from_usersays(path) do
    path
    |> Path.basename()
    |> String.replace_suffix("_usersays_en.json", "")
  end

  defp count_examples(path) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} when is_list(data) -> length(data)
          _ -> 0
        end

      _ ->
        0
    end
  end

  defp delete_source_files(files) do
    Enum.reduce(files, [], fn path, deleted ->
      if File.exists?(path) do
        case File.rm(path) do
          :ok ->
            Logger.debug("Deleted source file: #{path}")
            [path | deleted]

          {:error, reason} ->
            Logger.warning("Failed to delete #{path}: #{inspect(reason)}")
            deleted
        end
      else
        deleted
      end
    end)
  end

  # ============================================================================
  # Private: Parsing
  # ============================================================================

  # Parse any intent file format (usersays, training, or legacy)
  defp parse_intent_file(path, intent_name, limit) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} when is_list(data) ->
            parse_list_format(data, intent_name, limit)

          {:ok, data} when is_map(data) ->
            # Legacy definition file - no training examples, skip
            {[], []}

          _ ->
            {[], []}
        end

      _ ->
        {[], []}
    end
  end

  defp parse_list_format(data, intent_name, limit) do
    entries =
      case limit do
        :all -> data
        n when is_integer(n) -> Enum.take(data, n)
      end

    Enum.reduce(entries, {[], []}, fn entry, {ie_acc, ee_acc} ->
      # Handle both Dialogflow format (data array) and direct format (text field)
      {text, entities, metadata} = extract_text_entities_and_metadata(entry)

      # Skip empty text entries
      if text == "" do
        {ie_acc, ee_acc}
      else
        intent_example =
          %{"text" => text, "intent" => intent_name}
          |> maybe_add_metadata(metadata)

        entity_example = %{
          "text" => text,
          "expected" => entities
        }

        {[intent_example | ie_acc], [entity_example | ee_acc]}
      end
    end)
  end

  defp maybe_add_metadata(example, metadata) do
    example
    |> maybe_put("tokens", Map.get(metadata, :tokens))
    |> maybe_put("pos_tags", Map.get(metadata, :pos_tags))
    |> maybe_put("id", Map.get(metadata, :id))
  end

  defp maybe_put(map, _key, nil), do: map
  defp maybe_put(map, _key, []), do: map
  defp maybe_put(map, key, value), do: Map.put(map, key, value)

  # Extract text, entities, and metadata from an entry, handling multiple formats:
  # 1. Dialogflow format: {"data": [{"text": "..."}, {"text": "...", "meta": "...", "alias": "..."}]}
  # 2. Direct/training format: {"text": "...", "entities": [...], "tokens": [...], "pos_tags": [...]}
  defp extract_text_entities_and_metadata(entry) do
    cond do
      # Dialogflow format with data array
      Map.has_key?(entry, "data") ->
        segments = Map.get(entry, "data", [])
        text = assemble_text_from_segments(segments)
        entities = extract_entities_from_segments(segments)
        {text, entities, %{}}

      # Direct/training format with text field
      Map.has_key?(entry, "text") ->
        text = Map.get(entry, "text", "") |> String.trim()
        entities = extract_entities_from_direct(entry)

        metadata = %{
          tokens: Map.get(entry, "tokens"),
          pos_tags: Map.get(entry, "pos_tags"),
          id: Map.get(entry, "id")
        }

        {text, entities, metadata}

      # Unknown format
      true ->
        {"", [], %{}}
    end
  end

  defp assemble_text_from_segments(segments) when is_list(segments) do
    segments
    |> Enum.map(&Map.get(&1, "text", ""))
    |> Enum.join("")
    |> String.trim()
  end

  defp extract_entities_from_segments(segments) when is_list(segments) do
    segments
    |> Enum.filter(&Map.has_key?(&1, "meta"))
    |> Enum.map(fn segment ->
      %{
        "value" => Map.get(segment, "text", ""),
        "type" => Map.get(segment, "alias", "unknown")
      }
    end)
  end

  # Extract entities from direct format: {"entities": [{"entity_type": "person", "value": "Paris"}]}
  defp extract_entities_from_direct(entry) do
    entry
    |> Map.get("entities", [])
    |> Enum.map(fn entity ->
      %{
        "value" => Map.get(entity, "value", Map.get(entity, "text", "")),
        "type" => Map.get(entity, "entity_type", Map.get(entity, "type", "unknown"))
      }
    end)
  end

  # ============================================================================
  # Private: Context Variant Handling
  # ============================================================================

  # Normalize context variant intent names by converting dot-notation context variants
  # to a consistent format. Context variants represent follow-up utterances that
  # require prior conversational context.
  #
  # Examples of context variants:
  #   - "account.balance.check.context_.balance" (follow-up: "how much money")
  #   - "account.balance.check" (standalone: "check my credit card balance")
  #
  # These are kept SEPARATE because they have different semantic meanings:
  #   - Base intents: self-contained, complete utterances
  #   - Context variants: short follow-ups requiring prior context
  #
  # We only normalize the file-based naming convention inconsistencies.
  defp normalize_context_variants(examples, opts \\ []) do
    merge_contexts? = Keyword.get(opts, :merge_context_variants, false)

    if merge_contexts? do
      # Merge context variants into base intents (use with caution)
      Enum.map(examples, fn example ->
        original_intent = example["intent"]
        base_intent = extract_base_intent(original_intent)
        Map.put(example, "intent", base_intent)
      end)
    else
      # Keep context variants separate - just normalize naming format
      # "account.balance.check.context_.balance" stays as-is
      # This preserves the semantic distinction between standalone and follow-up utterances
      examples
    end
  end

  defp normalize_context_variants_for_entities(examples) do
    # Entity examples don't have intent field to normalize
    examples
  end

  # Extract base intent from context variant name
  # "account.balance.check.context_.balance" -> "account.balance.check"
  # "account.balance.check - context_ balance" -> "account.balance.check"
  defp extract_base_intent(intent_name) when is_binary(intent_name) do
    cond do
      # Dot notation: "base.intent.context_.something"
      String.contains?(intent_name, ".context_.") ->
        intent_name
        |> String.split(".context_.", parts: 2)
        |> List.first()
        |> String.trim()

      # Spaced notation: "base.intent - context_ something"
      String.contains?(intent_name, " - context_") ->
        intent_name
        |> String.split(" - context_", parts: 2)
        |> List.first()
        |> String.trim()

      true ->
        intent_name
    end
  end

  defp extract_base_intent(other), do: other

  # ============================================================================
  # Private: Deduplication by Richness
  # ============================================================================

  # Deduplicate by {text, intent}, keeping the entry with the richest metadata
  defp deduplicate_by_richness(examples) do
    examples
    |> Enum.group_by(fn e -> {e["text"], e["intent"]} end)
    |> Enum.map(fn {_key, entries} ->
      # Pick the entry with the most metadata (tokens, pos_tags, entities)
      Enum.max_by(entries, &richness_score/1)
    end)
  end

  defp richness_score(example) do
    tokens_score = if example["tokens"], do: length(example["tokens"]), else: 0
    pos_score = if example["pos_tags"], do: length(example["pos_tags"]), else: 0
    id_score = if example["id"], do: 1, else: 0
    tokens_score + pos_score + id_score
  end

  defp deduplicate_entity_examples(examples) do
    examples
    |> Enum.group_by(fn e -> e["text"] end)
    |> Enum.map(fn {_text, entries} ->
      # Pick the entry with the most entities
      Enum.max_by(entries, fn e -> length(e["expected"] || []) end)
    end)
  end

  # ============================================================================
  # Private: Saving
  # ============================================================================

  defp save_to_gold_standard(task, new_examples, append?) do
    existing =
      if append? do
        EvaluationStore.load_gold_standard(task)
      else
        []
      end

    # Deduplicate by text
    existing_texts = MapSet.new(existing, &Map.get(&1, "text"))

    unique_new =
      Enum.reject(new_examples, fn example ->
        MapSet.member?(existing_texts, Map.get(example, "text"))
      end)

    combined = existing ++ unique_new

    path = EvaluationStore.gold_standard_path(task)
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Jason.encode!(combined, pretty: true))

    length(unique_new)
  end
end
