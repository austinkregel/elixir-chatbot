defmodule Brain.Response.TemplateStore do
  @moduledoc """
  Stores and retrieves response templates from intent definition files.

  This module:
  - Loads response templates from data/intents/*.json files at startup
  - Loads custom smalltalk responses from data/customSmalltalkResponses_en.json
  - Builds TF-IDF embeddings per-template for similarity-based selection
  - Evaluates conditions for context-aware template selection
  - Provides slot-aware template matching and substitution
  - Supports enrichment hooks for real-time data integration

  Templates are categorized by intent and can include slot placeholders
  like $location, $artist, etc. that are substituted with entity values.

  ## Conditional Template Selection

  Templates can specify conditions that must match for selection:
  - `has_entity:person` - Entity of type "person" is present
  - `missing_entity:location` - No location entity
  - `slot_filled:address` - Slot has a value
  - `confidence:high` - Confidence >= 0.8

  When multiple templates match, semantic similarity to the query is used for ranking.
  """

  use GenServer
  require Logger

  alias Brain.Memory.Embedder
  alias Brain.Analysis.IntentRegistry
  alias Brain.Response.ConditionEvaluator

  @intents_path "data/intents"
  @custom_smalltalk_path "data/customSmalltalkResponses_en.json"
  @smalltalk_domain_path "priv/knowledge/domains/smalltalk.json"

  # Template struct with text, condition, and embedding
  defmodule Template do
    @moduledoc false
    defstruct [:text, :condition, :embedding, :intent]
  end

  # Load expressive fallbacks from smalltalk.json at compile time
  @external_resource @smalltalk_domain_path

  @expressive_fallbacks (case File.read(@smalltalk_domain_path) do
                           {:ok, content} ->
                             case Jason.decode(content) do
                               {:ok, data} ->
                                 frames = Map.get(data, "response_frames", %{})

                                 %{
                                   greeting: Map.get(frames, "greeting", ["Hello!"]),
                                   farewell: Map.get(frames, "farewell", ["Goodbye!"]),
                                   thanks: Map.get(frames, "thanks", ["You're welcome!"]),
                                   apology: Map.get(frames, "apology", ["No worries!"]),
                                   how_are_you: Map.get(frames, "how_are_you", ["I'm doing well!"])
                                 }

                               {:error, _} ->
                                 %{
                                   greeting: ["Hello!"],
                                   farewell: ["Goodbye!"],
                                   thanks: ["You're welcome!"],
                                   apology: ["No worries!"],
                                   how_are_you: ["I'm doing well!"]
                                 }
                             end

                           {:error, _} ->
                             %{
                               greeting: ["Hello!"],
                               farewell: ["Goodbye!"],
                               thanks: ["You're welcome!"],
                               apology: ["No worries!"],
                               how_are_you: ["I'm doing well!"]
                             }
                         end)

  # Client API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Check if the store is loaded and ready.
  """
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc """
  Get response templates for a specific intent.
  Returns a list of template strings.
  """
  def get_templates(intent) do
    GenServer.call(__MODULE__, {:get_templates, intent})
  end

  @doc """
  Get a random response template for an intent.
  """
  def get_random_template(intent) do
    case get_templates(intent) do
      [] -> nil
      templates -> Enum.random(templates)
    end
  end

  @doc """
  Get the best template for an intent using conditions and semantic ranking.

  This is the main entry point for context-aware template selection:
  1. Filter templates by conditions that match the context
  2. Rank matching templates by semantic similarity to the query
  3. Fall back to cross-intent semantic search if no conditions match

  ## Parameters
  - `intent` - The classified intent name
  - `query_text` - The original user query (for semantic ranking)
  - `context` - Map with entities, filled_slots, missing_slots, confidence, speech_act

  ## Returns
  - `{:ok, template_text}` - Best matching template
  - `{:ok, template_text, :fallback}` - Template found via cross-intent fallback
  - `{:error, :no_template}` - No suitable template found
  """
  def get_best_template(intent, query_text, context) do
    GenServer.call(__MODULE__, {:get_best_template, intent, query_text, context}, 5000)
  end

  @doc """
  Get structured templates with conditions for an intent.
  Returns a list of %Template{} structs.
  """
  def get_structured_templates(intent) do
    GenServer.call(__MODULE__, {:get_structured_templates, intent})
  end

  @doc """
  Filter templates by conditions that match the given context.
  """
  def filter_by_conditions(templates, context) when is_list(templates) do
    Enum.filter(templates, fn template ->
      ConditionEvaluator.evaluate(template.condition, context)
    end)
  end

  @doc """
  Rank templates by semantic similarity to the query.
  Returns templates sorted by similarity (highest first).
  """
  def rank_by_similarity(templates, query_embedding) when is_list(templates) do
    templates
    |> Enum.map(fn template ->
      similarity =
        if template.embedding do
          cosine_similarity(query_embedding, template.embedding)
        else
          0.0
        end

      {template, similarity}
    end)
    |> Enum.sort_by(fn {_, sim} -> -sim end)
    |> Enum.map(fn {template, _} -> template end)
  end

  @doc """
  Find the best matching template for given context using TF-IDF similarity.

  Options:
  - :intent - filter to specific intent
  - :filled_slots - list of slot names that have values
  - :top_k - number of candidates to return (default: 3)
  """
  def find_similar(query_text, opts \\ []) do
    GenServer.call(__MODULE__, {:find_similar, query_text, opts}, 5000)
  end

  @doc """
  Substitute slot placeholders in a template with entity values.

  Placeholders are in the format $slot_name (e.g., $location, $artist).
  """
  def substitute_slots(template, entities) when is_binary(template) do
    # Build a map of slot names to values
    slot_values = build_slot_value_map(entities)

    # Replace each placeholder with its value
    # Placeholders are $slot_name format
    Enum.reduce(slot_values, template, fn {slot_name, value}, acc ->
      # Replace both $slot_name and @slot_name formats
      acc
      |> String.replace("$#{slot_name}", value)
      |> String.replace("@#{slot_name}", value)
    end)
  end

  @doc """
  Get slot parameter definitions for an intent.
  Returns list of %{name, dataType, required, value} maps.
  """
  def get_parameters(intent) do
    GenServer.call(__MODULE__, {:get_parameters, intent})
  end

  @doc """
  List all loaded intents.
  """
  def list_intents do
    GenServer.call(__MODULE__, :list_intents)
  end

  @doc """
  Get statistics about loaded templates.
  """
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc """
  Get the intent name for a speech act sub_type.
  Delegates to IntentRegistry for the canonical mapping.
  """
  def intent_for_speech_act(sub_type) when is_atom(sub_type) do
    IntentRegistry.intent_for_speech_act(sub_type)
  end

  def intent_for_speech_act(_), do: nil

  @doc """
  Get a response for an expressive speech act.
  First tries to find a template, then falls back to built-in responses.
  """
  def get_expressive_response(sub_type) when is_atom(sub_type) do
    intent_name = intent_for_speech_act(sub_type)

    if intent_name && ready?() do
      case get_random_template(intent_name) do
        nil -> get_expressive_fallback(sub_type)
        template -> template
      end
    else
      get_expressive_fallback(sub_type)
    end
  end

  def get_expressive_response(_), do: nil

  @doc """
  Get a fallback response for an expressive speech act.
  """
  def get_expressive_fallback(sub_type) when is_atom(sub_type) do
    case Map.get(@expressive_fallbacks, sub_type) do
      nil -> nil
      responses -> Enum.random(responses)
    end
  end

  def get_expressive_fallback(_), do: nil

  # ============================================================================
  # Runtime CRUD API (for UI management)
  # ============================================================================

  @doc """
  Add a new template for an intent.

  Options:
  - `:condition` - Optional condition for when to use this template
  - `:source` - Source tag (default: :admin)

  Returns `{:ok, template}` or `{:error, reason}`.
  """
  def add_template(intent, text, opts \\ []) when is_binary(intent) and is_binary(text) do
    GenServer.call(__MODULE__, {:add_template, intent, text, opts})
  end

  @doc """
  Update an existing template text.

  Returns `{:ok, updated_template}` or `{:error, :not_found}`.
  """
  def update_template(intent, old_text, new_text) do
    GenServer.call(__MODULE__, {:update_template, intent, old_text, new_text})
  end

  @doc """
  Remove a template from an intent.

  Returns `:ok` or `{:error, :not_found}`.
  """
  def remove_template(intent, text) do
    GenServer.call(__MODULE__, {:remove_template, intent, text})
  end

  @doc """
  List all templates for an intent with their metadata.

  Returns a list of maps with :text, :condition, :source fields.
  """
  def list_templates_with_metadata(intent) do
    GenServer.call(__MODULE__, {:list_templates_with_metadata, intent})
  end

  @doc """
  Check if there are unsaved admin changes.
  """
  def has_unsaved_changes? do
    GenServer.call(__MODULE__, :has_unsaved_changes?)
  end

  @doc """
  Sync admin-added templates to the templates.json file.
  """
  def sync_to_file do
    GenServer.call(__MODULE__, :sync_to_file, 30_000)
  end

  @doc """
  Get the path to the templates JSON file.
  """
  def templates_file_path do
    Application.app_dir(:brain)
    |> Path.join("priv/response/templates.json")
  end

  # Server Callbacks

  @impl true
  def init(_opts) do
    # Load templates asynchronously
    send(self(), :load_templates)

    # Schedule periodic sync (every 5 minutes)
    :timer.send_interval(5 * 60 * 1000, :periodic_sync)

    {:ok,
     %{
       ready: false,
       templates: %{},
       structured_templates: %{},
       parameters: %{},
       embeddings: %{},
       all_template_structs: [],
       loading: true,
       dirty: false,
       admin_templates: %{}
     }}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, state.ready, state}
  end

  def handle_call({:get_templates, intent}, _from, state) do
    templates = Map.get(state.templates, intent, [])
    # Also try parent intent
    templates =
      if templates == [] do
        parent = get_parent_intent(intent)
        Map.get(state.templates, parent, [])
      else
        templates
      end

    {:reply, templates, state}
  end

  def handle_call({:get_parameters, intent}, _from, state) do
    params = Map.get(state.parameters, intent, [])
    {:reply, params, state}
  end

  def handle_call({:get_structured_templates, intent}, _from, state) do
    templates = Map.get(state.structured_templates, intent, [])

    # Also try parent intent if no templates found
    templates =
      if templates == [] do
        parent = get_parent_intent(intent)
        Map.get(state.structured_templates, parent, [])
      else
        templates
      end

    {:reply, templates, state}
  end

  def handle_call({:get_best_template, intent, query_text, context}, _from, state) do
    result = do_get_best_template(intent, query_text, context, state)
    {:reply, result, state}
  end

  def handle_call({:find_similar, query_text, opts}, _from, state) do
    result = do_find_similar(query_text, opts, state)
    {:reply, result, state}
  end

  def handle_call(:list_intents, _from, state) do
    intents = Map.keys(state.templates)
    {:reply, intents, state}
  end

  def handle_call(:stats, _from, state) do
    admin_count =
      state.admin_templates
      |> Map.values()
      |> List.flatten()
      |> length()

    stats = %{
      intent_count: map_size(state.templates),
      template_count: state.templates |> Map.values() |> List.flatten() |> length(),
      structured_template_count: length(state.all_template_structs),
      with_embeddings: map_size(state.embeddings),
      admin_template_count: admin_count,
      has_unsaved_changes: state.dirty,
      ready: state.ready
    }

    {:reply, stats, state}
  end

  # ============================================================================
  # CRUD Handler Implementations
  # ============================================================================

  def handle_call({:add_template, intent, text, opts}, _from, state) do
    condition = Keyword.get(opts, :condition)
    source = Keyword.get(opts, :source, :admin)

    new_template = %Template{
      text: text,
      condition: condition,
      embedding: nil,
      intent: intent
    }

    # Add to admin_templates tracking
    admin_for_intent = Map.get(state.admin_templates, intent, [])
    updated_admin = Map.put(state.admin_templates, intent, [new_template | admin_for_intent])

    # Also add to the main templates for immediate use
    existing_texts = Map.get(state.templates, intent, [])
    updated_texts = [text | existing_texts]

    existing_structured = Map.get(state.structured_templates, intent, [])
    updated_structured = [new_template | existing_structured]

    new_state = %{
      state
      | admin_templates: updated_admin,
        templates: Map.put(state.templates, intent, updated_texts),
        structured_templates: Map.put(state.structured_templates, intent, updated_structured),
        all_template_structs: [new_template | state.all_template_structs],
        dirty: true
    }

    Logger.info("Added template for #{intent}: #{String.slice(text, 0, 50)}...")

    {:reply, {:ok, %{text: text, condition: condition, source: source}}, new_state}
  end

  def handle_call({:update_template, intent, old_text, new_text}, _from, state) do
    existing_texts = Map.get(state.templates, intent, [])

    if old_text in existing_texts do
      # Update in templates
      updated_texts = Enum.map(existing_texts, fn t -> if t == old_text, do: new_text, else: t end)

      # Update in structured_templates
      existing_structured = Map.get(state.structured_templates, intent, [])

      updated_structured =
        Enum.map(existing_structured, fn t ->
          if t.text == old_text, do: %{t | text: new_text}, else: t
        end)

      # Update in all_template_structs
      updated_all =
        Enum.map(state.all_template_structs, fn t ->
          if t.intent == intent and t.text == old_text, do: %{t | text: new_text}, else: t
        end)

      # Update in admin_templates if it was an admin template
      updated_admin =
        Map.update(state.admin_templates, intent, [], fn templates ->
          Enum.map(templates, fn t ->
            if t.text == old_text, do: %{t | text: new_text}, else: t
          end)
        end)

      new_state = %{
        state
        | templates: Map.put(state.templates, intent, updated_texts),
          structured_templates: Map.put(state.structured_templates, intent, updated_structured),
          all_template_structs: updated_all,
          admin_templates: updated_admin,
          dirty: true
      }

      Logger.info("Updated template for #{intent}")
      {:reply, {:ok, %{text: new_text}}, new_state}
    else
      {:reply, {:error, :not_found}, state}
    end
  end

  def handle_call({:remove_template, intent, text}, _from, state) do
    existing_texts = Map.get(state.templates, intent, [])

    if text in existing_texts do
      updated_texts = Enum.reject(existing_texts, &(&1 == text))

      existing_structured = Map.get(state.structured_templates, intent, [])
      updated_structured = Enum.reject(existing_structured, &(&1.text == text))

      updated_all = Enum.reject(state.all_template_structs, &(&1.intent == intent and &1.text == text))

      updated_admin =
        Map.update(state.admin_templates, intent, [], fn templates ->
          Enum.reject(templates, &(&1.text == text))
        end)

      new_state = %{
        state
        | templates: Map.put(state.templates, intent, updated_texts),
          structured_templates: Map.put(state.structured_templates, intent, updated_structured),
          all_template_structs: updated_all,
          admin_templates: updated_admin,
          dirty: true
      }

      Logger.info("Removed template from #{intent}")
      {:reply, :ok, new_state}
    else
      {:reply, {:error, :not_found}, state}
    end
  end

  def handle_call({:list_templates_with_metadata, intent}, _from, state) do
    structured = Map.get(state.structured_templates, intent, [])

    templates_with_meta =
      Enum.map(structured, fn t ->
        # Check if it's an admin template
        admin_for_intent = Map.get(state.admin_templates, intent, [])
        is_admin = Enum.any?(admin_for_intent, &(&1.text == t.text))

        %{
          text: t.text,
          condition: t.condition,
          source: if(is_admin, do: :admin, else: :file),
          has_embedding: t.embedding != nil
        }
      end)

    {:reply, templates_with_meta, state}
  end

  def handle_call(:has_unsaved_changes?, _from, state) do
    {:reply, state.dirty, state}
  end

  def handle_call(:sync_to_file, _from, state) do
    result = do_sync_to_file(state)
    new_state = %{state | dirty: false}
    {:reply, result, new_state}
  end

  # ============================================================================
  # Best Template Selection Logic
  # ============================================================================

  defp do_get_best_template(intent, query_text, context, state) do
    # Get structured templates for this intent
    templates = Map.get(state.structured_templates, intent, [])

    # Also try parent intent if no templates found
    templates =
      if templates == [] do
        parent = get_parent_intent(intent)
        Map.get(state.structured_templates, parent, [])
      else
        templates
      end

    # Step 1: Filter by conditions
    matching = filter_by_conditions(templates, context)

    case matching do
      [] ->
        # Fallback: semantic search across all intents
        fallback_semantic_search(query_text, state)

      [single] ->
        # Only one match, use it
        {:ok, single.text}

      multiple ->
        # Step 2: Rank by similarity to query
        case Embedder.embed(query_text) do
          {:ok, query_embedding} ->
            best = rank_by_similarity(multiple, query_embedding) |> List.first()
            {:ok, best.text}

          _ ->
            # Embedder not ready, pick random
            {:ok, Enum.random(multiple).text}
        end
    end
  end

  defp fallback_semantic_search(query_text, state) do
    case Embedder.embed(query_text) do
      {:ok, query_embedding} ->
        # Search across all templates
        best =
          state.all_template_structs
          |> Enum.filter(& &1.embedding)
          |> Enum.map(fn template ->
            similarity = cosine_similarity(query_embedding, template.embedding)
            {template, similarity}
          end)
          |> Enum.filter(fn {_, sim} -> sim > 0.1 end)
          |> Enum.sort_by(fn {_, sim} -> -sim end)
          |> List.first()

        case best do
          {template, _similarity} ->
            {:ok, template.text, :fallback}

          nil ->
            {:error, :no_template}
        end

      _ ->
        {:error, :embedder_not_ready}
    end
  end

  @impl true
  def handle_info(:periodic_sync, state) do
    if state.dirty do
      Logger.debug("Periodic sync: saving admin template changes...")
      do_sync_to_file(state)
      {:noreply, %{state | dirty: false}}
    else
      {:noreply, state}
    end
  end

  @impl true
  def handle_info(:load_templates, state) do
    Logger.info("Loading response templates from intent files...")

    # First try to load from consolidated templates.json
    {templates, structured_templates} = load_consolidated_templates()

    # Fall back to legacy loading if templates.json doesn't exist
    {templates, parameters, structured_templates} =
      if map_size(templates) == 0 do
        load_all_intent_files_with_conditions()
      else
        # No parameters from consolidated file, use empty
        {templates, %{}, structured_templates}
      end

    Logger.info("Loaded templates for #{map_size(templates)} intents")

    # Load custom smalltalk responses and merge with templates
    custom_smalltalk = load_custom_smalltalk_responses()
    merged_templates = merge_custom_responses(templates, custom_smalltalk)

    # Also merge into structured templates (custom templates have no conditions)
    merged_structured = merge_custom_structured_responses(structured_templates, custom_smalltalk)

    Logger.info("Merged #{map_size(custom_smalltalk)} custom smalltalk responses")

    # Build embeddings for templates that have content (legacy)
    embeddings = build_template_embeddings(merged_templates)

    # Build per-template embeddings for structured templates
    all_template_structs = build_per_template_embeddings(merged_structured)

    # Update structured_templates with embedded versions
    structured_with_embeddings = group_templates_by_intent(all_template_structs)

    Logger.info("Built embeddings for #{length(all_template_structs)} individual templates")

    {:noreply,
     %{
       state
       | templates: merged_templates,
         structured_templates: structured_with_embeddings,
         parameters: parameters,
         embeddings: embeddings,
         all_template_structs: all_template_structs,
         ready: true,
         loading: false
     }}
  end

  # Private Functions

  # Load from consolidated templates.json (new format)
  defp load_consolidated_templates do
    path = templates_file_path()

    if File.exists?(path) do
      case File.read(path) do
        {:ok, content} ->
          case Jason.decode(content) do
            {:ok, data} when is_map(data) ->
              {templates, structured} =
                Enum.reduce(data, {%{}, %{}}, fn {intent, entry}, {t_acc, s_acc} ->
                  tpl_list = Map.get(entry, "templates", [])

                  texts = Enum.map(tpl_list, fn t -> t["text"] end)

                  structs =
                    Enum.map(tpl_list, fn t ->
                      %Template{
                        text: t["text"],
                        condition: t["condition"],
                        embedding: nil,
                        intent: intent
                      }
                    end)

                  {Map.put(t_acc, intent, texts), Map.put(s_acc, intent, structs)}
                end)

              Logger.info("Loaded #{map_size(templates)} intents from templates.json")
              {templates, structured}

            _ ->
              {%{}, %{}}
          end

        {:error, _} ->
          {%{}, %{}}
      end
    else
      {%{}, %{}}
    end
  end

  # Sync admin templates to file
  defp do_sync_to_file(state) do
    path = templates_file_path()

    # Load existing file
    existing =
      if File.exists?(path) do
        case File.read(path) do
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

    # Build updated data from structured_templates
    updated =
      Enum.reduce(state.structured_templates, existing, fn {intent, templates}, acc ->
        tpl_list =
          Enum.map(templates, fn t ->
            # Determine source
            admin_for_intent = Map.get(state.admin_templates, intent, [])
            is_admin = Enum.any?(admin_for_intent, &(&1.text == t.text))

            %{
              "text" => t.text,
              "condition" => t.condition,
              "source" => if(is_admin, do: "admin", else: "dialogflow")
            }
          end)

        Map.put(acc, intent, %{"templates" => tpl_list})
      end)

    # Write to file
    File.mkdir_p!(Path.dirname(path))

    case File.write(path, Jason.encode!(updated, pretty: true)) do
      :ok ->
        Logger.info("Synced templates to #{path}")
        {:ok, path}

      {:error, reason} ->
        Logger.warning("Failed to sync templates: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp load_all_intent_files_with_conditions do
    # Check if legacy intents directory exists
    if File.dir?(@intents_path) do
      intent_files =
        Path.join(@intents_path, "*.json")
        |> Path.wildcard()
        |> Enum.reject(&String.contains?(&1, "usersays"))

      Enum.reduce(intent_files, {%{}, %{}, %{}}, fn file_path, {templates_acc, params_acc, structured_acc} ->
        case load_intent_file_with_conditions(file_path) do
          {:ok, intent_name, speech_templates, parameters, structured_templates} ->
            templates_acc = Map.put(templates_acc, intent_name, speech_templates)
            params_acc = Map.put(params_acc, intent_name, parameters)
            structured_acc = Map.put(structured_acc, intent_name, structured_templates)
            {templates_acc, params_acc, structured_acc}

          {:error, _reason} ->
            {templates_acc, params_acc, structured_acc}
        end
      end)
    else
      # Legacy directory doesn't exist - return empty (templates.json should be used)
      {%{}, %{}, %{}}
    end
  end

  defp load_intent_file_with_conditions(file_path) do
    with {:ok, content} <- File.read(file_path),
         {:ok, data} <- Jason.decode(content) do
      intent_name = Map.get(data, "name", Path.basename(file_path, ".json"))

      # Extract speech templates with conditions from responses
      {speech_templates, structured_templates} = extract_templates_with_conditions(data, intent_name)

      # Also extract from conditionalResponses
      conditional_structured = extract_conditional_responses(data, intent_name)

      all_structured = structured_templates ++ conditional_structured

      # Extract parameter definitions
      parameters =
        data
        |> Map.get("responses", [])
        |> Enum.flat_map(fn response ->
          Map.get(response, "parameters", [])
        end)
        |> Enum.map(fn param ->
          %{
            name: Map.get(param, "name"),
            data_type: Map.get(param, "dataType"),
            required: Map.get(param, "required", false),
            value: Map.get(param, "value"),
            default: Map.get(param, "defaultValue", "")
          }
        end)

      {:ok, intent_name, speech_templates, parameters, all_structured}
    else
      {:error, reason} ->
        Logger.debug("Failed to load intent file #{file_path}: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp extract_templates_with_conditions(data, intent_name) do
    responses = Map.get(data, "responses", [])

    {texts, structs} =
      Enum.reduce(responses, {[], []}, fn response, {texts_acc, structs_acc} ->
        messages = Map.get(response, "messages", [])

        Enum.reduce(messages, {texts_acc, structs_acc}, fn msg, {t_acc, s_acc} ->
          speech_list = Map.get(msg, "speech", [])
          condition = Map.get(msg, "condition", "")

          # Create Template structs for each speech template
          new_structs =
            speech_list
            |> Enum.filter(&(is_binary(&1) and String.length(&1) > 0))
            |> Enum.map(fn text ->
              %Template{
                text: text,
                condition: if(condition == "", do: nil, else: condition),
                embedding: nil,
                intent: intent_name
              }
            end)

          new_texts = Enum.map(new_structs, & &1.text)

          {t_acc ++ new_texts, s_acc ++ new_structs}
        end)
      end)

    {texts, structs}
  end

  defp extract_conditional_responses(data, intent_name) do
    data
    |> Map.get("conditionalResponses", [])
    |> Enum.flat_map(fn cond_response ->
      condition = Map.get(cond_response, "condition", "")
      messages = Map.get(cond_response, "messages", [])

      Enum.flat_map(messages, fn msg ->
        speech_list = Map.get(msg, "speech", [])

        speech_list
        |> Enum.filter(&(is_binary(&1) and String.length(&1) > 0))
        |> Enum.map(fn text ->
          %Template{
            text: text,
            condition: if(condition == "", do: nil, else: condition),
            embedding: nil,
            intent: intent_name
          }
        end)
      end)
    end)
  end

  defp build_per_template_embeddings(structured_templates) do
    if Embedder.ready?() do
      structured_templates
      |> Enum.flat_map(fn {_intent, templates} -> templates end)
      |> Enum.map(fn template ->
        case Embedder.embed(template.text) do
          {:ok, embedding} ->
            %{template | embedding: embedding}

          _ ->
            template
        end
      end)
    else
      # Return templates without embeddings if embedder not ready
      Enum.flat_map(structured_templates, fn {_intent, templates} -> templates end)
    end
  end

  defp group_templates_by_intent(template_structs) do
    Enum.group_by(template_structs, & &1.intent)
  end

  defp merge_custom_structured_responses(structured_templates, custom) do
    Enum.reduce(custom, structured_templates, fn {action, answers}, acc ->
      new_templates =
        Enum.map(answers, fn text ->
          %Template{
            text: text,
            condition: nil,
            embedding: nil,
            intent: action
          }
        end)

      existing = Map.get(acc, action, [])
      Map.put(acc, action, existing ++ new_templates)
    end)
  end


  defp build_template_embeddings(templates) do
    # Only build embeddings if Embedder is ready
    if Embedder.ready?() do
      templates
      |> Enum.filter(fn {_intent, tpls} -> length(tpls) > 0 end)
      |> Enum.reduce(%{}, fn {intent, tpls}, acc ->
        # Combine all templates for this intent into one embedding
        combined_text = Enum.join(tpls, " ")

        case Embedder.embed(combined_text) do
          {:ok, embedding} ->
            Map.put(acc, intent, embedding)

          _ ->
            acc
        end
      end)
    else
      %{}
    end
  end

  defp do_find_similar(query_text, opts, state) do
    intent_filter = Keyword.get(opts, :intent)
    top_k = Keyword.get(opts, :top_k, 3)

    # Get query embedding
    case Embedder.embed(query_text) do
      {:ok, query_embedding} ->
        # Filter and score templates
        candidates =
          state.embeddings
          |> Enum.filter(fn {intent, _} ->
            intent_filter == nil or intent == intent_filter or
              String.starts_with?(intent, intent_filter <> ".")
          end)
          |> Enum.map(fn {intent, embedding} ->
            similarity = cosine_similarity(query_embedding, embedding)
            templates = Map.get(state.templates, intent, [])
            {intent, similarity, templates}
          end)
          |> Enum.filter(fn {_, sim, tpls} -> sim > 0.1 and length(tpls) > 0 end)
          |> Enum.sort_by(fn {_, sim, _} -> -sim end)
          |> Enum.take(top_k)

        {:ok, candidates}

      _ ->
        {:error, :embedder_not_ready}
    end
  end

  defp cosine_similarity(vec1, vec2) when is_list(vec1) and is_list(vec2) do
    if length(vec1) != length(vec2) do
      0.0
    else
      dot = Enum.zip(vec1, vec2) |> Enum.reduce(0.0, fn {a, b}, sum -> sum + a * b end)
      mag1 = :math.sqrt(Enum.reduce(vec1, 0.0, fn x, sum -> sum + x * x end))
      mag2 = :math.sqrt(Enum.reduce(vec2, 0.0, fn x, sum -> sum + x * x end))

      if mag1 == 0.0 or mag2 == 0.0, do: 0.0, else: dot / (mag1 * mag2)
    end
  end

  defp build_slot_value_map(entities) when is_list(entities) do
    Enum.reduce(entities, %{}, fn entity, acc ->
      entity_type = entity[:entity_type]
      value = entity[:value]

      if entity_type && value do
        # Map entity type to common slot names
        slot_names = entity_type_to_slot_names(entity_type)

        Enum.reduce(slot_names, acc, fn slot_name, inner_acc ->
          Map.put(inner_acc, slot_name, value)
        end)
      else
        acc
      end
    end)
  end

  defp build_slot_value_map(_), do: %{}

  # Map entity types to their corresponding slot names
  defp entity_type_to_slot_names(entity_type) do
    mappings = %{
      "location" => ["location", "address", "place"],
      "city" => ["location", "address", "city"],
      "sys.location" => ["location", "address"],
      "date" => ["date", "date-time"],
      "time" => ["time", "date-time"],
      "sys.date-time" => ["date-time", "date", "time"],
      "music-artist" => ["artist", "music-artist"],
      "sys.music-artist" => ["artist", "music-artist"],
      "song" => ["song"],
      "weather-condition" => ["condition", "weather-condition"],
      "topic" => ["topic", "keyword", "category"],
      "device" => ["device"],
      "action" => ["action"]
    }

    Map.get(mappings, entity_type, [entity_type])
  end

  defp get_parent_intent(intent) when is_binary(intent) do
    case String.split(intent, ".") do
      [_single] -> intent
      parts -> Enum.take(parts, length(parts) - 1) |> Enum.join(".")
    end
  end

  defp get_parent_intent(_), do: nil

  # Load custom smalltalk responses from JSON file
  defp load_custom_smalltalk_responses do
    case File.read(@custom_smalltalk_path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} when is_list(data) ->
            # Convert array format to map: %{action => customAnswers}
            Enum.reduce(data, %{}, fn item, acc ->
              action = Map.get(item, "action")
              answers = Map.get(item, "customAnswers", [])

              if is_binary(action) and is_list(answers) and length(answers) > 0 do
                Map.put(acc, action, answers)
              else
                acc
              end
            end)

          {:ok, data} when is_map(data) ->
            # Already in map format
            data

          {:error, reason} ->
            Logger.warning("Failed to parse custom smalltalk responses: #{inspect(reason)}")
            %{}
        end

      {:error, reason} ->
        Logger.debug("Custom smalltalk responses not found: #{inspect(reason)}")
        %{}
    end
  end

  # Merge custom responses with intent templates
  # Custom responses take precedence when both exist
  defp merge_custom_responses(templates, custom) do
    Map.merge(templates, custom, fn _key, intent_tpls, custom_tpls ->
      # Combine both, putting custom first (they'll be randomly selected anyway)
      custom_tpls ++ intent_tpls
    end)
  end
end
