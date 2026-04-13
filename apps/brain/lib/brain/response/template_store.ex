defmodule Brain.Response.TemplateStore do
  @moduledoc "Stores and retrieves response templates from intent definition files.\n\nThis module:\n- Loads response templates from data/intents/*.json files at startup\n- Loads custom smalltalk responses from data/customSmalltalkResponses_en.json\n- Builds TF-IDF embeddings per-template for similarity-based selection\n- Evaluates conditions for context-aware template selection\n- Provides slot-aware template matching and substitution\n- Supports enrichment hooks for real-time data integration\n\nTemplates are categorized by intent and can include slot placeholders\nlike $location, $artist, etc. that are substituted with entity values.\n\n## Conditional Template Selection\n\nTemplates can specify conditions that must match for selection:\n- `has_entity:person` - Entity of type \"person\" is present\n- `missing_entity:location` - No location entity\n- `slot_filled:address` - Slot has a value\n- `confidence:high` - Confidence >= 0.8\n\nWhen multiple templates match, semantic similarity to the query is used for ranking.\n"

  use GenServer
  require Logger

  alias Brain.Memory.Embedder
  alias Brain.Analysis.IntentRegistry
  alias Brain.Response.ConditionEvaluator

  @intents_path "data/intents"
  @custom_smalltalk_path "data/customSmalltalkResponses_en.json"
  @smalltalk_domain_path "priv/knowledge/domains/smalltalk.json"
  defmodule Template do
    @moduledoc false
    defstruct [:text, :condition, :embedding, :intent]
  end

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
                                   how_are_you:
                                     Map.get(frames, "how_are_you", ["I'm doing well!"])
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

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Check if the store is loaded and ready.\n"
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc "Get response templates for a specific intent.\nReturns a list of template strings.\n"
  def get_templates(intent) do
    Brain.Telemetry.span(:response_template_lookup, %{intent: intent}, fn ->
      GenServer.call(__MODULE__, {:get_templates, intent}, 5_000)
    end)
  end

  @doc "Get a random response template for an intent.\n"
  def get_random_template(intent) do
    case get_templates(intent) do
      [] -> nil
      templates -> Enum.random(templates)
    end
  end

  @doc "Get the best template for an intent using conditions and semantic ranking.\n\nThis is the main entry point for context-aware template selection:\n1. Filter templates by conditions that match the context\n2. Rank matching templates by semantic similarity to the query\n3. Fall back to cross-intent semantic search if no conditions match\n\n## Parameters\n- `intent` - The classified intent name\n- `query_text` - The original user query (for semantic ranking)\n- `context` - Map with entities, filled_slots, missing_slots, confidence, speech_act\n\n## Returns\n- `{:ok, template_text}` - Best matching template\n- `{:ok, template_text, :fallback}` - Template found via cross-intent fallback\n- `{:error, :no_template}` - No suitable template found\n"
  def get_best_template(intent, query_text, context) do
    GenServer.call(__MODULE__, {:get_best_template, intent, query_text, context}, 5000)
  end

  @doc "Get structured templates with conditions for an intent.\nReturns a list of %Template{} structs.\n"
  def get_structured_templates(intent) do
    GenServer.call(__MODULE__, {:get_structured_templates, intent}, 5_000)
  end

  @doc "Filter templates by conditions that match the given context.\n"
  def filter_by_conditions(templates, context) when is_list(templates) do
    Enum.filter(templates, fn template ->
      ConditionEvaluator.evaluate(template.condition, context)
    end)
  end

  @doc "Rank templates by semantic similarity to the query.\nReturns templates sorted by similarity (highest first).\n"
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

  @doc "Find the best matching template for given context using TF-IDF similarity.\n\nOptions:\n- :intent - filter to specific intent\n- :filled_slots - list of slot names that have values\n- :top_k - number of candidates to return (default: 3)\n"
  def find_similar(query_text, opts \\ []) do
    GenServer.call(__MODULE__, {:find_similar, query_text, opts}, 5000)
  end

  @doc "Substitute slot placeholders in a template with entity values.\n\nPlaceholders are in the format $slot_name (e.g., $location, $artist).\n"
  def substitute_slots(template, entities) when is_binary(template) do
    slot_values = build_slot_value_map(entities)

    Enum.reduce(slot_values, template, fn {slot_name, value}, acc ->
      acc
      |> String.replace("$#{slot_name}", value)
      |> String.replace("@#{slot_name}", value)
    end)
  end

  @doc "Get slot parameter definitions for an intent.\nReturns list of %{name, dataType, required, value} maps.\n"
  def get_parameters(intent) do
    GenServer.call(__MODULE__, {:get_parameters, intent}, 5_000)
  end

  @doc "List all loaded intents.\n"
  def list_intents do
    GenServer.call(__MODULE__, :list_intents, 5_000)
  end

  @doc "Get statistics about loaded templates.\n"
  def stats do
    GenServer.call(__MODULE__, :stats, 5_000)
  end

  @doc "Get the intent name for a speech act sub_type.\nDelegates to IntentRegistry for the canonical mapping.\n"
  def intent_for_speech_act(sub_type) when is_atom(sub_type) do
    IntentRegistry.intent_for_speech_act(sub_type)
  end

  def intent_for_speech_act(_) do
    nil
  end

  @doc "Get a response for an expressive speech act.\nFirst tries to find a template, then falls back to built-in responses.\n"
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

  def get_expressive_response(_) do
    nil
  end

  @doc "Get a fallback response for an expressive speech act.\n"
  def get_expressive_fallback(sub_type) when is_atom(sub_type) do
    case Map.get(@expressive_fallbacks, sub_type) do
      nil -> nil
      responses -> Enum.random(responses)
    end
  end

  def get_expressive_fallback(_) do
    nil
  end

  @doc "Add a new template for an intent.\n\nOptions:\n- `:condition` - Optional condition for when to use this template\n- `:source` - Source tag (default: :admin)\n\nReturns `{:ok, template}` or `{:error, reason}`.\n"
  def add_template(intent, text, opts \\ []) when is_binary(intent) and is_binary(text) do
    GenServer.call(__MODULE__, {:add_template, intent, text, opts}, 30_000)
  end

  @doc "Update an existing template text.\n\nReturns `{:ok, updated_template}` or `{:error, :not_found}`.\n"
  def update_template(intent, old_text, new_text) do
    GenServer.call(__MODULE__, {:update_template, intent, old_text, new_text}, 5_000)
  end

  @doc "Remove a template from an intent.\n\nReturns `:ok` or `{:error, :not_found}`.\n"
  def remove_template(intent, text) do
    GenServer.call(__MODULE__, {:remove_template, intent, text}, 5_000)
  end

  @doc "List all templates for an intent with their metadata.\n\nReturns a list of maps with :text, :condition, :source fields.\n"
  def list_templates_with_metadata(intent) do
    GenServer.call(__MODULE__, {:list_templates_with_metadata, intent}, 5_000)
  end

  @doc "Check if there are unsaved admin changes.\n"
  def has_unsaved_changes? do
    GenServer.call(__MODULE__, :has_unsaved_changes?, 5_000)
  end

  @doc "Sync admin-added templates to the templates.json file.\n"
  def sync_to_file do
    GenServer.call(__MODULE__, :sync_to_file, 30_000)
  end

  @doc "Get the path to the templates JSON file.\n"
  def templates_file_path do
    Application.app_dir(:brain)
    |> Path.join("priv/response/templates.json")
  end

  @impl true
  def init(_opts) do
    send(self(), :load_templates)
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

  def handle_call({:add_template, intent, text, opts}, _from, state) do
    condition = Keyword.get(opts, :condition)
    source = Keyword.get(opts, :source, :admin)

    new_template = %Template{
      text: text,
      condition: condition,
      embedding: nil,
      intent: intent
    }

    admin_for_intent = Map.get(state.admin_templates, intent, [])
    updated_admin = Map.put(state.admin_templates, intent, [new_template | admin_for_intent])
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
      updated_texts =
        Enum.map(existing_texts, fn t ->
          if t == old_text do
            new_text
          else
            t
          end
        end)

      existing_structured = Map.get(state.structured_templates, intent, [])

      updated_structured =
        Enum.map(existing_structured, fn t ->
          if t.text == old_text do
            %{t | text: new_text}
          else
            t
          end
        end)

      updated_all =
        Enum.map(state.all_template_structs, fn t ->
          if t.intent == intent and t.text == old_text do
            %{t | text: new_text}
          else
            t
          end
        end)

      updated_admin =
        Map.update(state.admin_templates, intent, [], fn templates ->
          Enum.map(templates, fn t ->
            if t.text == old_text do
              %{t | text: new_text}
            else
              t
            end
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

      updated_all =
        Enum.reject(state.all_template_structs, &(&1.intent == intent and &1.text == text))

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
        admin_for_intent = Map.get(state.admin_templates, intent, [])
        is_admin = Enum.any?(admin_for_intent, &(&1.text == t.text))

        %{
          text: t.text,
          condition: t.condition,
          source:
            if(is_admin) do
              :admin
            else
              :file
            end,
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

  defp do_get_best_template(intent, query_text, context, state) do
    templates = Map.get(state.structured_templates, intent, [])

    templates =
      if templates == [] do
        parent = get_parent_intent(intent)
        Map.get(state.structured_templates, parent, [])
      else
        templates
      end

    matching = filter_by_conditions(templates, context)

    case matching do
      [] ->
        fallback_semantic_search(query_text, state)

      [single] ->
        {:ok, single.text}

      multiple ->
        if is_binary(query_text) do
          case Embedder.embed(query_text) do
            {:ok, query_embedding} ->
              best = rank_by_similarity(multiple, query_embedding) |> List.first()
              {:ok, best.text}

            _ ->
              {:ok, Enum.random(multiple).text}
          end
        else
          {:ok, Enum.random(multiple).text}
        end
    end
  end

  defp fallback_semantic_search(nil, _state), do: {:ok, nil}
  defp fallback_semantic_search("", _state), do: {:ok, nil}

  defp fallback_semantic_search(query_text, state) do
    case Embedder.embed(query_text) do
      {:ok, query_embedding} ->
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
    {templates, structured_templates} = load_consolidated_templates()

    {templates, parameters, structured_templates} =
      if map_size(templates) == 0 do
        load_all_intent_files_with_conditions()
      else
        {templates, %{}, structured_templates}
      end

    Logger.info("Loaded templates for #{map_size(templates)} intents")
    custom_smalltalk = load_custom_smalltalk_responses()
    merged_templates = merge_custom_responses(templates, custom_smalltalk)
    merged_structured = merge_custom_structured_responses(structured_templates, custom_smalltalk)

    Logger.info("Merged #{map_size(custom_smalltalk)} custom smalltalk responses")
    embeddings = build_template_embeddings(merged_templates)
    all_template_structs = build_per_template_embeddings(merged_structured)
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

  defp do_sync_to_file(state) do
    path = templates_file_path()

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

    updated =
      Enum.reduce(state.structured_templates, existing, fn {intent, templates}, acc ->
        tpl_list =
          Enum.map(templates, fn t ->
            admin_for_intent = Map.get(state.admin_templates, intent, [])
            is_admin = Enum.any?(admin_for_intent, &(&1.text == t.text))

            %{
              "text" => t.text,
              "condition" => t.condition,
              "source" =>
                if(is_admin) do
                  "admin"
                else
                  "dialogflow"
                end
            }
          end)

        Map.put(acc, intent, %{"templates" => tpl_list})
      end)

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
    if File.dir?(@intents_path) do
      intent_files =
        Path.join(@intents_path, "*.json")
        |> Path.wildcard()
        |> Enum.reject(&String.contains?(&1, "usersays"))

      Enum.reduce(intent_files, {%{}, %{}, %{}}, fn file_path,
                                                    {templates_acc, params_acc, structured_acc} ->
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
      {%{}, %{}, %{}}
    end
  end

  defp load_intent_file_with_conditions(file_path) do
    with {:ok, content} <- File.read(file_path),
         {:ok, data} <- Jason.decode(content) do
      intent_name = Map.get(data, "name", Path.basename(file_path, ".json"))

      {speech_templates, structured_templates} =
        extract_templates_with_conditions(data, intent_name)

      conditional_structured = extract_conditional_responses(data, intent_name)

      all_structured = structured_templates ++ conditional_structured

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

          new_structs =
            speech_list
            |> Enum.filter(&(is_binary(&1) and String.length(&1) > 0))
            |> Enum.map(fn text ->
              %Template{
                text: text,
                condition:
                  if(condition == "") do
                    nil
                  else
                    condition
                  end,
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
            condition:
              if(condition == "") do
                nil
              else
                condition
              end,
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
        if is_binary(template.text) and template.text != "" do
          case Embedder.embed(template.text) do
            {:ok, embedding} ->
              %{template | embedding: embedding}

            _ ->
              template
          end
        else
          template
        end
      end)
    else
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
    if Embedder.ready?() do
      templates
      |> Enum.filter(fn {_intent, tpls} -> tpls != [] end)
      |> Enum.reduce(%{}, fn {intent, tpls}, acc ->
        combined_text =
          tpls
          |> Enum.filter(&is_binary/1)
          |> Enum.join(" ")

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

  defp do_find_similar(query_text, _opts, _state) when not is_binary(query_text) or query_text == "" do
    {:ok, []}
  end

  defp do_find_similar(query_text, opts, state) do
    intent_filter = Keyword.get(opts, :intent)
    top_k = Keyword.get(opts, :top_k, 3)

    case Embedder.embed(query_text) do
      {:ok, query_embedding} ->
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
          |> Enum.filter(fn {_, sim, tpls} -> sim > 0.1 and tpls != [] end)
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

      if mag1 == 0.0 or mag2 == 0.0 do
        0.0
      else
        dot / (mag1 * mag2)
      end
    end
  end

  defp build_slot_value_map(entities) when is_list(entities) do
    Enum.reduce(entities, %{}, fn entity, acc ->
      entity_type = entity[:entity_type]
      value = entity[:value]

      if entity_type && value do
        slot_names = entity_type_to_slot_names(entity_type)

        Enum.reduce(slot_names, acc, fn slot_name, inner_acc ->
          Map.put(inner_acc, slot_name, value)
        end)
      else
        acc
      end
    end)
  end

  defp build_slot_value_map(_) do
    %{}
  end

  @external_resource Path.join(:code.priv_dir(:brain), "knowledge/entity_slot_mappings.json")
  @entity_slot_names_config Path.join(:code.priv_dir(:brain), "knowledge/entity_slot_mappings.json")
                            |> File.read!()
                            |> Jason.decode!()
                            |> Map.get("entity_type_to_slot_names", %{})

  defp entity_type_to_slot_names(entity_type) do
    Map.get(@entity_slot_names_config, entity_type, [entity_type])
  end

  defp get_parent_intent(intent) when is_binary(intent) do
    case String.split(intent, ".") do
      [_single] -> intent
      parts -> Enum.take(parts, length(parts) - 1) |> Enum.join(".")
    end
  end

  defp get_parent_intent(_) do
    nil
  end

  defp load_custom_smalltalk_responses do
    case File.read(@custom_smalltalk_path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} when is_list(data) ->
            Enum.reduce(data, %{}, fn item, acc ->
              action = Map.get(item, "action")
              answers = Map.get(item, "customAnswers", [])

              if is_binary(action) and is_list(answers) and answers != [] do
                Map.put(acc, action, answers)
              else
                acc
              end
            end)

          {:ok, data} when is_map(data) ->
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

  defp merge_custom_responses(templates, custom) do
    Map.merge(templates, custom, fn _key, intent_tpls, custom_tpls ->
      custom_tpls ++ intent_tpls
    end)
  end
end
