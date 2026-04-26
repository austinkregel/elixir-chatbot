defmodule Brain.ML.EntityExtractor do
  @moduledoc "Entity extraction using gazetteer lookups and classical NLP techniques.\n\nThis module extracts entities from user input text using:\n- Gazetteer lookups for known entities (cities, artists, devices, etc.)\n- BIO-tagged sequence model for unknown entity detection\n- Token-based pattern matching for system entities (dates, numbers)\n\nAvoids regex in favor of tokenizer-based approaches.\n\n## Process Lifecycle\n\nThis module runs as a GenServer that starts with the application,\nensuring the entity maps are always loaded and the process is always\nregistered. This allows the ops dashboard to correctly report the\nloaded status.\n"

  alias Brain.Analysis
  alias Brain.ML
  use GenServer
  require Logger

  alias ML.{Gazetteer, Tokenizer, EntityTrainer, POSTagger}
  alias Analysis.{EntityDisambiguator, TypeHierarchy}
  alias Brain.Telemetry

  @type_mappings_path "evaluation/ner/type_mappings.json"

  @type entity_match :: %{
          entity: String.t(),
          value: String.t(),
          match: String.t(),
          start_pos: integer(),
          end_pos: integer(),
          confidence: float()
        }

  @type entity_map :: %{String.t() => %{entity_type: String.t(), value: String.t()}}

  defp is_location_preposition?(text) do
    prep_tag = TypeHierarchy.config(["pos_tag_roles", "preposition_tag"], "ADP")

    case POSTagger.load_model() do
      {:ok, model} ->
        predictions = POSTagger.predict([text], model)

        case predictions do
          [{_word, tag}] -> tag == prep_tag
          _ -> false
        end

      {:error, _} ->
        false
    end
  end

  defp common_word?(text) do
    lower = String.downcase(text)
    function_tags = TypeHierarchy.config(["pos_tag_roles", "function_word_tags"], [])

    case POSTagger.load_model() do
      {:ok, model} ->
        predictions = POSTagger.predict([lower], model)

        case predictions do
          [{_word, tag}] -> tag in function_tags
          _ -> false
        end

      {:error, _} ->
        false
    end
  end

  @doc """
  Starts the entity extractor GenServer.

  ## Options
    - `:name` - The name to register the GenServer under (default: `#{__MODULE__}`)
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @impl true
  def init(_opts) do
    send(self(), :load_entity_maps)

    {:ok, %{entity_maps: %{}, loaded: false, loading: true}}
  end

  @impl true
  def handle_info(:load_entity_maps, state) do
    entity_maps = do_load_entity_maps()

    Logger.info("EntityExtractor: Entity maps loaded", %{
      entities_count: map_size(entity_maps)
    })

    {:noreply, %{state | entity_maps: entity_maps, loaded: true, loading: false}}
  end

  @impl true
  def handle_call(:get_entity_maps, _from, state) do
    {:reply, state.entity_maps, state}
  end

  @impl true
  def handle_call(:is_loaded?, _from, state) do
    {:reply, state.loaded, state}
  end

  @impl true
  def handle_call(:get_status, _from, state) do
    status = %{
      loaded: state.loaded,
      loading: state.loading,
      entities_count: map_size(state.entity_maps)
    }

    {:reply, status, state}
  end

  @impl true
  def handle_call({:reload}, _from, state) do
    entity_maps = do_load_entity_maps()

    Logger.info("EntityExtractor: Entity maps reloaded", %{
      entities_count: map_size(entity_maps)
    })

    {:reply, {:ok, entity_maps},
     %{state | entity_maps: entity_maps, loaded: true, loading: false}}
  end

  @doc "Returns true if entity maps are loaded.\n"
  def is_loaded?(opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)

    try do
      GenServer.call(server, :is_loaded?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc "Returns the current status of the entity extractor.\n"
  def get_status(opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)

    try do
      GenServer.call(server, :get_status, 100)
    catch
      :exit, _ -> %{loaded: false, loading: false, entities_count: 0}
    end
  end

  @doc "Load entity maps from saved gazetteer or build fresh.\nReturns {:ok, maps} or {:error, reason}.\n\nNote: With the GenServer implementation, entity maps are loaded automatically\non startup. This function now reloads the maps if called explicitly.\n"
  def load_entity_maps(opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)

    try do
      GenServer.call(server, {:reload}, 30_000)
    catch
      :exit, _ ->
        {:ok, do_load_entity_maps()}
    end
  end

  defp do_load_entity_maps do
    models_path = Application.get_env(:brain, :ml)[:models_path]
    gazetteer_path = Path.join(models_path || Brain.priv_path("ml_models"), "gazetteer.term")
    Brain.ML.ModelStore.ensure_local("gazetteer.term", gazetteer_path)

    unless File.exists?(gazetteer_path) do
      raise """
      EntityExtractor: missing gazetteer at #{gazetteer_path}.
      Run `mix train` (gazetteer stage) or, in test, `Brain.Test.ModelFactory.ensure_gazetteer_on_disk!/0`
      before starting the Brain application.
      """
    end

    case File.read(gazetteer_path) do
      {:ok, binary} ->
        try do
          :erlang.binary_to_term(binary)
        rescue
          e ->
            reraise """
            EntityExtractor: corrupt gazetteer at #{gazetteer_path}: #{Exception.message(e)}
            Retrain with `mix train` or delete the file and regenerate.
            """,
                    __STACKTRACE__
        end

      {:error, reason} ->
        raise "EntityExtractor: cannot read gazetteer at #{gazetteer_path}: #{inspect(reason)}"
    end
  end

  @doc "Get entity maps from GenServer.\n"
  def get_entity_maps(opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)

    try do
      GenServer.call(server, :get_entity_maps, 100)
    rescue
      _e ->
        load_entity_maps_fallback()
    catch
      :exit, _ ->
        load_entity_maps_fallback()
    end
  end

  defp load_entity_maps_fallback do
    do_load_entity_maps()
  end

  @doc "Extract entities from text using gazetteer lookups and pattern matching.\nReturns a list of entity matches with positions and confidence scores.\n\n## Options\n\n- `:entity_maps` - Pre-loaded entity maps (optional)\n- `:discourse` - Discourse analysis result for disambiguation context\n- `:speech_act` - Speech act classification result for disambiguation context\n- `:skip_disambiguation` - If true, skip the disambiguation step (default: false)\n- `:world_id` - World ID for world-scoped type inference (required for disambiguation)\n"
  def extract_entities(text, opts \\ [])

  def extract_entities(text, opts) when is_list(opts) do
    Telemetry.span(:entity_extract, %{text_length: String.length(text || "")}, fn ->
      do_extract_entities(text, opts)
    end)
  end

  def extract_entities(text, entity_maps) when is_map(entity_maps) do
    extract_entities(text, entity_maps: entity_maps)
  end

  def extract_entities(text, nil) do
    extract_entities(text, [])
  end

  defp do_extract_entities(text, opts) do
    entity_maps = Keyword.get(opts, :entity_maps) || get_entity_maps()
    discourse = Keyword.get(opts, :discourse)
    speech_act = Keyword.get(opts, :speech_act)
    skip_disambiguation = Keyword.get(opts, :skip_disambiguation, false)
    world_id = Keyword.get(opts, :world_id)
    tokens = Tokenizer.tokenize(text)
    gaz_context = Keyword.take(opts, [:domain, :intent, :world_id])
    gazetteer_entities = extract_gazetteer_entities(tokens, entity_maps, gaz_context)
    system_entities = extract_system_entities(tokens, text)
    location_entities = extract_location_hints(tokens, entity_maps)
    proper_noun_entities = extract_proper_noun_hints(tokens, entity_maps)

    all_entities =
      gazetteer_entities ++ system_entities ++ location_entities ++ proper_noun_entities

    resolved_entities =
      all_entities
      |> resolve_entity_conflicts()
      |> merge_adjacent_entities(tokens)

    disambiguated_entities =
      if skip_disambiguation or (is_nil(discourse) and is_nil(speech_act)) do
        resolved_entities
      else
        disambiguate_entities(resolved_entities, tokens, discourse, speech_act, text, world_id)
      end

    min_confidence = Keyword.get(opts, :min_confidence) || get_min_confidence_threshold()

    disambiguated_entities
    |> filter_by_confidence(min_confidence)
    |> normalize_entity_types()
  end

  @doc """
  Loads the entity type normalization mappings from the data file.

  Returns a map of system type -> canonical type. Cached in persistent_term
  after first load.
  """
  def load_type_mappings do
    case :persistent_term.get({__MODULE__, :type_mappings}, nil) do
      nil ->
        mappings = do_load_type_mappings()
        :persistent_term.put({__MODULE__, :type_mappings}, mappings)
        mappings

      mappings ->
        mappings
    end
  end

  defp do_load_type_mappings do
    path = Brain.priv_path(@type_mappings_path)

    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, %{"system_to_canonical" => mappings}} when is_map(mappings) ->
            mappings

          _ ->
            Logger.warning("Invalid type_mappings.json format, using empty mappings")
            %{}
        end

      {:error, _} ->
        Logger.debug("No type_mappings.json found at #{path}, using empty mappings")
        %{}
    end
  end

  @doc """
  Refines entity types based on intent context and slot schemas.

  After intent classification, numeric entities can be refined to specific
  slot types (final-value, change-value, etc.) based on what the intent's
  slot schema expects. Person entities can be split into given-name/last-name.

  ## Options
    - `:intent` - The classified intent string (e.g., "smarthome.heating.set")
  """
  def refine_entity_types(entities, intent, _opts \\ []) when is_list(entities) and is_binary(intent) do
    schema = Analysis.SlotDetector.get_schema(intent)

    entities
    |> refine_numeric_types(schema)
    |> refine_person_subtypes()
    |> refine_temporal_subtypes()
  end

  defp refine_numeric_types(entities, nil), do: entities

  defp refine_numeric_types(entities, schema) do
    mappings = Map.get(schema, "entity_mappings", %{})

    numeric_slots =
      mappings
      |> Enum.filter(fn {_slot, types} ->
        Enum.any?(types, fn t -> t in ["sys.number", "number"] end)
      end)
      |> Enum.map(fn {slot, _types} -> slot end)

    number_entities = Enum.filter(entities, fn e -> e[:entity_type] == "number" end)
    non_number_entities = Enum.reject(entities, fn e -> e[:entity_type] == "number" end)

    refined_numbers =
      case {numeric_slots, number_entities} do
        {[], _} -> number_entities
        {_, []} -> []
        {[single_slot], _} ->
          Enum.map(number_entities, fn e -> Map.put(e, :entity_type, single_slot) end)
        {slots, nums} ->
          assign_numbers_to_slots(nums, slots)
      end

    non_number_entities ++ refined_numbers
  end

  defp assign_numbers_to_slots(numbers, slots) do
    sorted_numbers = Enum.sort_by(numbers, & &1.start_pos)

    sorted_numbers
    |> Enum.with_index()
    |> Enum.map(fn {entity, idx} ->
      slot = Enum.at(slots, idx) || List.last(slots)
      Map.put(entity, :entity_type, slot)
    end)
  end

  defp refine_person_subtypes(entities) do
    Enum.flat_map(entities, fn entity ->
      if entity[:entity_type] == "person" and entity[:source] == :pos_tagger_propn do
        value = entity[:value] || ""
        tokens = String.split(value, " ", trim: true)

        case tokens do
          [single] ->
            [Map.put(entity, :entity_type, "given-name") |> Map.put(:value, single)]

          [first | rest] ->
            last = List.last(rest)
            given = %{entity | entity_type: "given-name", value: first,
                       match: first, confidence: entity[:confidence]}
            family = %{entity | entity_type: "last-name", value: last,
                        match: last, confidence: entity[:confidence] * 0.95}
            [given, family]

          _ ->
            [entity]
        end
      else
        [entity]
      end
    end)
  end

  defp refine_temporal_subtypes(entities) do
    Enum.map(entities, fn entity ->
      entity_type = entity[:entity_type] || ""
      value = to_string(entity[:value] || "")

      cond do
        entity_type == "number" and is_year_value?(value) ->
          Map.put(entity, :entity_type, "year")

        entity_type in ["date-time", "date", "sys-date", "relative_date"] and is_year_only?(value) ->
          Map.put(entity, :entity_type, "year")

        true ->
          entity
      end
    end)
  end

  defp is_year_value?(value) do
    case Integer.parse(value) do
      {n, ""} -> n >= 1900 and n <= 2100
      _ -> false
    end
  end

  defp is_year_only?(value) do
    trimmed = String.trim(value)
    case Integer.parse(trimmed) do
      {n, ""} -> n >= 1900 and n <= 2100
      _ -> false
    end
  end

  @doc """
  Normalizes entity types using the data-driven type mappings.

  Converts system-specific types (e.g., "heating", "music-artist") to
  canonical gold-standard types (e.g., "device", "artist").
  """
  def normalize_entity_types(entities) when is_list(entities) do
    mappings = load_type_mappings()

    if map_size(mappings) == 0 do
      entities
    else
      Enum.map(entities, fn entity ->
        entity_type = Map.get(entity, :entity_type) || Map.get(entity, "entity_type", "")
        canonical = Map.get(mappings, entity_type, entity_type)
        Map.put(entity, :entity_type, canonical)
      end)
    end
  end

  def normalize_entity_types(entity) when is_map(entity) do
    [entity] |> normalize_entity_types() |> hd()
  end

  defp get_min_confidence_threshold do
    Application.get_env(:brain, :ml)[:entity_confidence_threshold] ||
      Application.get_env(:brain, :ml)[:confidence_threshold] ||
      0.51
  end

  defp filter_by_confidence(entities, min_confidence) when is_float(min_confidence) do
    Enum.filter(entities, fn entity ->
      confidence = Map.get(entity, :confidence) || Map.get(entity, "confidence", 0.0)
      confidence >= min_confidence
    end)
  end

  defp filter_by_confidence(entities, _) do
    entities
  end

  @doc "Extract entities using the BIO-tagged model (if available).\nFalls back to gazetteer if model not loaded.\n"
  def extract_entities_with_model(text, opts \\ []) do
    entity_maps = Keyword.get(opts, :entity_maps) || get_entity_maps()
    gazetteer_entities = extract_entities(text, entity_maps)

    model_entities =
      case EntityTrainer.load_model() do
        {:ok, model} ->
          tokens = Tokenizer.tokenize_words(text)
          predictions = EntityTrainer.predict(tokens, model)

          EntityTrainer.extract_entities_from_bio(predictions)
          |> Enum.map(fn entity ->
            %{
              entity_type: entity.entity_type,
              value: entity.value,
              match: entity.value,
              start_pos: 0,
              end_pos: String.length(entity.value) - 1,
              confidence: entity.confidence
            }
          end)

        {:error, _} ->
          []
      end

    all_entities = gazetteer_entities ++ model_entities

    all_entities
    |> Enum.uniq_by(fn e -> {String.downcase(e.value), e.entity_type} end)
    |> resolve_entity_conflicts()
  end

  defp extract_gazetteer_entities(tokens, entity_maps, gaz_context) do
    if Gazetteer.loaded?() do
      extract_with_gazetteer_server(tokens, gaz_context)
    else
      extract_with_local_maps(tokens, entity_maps)
    end
  end

  defp extract_with_gazetteer_server(tokens, gaz_context) do
    token_texts =
      Enum.map(tokens, fn t ->
        t.text
        |> String.downcase()
        |> strip_non_word_chars()
        |> String.trim()
      end)

    Gazetteer.lookup_spans(token_texts, gaz_context)
    |> Enum.map(fn {start_idx, end_idx, entity_info} ->
      start_token = Enum.at(tokens, start_idx)
      end_token = Enum.at(tokens, end_idx)

      matched_tokens = Enum.slice(tokens, start_idx..end_idx)
      match_text = Enum.map_join(matched_tokens, " ", & &1.text)

      case entity_info do
        infos when is_list(infos) and length(infos) > 1 ->
          primary_info = score_and_pick_primary(infos)
          primary_type = Map.get(primary_info, :entity_type, "unknown")
          entity_value = Map.get(primary_info, :value, match_text)

          %{
            entity_type: primary_type,
            value: entity_value,
            match: match_text,
            start_pos: start_token.start_pos,
            end_pos: end_token.end_pos,
            confidence: calculate_confidence(match_text, primary_type, entity_value),
            types: infos,
            source: :gazetteer
          }

        [single_info] ->
          entity_type = Map.get(single_info, :entity_type, "unknown")
          entity_value = Map.get(single_info, :value, match_text)

          %{
            entity_type: entity_type,
            value: entity_value,
            match: match_text,
            start_pos: start_token.start_pos,
            end_pos: end_token.end_pos,
            confidence: calculate_confidence(match_text, entity_type, entity_value),
            source: :gazetteer
          }

        single_info when is_map(single_info) ->
          entity_type = Map.get(single_info, :entity_type, "unknown")
          entity_value = Map.get(single_info, :value, match_text)

          %{
            entity_type: entity_type,
            value: entity_value,
            match: match_text,
            start_pos: start_token.start_pos,
            end_pos: end_token.end_pos,
            confidence: calculate_confidence(match_text, entity_type, entity_value),
            source: :gazetteer
          }

        _ ->
          %{
            entity_type: "unknown",
            value: match_text,
            match: match_text,
            start_pos: start_token.start_pos,
            end_pos: end_token.end_pos,
            confidence: 0.5,
            source: :gazetteer
          }
      end
    end)
  end

  defp extract_with_local_maps(tokens, entity_maps) do
    max_span = 5
    token_count = length(tokens)

    find_all_local_spans(tokens, entity_maps, 0, token_count, max_span, [])
  end

  defp find_all_local_spans(_tokens, _entity_maps, start_idx, token_count, _max_span, acc)
       when start_idx >= token_count do
    resolve_entity_conflicts(Enum.reverse(acc))
  end

  defp find_all_local_spans(tokens, entity_maps, start_idx, token_count, max_span, acc) do
    match = find_longest_local_match(tokens, entity_maps, start_idx, max_span)

    case match do
      {:ok, end_idx, entity_infos, match_text} ->
        start_token = Enum.at(tokens, start_idx)
        end_token = Enum.at(tokens, end_idx)

        new_entities =
          Enum.map(entity_infos, fn entity_info ->
            entity_type = Map.get(entity_info, :entity_type, "unknown")
            entity_value = Map.get(entity_info, :value, match_text)

            %{
              entity_type: entity_type,
              value: entity_value,
              match: match_text,
              start_pos: start_token.start_pos,
              end_pos: end_token.end_pos,
              confidence: calculate_confidence(match_text, entity_type, entity_value),
              source: :gazetteer
            }
          end)

        find_all_local_spans(
          tokens,
          entity_maps,
          start_idx + 1,
          token_count,
          max_span,
          new_entities ++ acc
        )

      :not_found ->
        find_all_local_spans(tokens, entity_maps, start_idx + 1, token_count, max_span, acc)
    end
  end

  defp find_longest_local_match(tokens, entity_maps, start_idx, max_span) do
    token_count = length(tokens)
    actual_max = min(max_span, token_count - start_idx)

    if actual_max < 1 do
      :not_found
    else
      actual_max..1//-1
      |> Enum.reduce_while(:not_found, fn span_len, _acc ->
        span_tokens = Enum.slice(tokens, start_idx, span_len)
        phrase = Enum.map_join(span_tokens, " ", & &1.text)
        normalized = String.downcase(phrase)

        case Map.get(entity_maps, normalized) do
          nil ->
            {:cont, :not_found}

          entity_info when is_list(entity_info) ->
            {:halt, {:ok, start_idx + span_len - 1, entity_info, phrase}}

          entity_info when is_map(entity_info) ->
            {:halt, {:ok, start_idx + span_len - 1, [entity_info], phrase}}
        end
      end)
    end
  end

  defp extract_system_entities(tokens, _text) do
    number_entities = extract_numbers_from_tokens(tokens)
    date_entities = extract_dates_from_tokens(tokens)
    age_refined = refine_age_entities(number_entities, tokens)

    age_refined ++ date_entities
  end

  defp refine_age_entities(number_entities, tokens) do
    age_indicators = ["years", "year", "old", "aged", "age"]
    token_texts = Enum.map(tokens, fn t -> String.downcase(t.text) end)

    Enum.map(number_entities, fn entity ->
      entity_idx = Enum.find_index(tokens, fn t -> t.start_pos == entity.start_pos end)

      has_age_context =
        if entity_idx do
          next_tokens = Enum.slice(token_texts, (entity_idx + 1)..min(entity_idx + 3, length(token_texts) - 1))
          Enum.any?(next_tokens, fn t -> t in age_indicators end)
        else
          false
        end

      if has_age_context do
        Map.put(entity, :entity_type, "age")
      else
        entity
      end
    end)
  end

  defp extract_numbers_from_tokens(tokens) do
    tokens
    |> Enum.filter(fn token -> token.type == :number end)
    |> Enum.map(fn token ->
      %{
        entity_type: "number",
        value: token.text,
        match: token.text,
        start_pos: token.start_pos,
        end_pos: token.end_pos,
        confidence: 0.9,
        source: :system
      }
    end)
  end

  defp extract_dates_from_tokens(tokens) do
    tokens
    |> Enum.with_index()
    |> Enum.flat_map(fn {token, idx} ->
      lower = String.downcase(token.text)

      case Gazetteer.lookup(lower) do
        :not_found ->
          []

        {:ok, result} ->
          matches =
            if is_map(result) do
              [result]
            else
              result
            end

          temporal_types = TypeHierarchy.config("temporal_types", [])
          month_types = TypeHierarchy.config("month_types", [])

          temporal_match =
            Enum.find(matches, fn match ->
              entity_type = match[:entity_type] || match["entity_type"] || ""
              entity_type in temporal_types
            end)

          case temporal_match do
            nil ->
              []

            match ->
              entity_type = match[:entity_type] || match["entity_type"]

              if entity_type in month_types do
                maybe_date = check_for_date_pattern(tokens, idx)

                case maybe_date do
                  nil ->
                    [
                      %{
                        entity_type: entity_type,
                        value: token.text,
                        match: token.text,
                        start_pos: token.start_pos,
                        end_pos: token.end_pos,
                        confidence: 0.8,
                        source: :system
                      }
                    ]

                  date_entity ->
                    [Map.put(date_entity, :source, :system)]
                end
              else
                [
                  %{
                    entity_type: entity_type,
                    value: token.text,
                    match: token.text,
                    start_pos: token.start_pos,
                    end_pos: token.end_pos,
                    confidence: 0.9,
                    source: :system
                  }
                ]
              end
          end

        _ ->
          []
      end
    end)
  end

  defp check_for_date_pattern(tokens, month_idx) do
    next_token = Enum.at(tokens, month_idx + 1)
    month_token = Enum.at(tokens, month_idx)

    if next_token != nil and next_token.type == :number do
      day_num = next_token.text
      year_token = Enum.at(tokens, month_idx + 2)

      if year_token != nil and year_token.type == :number and String.length(year_token.text) == 4 do
        match_text = "#{month_token.text} #{day_num} #{year_token.text}"

        %{
          entity_type: "date",
          value: match_text,
          match: match_text,
          start_pos: month_token.start_pos,
          end_pos: year_token.end_pos,
          confidence: 0.9
        }
      else
        match_text = "#{month_token.text} #{day_num}"

        %{
          entity_type: "date",
          value: match_text,
          match: match_text,
          start_pos: month_token.start_pos,
          end_pos: next_token.end_pos,
          confidence: 0.85
        }
      end
    else
      nil
    end
  end

  defp extract_location_hints(tokens, entity_maps) do
    tokens
    |> Enum.with_index()
    |> Enum.flat_map(fn {token, idx} ->
      if is_location_preposition?(token.text) do
        extract_following_location(tokens, idx + 1, entity_maps)
      else
        []
      end
    end)
  end

  defp extract_following_location(tokens, start_idx, entity_maps) do
    remaining = Enum.drop(tokens, start_idx)

    location_tokens =
      remaining
      |> Enum.take_while(fn token ->
        capitalized?(token.text) and not common_word?(token.text)
      end)

    if location_tokens != [] do
      location_text = Enum.map_join(location_tokens, " ", & &1.text)
      normalized = String.downcase(location_text)

      unless Map.has_key?(entity_maps, normalized) do
        first_token = List.first(location_tokens)
        last_token = List.last(location_tokens)
        loc_type = classify_location_type(location_text, remaining)

        [
          %{
            entity_type: loc_type,
            value: location_text,
            match: location_text,
            start_pos: first_token.start_pos,
            end_pos: last_token.end_pos,
            confidence: 0.7
          }
        ]
      else
        []
      end
    else
      []
    end
  end

  @street_indicators ["street", "st", "avenue", "ave", "boulevard", "blvd",
                       "road", "rd", "drive", "dr", "lane", "ln", "way",
                       "court", "ct", "place", "pl", "circle", "cir",
                       "highway", "hwy", "parkway", "pkwy", "terrace"]

  defp classify_location_type(location_text, surrounding_tokens) do
    lower = String.downcase(location_text)
    words = String.split(lower, " ", trim: true)

    has_number = Enum.any?(surrounding_tokens, fn t -> t.type == :number end) or
                 Enum.any?(words, fn w ->
                   case Integer.parse(w) do
                     {_, ""} -> true
                     _ -> false
                   end
                 end)

    has_street_word = Enum.any?(words, fn w -> w in @street_indicators end)

    if has_number and has_street_word do
      "address"
    else
      TypeHierarchy.parent_of("city") || "location"
    end
  end

  defp extract_proper_noun_hints(tokens, entity_maps) do
    extract_with_pos_tagger(tokens, entity_maps)
  end

  defp extract_with_pos_tagger(tokens, entity_maps) do
    propn_tag = TypeHierarchy.config(["pos_tag_roles", "proper_noun"], "PROPN")
    default_type = TypeHierarchy.config("default_propn_type", "person")

    case POSTagger.load_model() do
      {:ok, model} ->
        token_texts = Enum.map(tokens, & &1.text)
        predictions = POSTagger.predict(token_texts, model)

        predictions
        |> Enum.with_index()
        |> Enum.reduce([], fn {{word, tag}, idx}, acc ->
          if tag == propn_tag and capitalized?(word) and not in_gazetteer?(word, entity_maps) do
            token = Enum.at(tokens, idx)

            case acc do
              [%{source: :pos_tagger_propn, _merge_end_idx: prev_idx} = prev | rest]
                when prev_idx == idx - 1 ->
                merged = %{prev |
                  value: prev.value <> " " <> word,
                  match: prev.match <> " " <> word,
                  end_pos: token.end_pos,
                  confidence: min(0.8, prev.confidence + 0.05),
                  _merge_end_idx: idx
                }
                [merged | rest]

              _ ->
                entity = %{
                  entity_type: default_type,
                  value: word,
                  match: word,
                  start_pos: token.start_pos,
                  end_pos: token.end_pos,
                  confidence: 0.65,
                  source: :pos_tagger_propn,
                  _merge_end_idx: idx
                }
                [entity | acc]
            end
          else
            acc
          end
        end)
        |> Enum.map(&Map.delete(&1, :_merge_end_idx))
        |> Enum.reverse()

      {:error, _} ->
        []
    end
  end

  defp in_gazetteer?(word, entity_maps) when is_binary(word) do
    normalized = String.downcase(word)
    Map.has_key?(entity_maps, normalized) or Gazetteer.lookup(normalized) != :not_found
  end

  defp in_gazetteer?(%{text: text}, entity_maps) when is_binary(text),
    do: in_gazetteer?(text, entity_maps)

  defp in_gazetteer?(%{"text" => text}, entity_maps) when is_binary(text),
    do: in_gazetteer?(text, entity_maps)

  defp in_gazetteer?(_, _), do: false

  defp score_and_pick_primary(infos) when is_list(infos) do
    adjustments = TypeHierarchy.config("confidence_adjustments", %{})
    source_priority = TypeHierarchy.config("source_priority", %{})

    infos
    |> Enum.sort_by(fn info ->
      entity_type = Map.get(info, :entity_type, "unknown")
      source = Map.get(info, :source, :unknown) |> to_string()

      type_score = Map.get(adjustments, entity_type, 0.0)
      source_score = Map.get(source_priority, source, 0)

      {-source_score, -type_score}
    end)
    |> hd()
  end

  defp calculate_confidence(match_text, entity_type, entity_value) do
    base = min(0.9, 0.5 + String.length(match_text) * 0.03)
    adjustments = TypeHierarchy.config("confidence_adjustments", %{})

    type_bonus = Map.get(adjustments, entity_type, 0.0)

    casing_penalty =
      if entity_value && match_text != entity_value do
        match_lower = String.downcase(match_text)
        value_lower = String.downcase(entity_value)

        if match_lower == value_lower && match_text != entity_value do
          value_is_capitalized = capitalized?(entity_value)
          match_is_lowercase = match_text == match_lower

          penalty =
            if value_is_capitalized && match_is_lowercase do
              0.3
            else
              0.15
            end

          -penalty
        else
          0.0
        end
      else
        0.0
      end

    min(0.95, max(0.1, base + type_bonus + casing_penalty))
  end

  defp capitalized?(text) do
    first = String.first(text)
    first != nil and first == String.upcase(first) and first != String.downcase(first)
  end

  defp resolve_entity_conflicts(matches) do
    sorted = Enum.sort_by(matches, fn m -> {m.start_pos, -String.length(m.match)} end)
    resolve_overlaps(sorted, [])
  end

  defp merge_adjacent_entities(entities, tokens) do
    sorted = Enum.sort_by(entities, & &1.start_pos)

    {merged, _} =
      Enum.reduce(sorted, {[], nil}, fn entity, {acc, prev} ->
        if prev != nil and
             types_compatible_for_merge?(prev, entity) and
             adjacent_in_text?(prev, entity, tokens) do
          merged_type = common_compatible_type(prev, entity)
          combined = %{prev |
            value: prev.value <> " " <> entity.value,
            match: prev.match <> " " <> entity.match,
            end_pos: entity.end_pos,
            entity_type: merged_type,
            confidence: min(0.85, max(prev.confidence, entity.confidence) + 0.05)
          }
          |> Map.delete(:types)
          {List.replace_at(acc, -1, combined), combined}
        else
          {acc ++ [entity], entity}
        end
      end)

    merged
  end

  defp types_compatible_for_merge?(prev, next) do
    prev_types = all_entity_types(prev) |> MapSet.to_list()
    next_types = all_entity_types(next) |> MapSet.to_list()

    Enum.any?(prev_types, fn pt ->
      Enum.any?(next_types, fn nt ->
        TypeHierarchy.compatible?(pt, nt)
      end)
    end)
  end

  defp common_compatible_type(prev, next) do
    prev_types = all_entity_types(prev)
    next_types = all_entity_types(next)
    common = MapSet.intersection(prev_types, next_types)

    cond do
      MapSet.size(common) > 0 ->
        Enum.at(MapSet.to_list(common), 0)

      TypeHierarchy.compatible?(prev.entity_type, next.entity_type) ->
        narrower_type(prev.entity_type, next.entity_type)

      true ->
        prev.entity_type
    end
  end

  defp narrower_type(a, b) do
    cond do
      TypeHierarchy.is_a?(a, b) -> a
      TypeHierarchy.is_a?(b, a) -> b
      true -> a
    end
  end

  defp all_entity_types(entity) do
    primary = [entity.entity_type]
    from_types = entity
      |> Map.get(:types, [])
      |> Enum.map(&(&1[:entity_type] || &1["entity_type"]))
      |> Enum.filter(&is_binary/1)

    MapSet.new(primary ++ from_types)
  end

  defp adjacent_in_text?(prev, next, tokens) do
    between = Enum.filter(tokens, fn t ->
      t.start_pos > prev.end_pos and t.end_pos < next.start_pos
    end)

    Enum.empty?(between) or Enum.all?(between, &(&1.type == :punctuation))
  end

  defp resolve_overlaps([], resolved) do
    Enum.reverse(resolved)
  end

  defp resolve_overlaps([current | rest], resolved) do
    case resolved do
      [] ->
        resolve_overlaps(rest, [current])

      [last | _] = resolved_list ->
        if current.start_pos <= last.end_pos do
          if String.length(current.match) > String.length(last.match) do
            resolve_overlaps(rest, [current | tl(resolved_list)])
          else
            resolve_overlaps(rest, resolved_list)
          end
        else
          resolve_overlaps(rest, [current | resolved_list])
        end
    end
  end

  defp strip_non_word_chars(text) do
    text
    |> String.graphemes()
    |> Enum.filter(&word_or_space_or_hyphen?/1)
    |> Enum.join()
  end

  defp word_or_space_or_hyphen?(grapheme) do
    case grapheme do
      "-" -> true
      " " -> true
      "\t" -> true
      "\n" -> true
      <<c::utf8>> when c in 97..122 or c in 65..90 or c in 48..57 -> true
      <<c::utf8>> when c > 127 -> letter_codepoint?(c)
      _ -> false
    end
  end

  defp letter_codepoint?(codepoint) do
    (codepoint >= 192 and codepoint <= 591) or
      (codepoint >= 880 and codepoint <= 1023) or
      (codepoint >= 1024 and codepoint <= 1279) or
      (codepoint >= 7680 and codepoint <= 7935)
  end

  defp disambiguate_entities(entities, tokens, discourse, speech_act, original_text, world_id) do
    classified_intent = extract_intent_from_speech_act(speech_act)

    context = %{
      discourse: discourse,
      speech_act: speech_act,
      intent: classified_intent,
      original_text: original_text,
      world_id: world_id
    }

    pos_tagged = get_pos_tags(tokens)

    disambiguated =
      entities
      |> Enum.map(fn entity ->
        types = get_entity_types(entity)
        entity_type = entity[:entity_type] || ""

        needs_disambiguation =
          length(types) > 1 or EntityDisambiguator.requires_inference?(entity_type)

        if needs_disambiguation do
          result = EntityDisambiguator.disambiguate_single(entity, pos_tagged, context)

          result = refine_with_poincare(result, types, context)

          :telemetry.execute(
            [:chat_bot, :analysis, :disambiguation, :entity],
            %{
              type_count: max(length(types), 1),
              selected_type: result[:entity_type]
            },
            %{
              value: entity[:value],
              available_types: Enum.map(types, &(&1[:entity_type] || &1[:type])),
              context_type: context_type(context),
              pos_pattern: extract_pos_pattern(pos_tagged)
            }
          )

          result
        else
          entity
        end
      end)

    ambiguous_count =
      Enum.count(entities, fn e ->
        types = get_entity_types(e)
        entity_type = e[:entity_type] || ""
        length(types) > 1 or EntityDisambiguator.requires_inference?(entity_type)
      end)

    if ambiguous_count > 0 do
      :telemetry.execute(
        [:chat_bot, :analysis, :disambiguation, :complete],
        %{
          total_entities: length(entities),
          ambiguous_entities: ambiguous_count
        },
        %{
          has_discourse: discourse != nil,
          has_speech_act: speech_act != nil
        }
      )
    end

    disambiguated
  end

  defp refine_with_poincare(result, candidate_types, context) when length(candidate_types) > 1 do
    if Brain.ML.Poincare.Embeddings.ready?() do
      intent = context[:intent] || ""
      current_type = to_string(result[:entity_type] || "")

      scored =
        candidate_types
        |> Enum.map(fn candidate ->
          type_str = to_string(candidate[:entity_type] || candidate[:type] || "")
          dist = poincare_distance(type_str, intent)
          {type_str, dist, candidate}
        end)
        |> Enum.filter(fn {_t, d, _c} -> d != nil end)
        |> Enum.sort_by(fn {_t, d, _c} -> d end)

      case scored do
        [{best_type, best_dist, _best_candidate} | _] when best_type != current_type ->
          current_dist = poincare_distance(current_type, intent)

          if current_dist != nil and best_dist < current_dist * 0.8 do
            Logger.debug("Poincare refinement: #{current_type} -> #{best_type}",
              current_dist: current_dist, best_dist: best_dist)

            :telemetry.execute(
              [:chat_bot, :analysis, :poincare_refinement],
              %{distance_ratio: best_dist / max(current_dist, 1.0e-7)},
              %{from_type: current_type, to_type: best_type}
            )

            Map.put(result, :entity_type, best_type)
          else
            result
          end

        _ ->
          result
      end
    else
      log_once(:poincare_not_ready, "Poincare embeddings not ready; skipping type refinement")

      :telemetry.execute(
        [:chat_bot, :ml, :poincare, :unavailable],
        %{count: 1},
        %{stage: :entity_disambiguation}
      )

      result
    end
  rescue
    e ->
      Logger.warning("Poincare refinement failed: #{Exception.message(e)}")
      result
  end

  defp refine_with_poincare(result, _types, _context), do: result

  defp poincare_distance(type1, type2) do
    case Brain.ML.Poincare.Embeddings.entity_distance(type1, type2) do
      {:ok, dist} -> dist
      _ -> nil
    end
  rescue
    _ -> nil
  end

  defp context_type(context) do
    profile = context[:profile]
    intent = context[:intent] || extract_intent_from_speech_act(context[:speech_act])

    cond do
      profile != nil and is_map(profile) and Map.get(profile, :domain) == :introduction -> :introduction
      profile != nil and is_map(profile) and Map.get(profile, :domain) == :weather -> :weather
      profile != nil and is_map(profile) and Map.get(profile, :domain) == :music -> :music
      profile != nil and is_map(profile) and Map.get(profile, :domain) == :smarthome -> :device
      String.starts_with?(to_string(intent || ""), "greeting") -> :introduction
      String.starts_with?(to_string(intent || ""), "weather") -> :weather
      String.starts_with?(to_string(intent || ""), "music") -> :music
      String.starts_with?(to_string(intent || ""), "smarthome") -> :device
      true -> :default
    end
  end

  defp extract_intent_from_speech_act(nil) do
    nil
  end

  defp extract_intent_from_speech_act(speech_act) do
    indicators = get_field(speech_act, :indicators) || []

    Enum.find_value(indicators, fn indicator ->
      case String.split(to_string(indicator), ":", parts: 2) do
        ["intent", intent] -> intent
        _ -> nil
      end
    end)
  end

  defp get_field(nil, _key) do
    nil
  end

  defp get_field(struct, key) when is_struct(struct) do
    Map.get(struct, key)
  end

  defp get_field(map, key) when is_map(map) do
    Map.get(map, key)
  end

  defp get_field(_, _) do
    nil
  end

  defp extract_pos_pattern(pos_tagged) do
    pos_tagged
    |> Enum.take(3)
    |> Enum.map_join(
      "-",
      fn
        {_token, tag} -> tag
        tag -> tag
      end
    )
  end

  defp get_pos_tags(tokens) do
    case POSTagger.load_model() do
      {:ok, model} ->
        token_texts =
          Enum.map(tokens, fn
            %{text: text} -> text
            text when is_binary(text) -> text
            _ -> ""
          end)

        POSTagger.predict(token_texts, model)

      {:error, _} ->
        Enum.map(tokens, fn
          %{text: text} -> {text, "X"}
          text when is_binary(text) -> {text, "X"}
          _ -> {"", "X"}
        end)
    end
  end

  defp get_entity_types(entity) do
    cond do
      is_list(Map.get(entity, :types)) ->
        entity.types

      is_list(Map.get(entity, "types")) ->
        entity["types"]

      true ->
        []
    end
  end

  defp log_once(key, message) do
    pt_key = {__MODULE__, :logged, key}
    unless :persistent_term.get(pt_key, false) do
      Logger.info(message)
      :persistent_term.put(pt_key, true)
    end
  end
end
