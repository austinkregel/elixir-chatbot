defmodule Brain.ML.EntityExtractor do
  @moduledoc "Entity extraction using gazetteer lookups and classical NLP techniques.\n\nThis module extracts entities from user input text using:\n- Gazetteer lookups for known entities (cities, artists, devices, etc.)\n- the world's BIO entity model (`Brain.ML.EntityModel`), run on the same tokens\n- Token-based pattern matching for system entities (dates, numbers)\n\nAvoids regex in favor of tokenizer-based approaches.\n\nKnown entities come from `Brain.ML.Gazetteer`, which loads its sources before\nthe application finishes starting. There is no second copy of the gazetteer\nhere.\n\nEvery source's spans are kept as readings (`readings/2`). Overlapping readings\nform one span, decided by `Brain.ML.SpanReliability` from how reliable each\nsource has been for each type on the NER gold standard's training split. A\ndecided span (`resolved: true`) is an entity; a span the data cannot decide\n(`resolved: false`) is no entity of any type, and travels apart from the\nentities (`extract/2`'s `:unresolved_spans`), carrying its readings, its\ncandidates and why it was left undecided. Nothing that consumes entities is\ngiven one.\n"

  alias Brain.Analysis
  alias Brain.ML
  require Logger

  alias ML.{EntityModel, EntityTrainer, Gazetteer, SpanReliability, Tokenizer, POSTagger}
  alias Analysis.{EntityDisambiguator, EntityTypeScorer, SlotTypePreferences, TypeHierarchy}
  alias Brain.Telemetry

  @type_mappings_path "evaluation/ner/type_mappings.json"

  # A decided entity: the reading `Brain.ML.SpanReliability.decide/3` trusted
  # most for the span, with the keys of the other readings of its hypothesis
  # (a gazetteer reading's `:types`, `:metadata`, `:tag`, ...). Disambiguation
  # and type scoring add their own keys beside these.
  @type entity_match :: %{
          required(:entity_type) => String.t(),
          required(:value) => String.t(),
          required(:match) => String.t(),
          required(:start_pos) => non_neg_integer(),
          required(:end_pos) => non_neg_integer(),
          required(:confidence) => float(),
          required(:source) => atom(),
          required(:resolved) => true,
          required(:readings) => [map()],
          required(:candidates) => Brain.Lattice.t(),
          required(:decision) => map(),
          optional(atom()) => term()
        }

  @type entity_map :: %{String.t() => %{entity_type: String.t(), value: String.t()}}

  @doc "Returns true when the gazetteer extraction reads from has loaded.\n"
  def is_loaded?, do: Gazetteer.loaded?()

  @doc "Extract entities from text using gazetteer lookups and pattern matching.\n\nReturns `%{entities: entities, unresolved_spans: spans, candidates: candidates}`,\nthe first two in text order. `entities` holds only the spans the data decided\n(`resolved: true`), with their positions, combined confidence and `:readings`.\n`unresolved_spans` holds the spans it could not decide (`resolved: false`): each\nkeeps its `:readings` (every source's reading, with the source's reliability),\nits `:candidates` (the `Brain.Lattice` of hypotheses its measured readings\nsupported), its `:unresolved_reason`, and as `:entity_type` the type its readings\nshare, or nil. An unresolved span is no entity, so it never appears among the\nentities. `candidates` is `extract_candidates/2`'s result, which `disambiguate/3`\ncan disambiguate again in a context known later.\n\nExtraction is `extract_candidates/2`, then `disambiguate/3` with the\n`:discourse` and `:speech_act` given, then the confidence floor.\n\n## Options\n\n- `:discourse` - Discourse analysis result for disambiguation context\n- `:speech_act` - Speech act classification result for disambiguation context\n- `:skip_disambiguation` - If true, skip the disambiguation step (default: false)\n- `:world_id` - required: the world whose entities are extracted. Absent raises.\n- `:language` - the `Brain.Language` code whose span reliabilities decide the spans.\n  Absent means English; an unsupported code raises.\n- `:min_confidence` - a decided span whose combined confidence is below it is\n  withheld: returned among the unresolved spans, with\n  `unresolved_reason: :below_confidence_floor` (default: the configured entity\n  confidence threshold).\n"
  @spec extract(String.t(), keyword()) :: %{entities: [map()], unresolved_spans: [map()], candidates: map()}
  def extract(text, opts) when is_list(opts) do
    Telemetry.span(:entity_extract, %{text_length: String.length(text || "")}, fn ->
      candidates = extract_candidates(text, opts)

      disambiguated =
        disambiguate(candidates, %{discourse: Keyword.get(opts, :discourse), speech_act: Keyword.get(opts, :speech_act)}, opts)

      {kept, withheld} = apply_confidence_floor(disambiguated, min_confidence(opts), text)

      %{
        entities: kept,
        unresolved_spans: Enum.sort_by(candidates.unresolved_spans ++ withheld, &{&1.start_pos, &1.end_pos}),
        candidates: candidates
      }
    end)
  end

  @doc "The decided entities of `text`: `extract/2`'s `:entities`, with its options.\n"
  def extract_entities(text, opts \\ [])

  def extract_entities(text, opts) when is_list(opts) do
    extract(text, opts).entities
  end

  def extract_entities(text, nil) do
    extract_entities(text, [])
  end

  @doc """
  The spans of `text` the data decided, before disambiguation and before the
  confidence floor: `%{text: text, tokens: tokens, world_id: world_id, entities:
  decided, unresolved_spans: undecided}`. `entities` are the decided spans
  (adjacent ones of compatible types merged) and `unresolved_spans` the spans
  `Brain.ML.SpanReliability.decide/3` could not decide. Takes `extract/2`'s
  `:world_id` and `:language`.
  """
  @spec extract_candidates(String.t(), keyword()) :: map()
  def extract_candidates(text, opts) when is_binary(text) and is_list(opts) do
    debug? = Application.get_env(:brain, :debug_pipeline_timing, false)
    t0 = if debug?, do: System.monotonic_time(:millisecond)

    world_id = world_id!(opts)
    reliability = SpanReliability.loaded!(language!(opts))
    tokens = Tokenizer.tokenize(text)

    if debug?, do: Logger.info("    extractor:tokenize=#{System.monotonic_time(:millisecond) - t0}ms")

    readings = collect_readings(text, tokens, opts, world_id)

    if debug?, do: Logger.info("    extractor:readings=#{System.monotonic_time(:millisecond) - t0}ms (#{length(readings)} found)")

    # The spans the data decided are entities; the rest leave here and never
    # reach a stage that acts on entities.
    {decided, unresolved} =
      readings
      |> SpanReliability.decide(reliability, text: text, canonical: load_type_mappings(), hierarchy: TypeHierarchy)
      |> Enum.split_with(&match?(%{resolved: true}, &1))

    entities = merge_adjacent_entities(decided, tokens)

    if debug?, do: Logger.info("    extractor:decide+merge=#{System.monotonic_time(:millisecond) - t0}ms (#{length(entities)} decided, #{length(unresolved)} unresolved)")

    %{text: text, tokens: tokens, world_id: world_id, entities: entities, unresolved_spans: unresolved}
  end

  @doc """
  Disambiguates the decided spans of `extract_candidates/2`'s result in
  `context`, a map that may hold `:discourse`, `:speech_act` and `:profile`
  (the chunk's `Brain.Analysis.ChunkProfile`, whose domain ranks the slot types
  an ambiguous entity is weighed against). Returns every decided span,
  disambiguated, before the confidence floor. With none of the three, or with
  `skip_disambiguation: true` in `opts`, the spans are returned as decided.
  """
  @spec disambiguate(map(), map(), keyword()) :: [map()]
  def disambiguate(%{text: text, tokens: tokens, world_id: world_id, entities: entities}, context, opts)
      when is_map(context) and is_list(opts) do
    debug? = Application.get_env(:brain, :debug_pipeline_timing, false)
    t0 = if debug?, do: System.monotonic_time(:millisecond)

    case Map.keys(context) -- [:discourse, :speech_act, :profile] do
      [] -> :ok
      other -> raise ArgumentError, "EntityExtractor.disambiguate/3: unknown context keys #{inspect(other)}"
    end

    discourse = Map.get(context, :discourse)
    speech_act = Map.get(context, :speech_act)
    profile = Map.get(context, :profile)

    disambiguated =
      if Keyword.get(opts, :skip_disambiguation, false) or (is_nil(discourse) and is_nil(speech_act) and is_nil(profile)) do
        entities
      else
        disambiguate_entities(entities, tokens, %{discourse: discourse, speech_act: speech_act, profile: profile}, text, world_id)
      end

    if debug?, do: Logger.info("    extractor:disambiguate=#{System.monotonic_time(:millisecond) - t0}ms")

    disambiguated
  end

  # The type an entity holds, and the record of how it was chosen: what
  # `put_types/2` carries from one disambiguation onto another's entities.
  @type_fields [
    :entity_type,
    :types,
    :type_posteriors,
    :p_ordinary,
    :disambiguation_source,
    :disambiguation_reason,
    :disambiguation_confidence
  ]

  @doc """
  Gives each of `entities` the type fields of the entity of `retyped` with the
  same `:start_pos` and `:end_pos` -- its `:entity_type` and the record of how
  disambiguation chose it (#{Enum.map_join(@type_fields, ", ", &"`#{inspect(&1)}`")}),
  each set as `retyped` has it or removed when it has none -- and nothing else:
  values, confidences and every other key stay as they are. An entity without
  a position (one anaphora resolution supplied) is left as it is. A positioned
  entity with no entity, or more than one, at its offsets in `retyped` raises.
  """
  @spec put_types([map()], [map()]) :: [map()]
  def put_types(entities, retyped) when is_list(entities) and is_list(retyped) do
    Enum.map(entities, fn
      %{start_pos: start_pos, end_pos: end_pos} = entity when is_integer(start_pos) and is_integer(end_pos) ->
        case Enum.filter(retyped, &(Map.get(&1, :start_pos) == start_pos and Map.get(&1, :end_pos) == end_pos)) do
          [source] ->
            entity
            |> Map.drop(@type_fields)
            |> Map.merge(Map.take(source, @type_fields))

          found ->
            raise ArgumentError,
                  "EntityExtractor.put_types/2: #{length(found)} retyped entities at #{start_pos}..#{end_pos} " <>
                    "for #{inspect(Map.get(entity, :value))}; exactly one was expected"
        end

      entity ->
        entity
    end)
  end

  @doc """
  Decides again, now that the turn's intent is known, the spans `extract/2`
  left unresolved. A span left for `:insufficient_evidence` is decided again
  by `Brain.ML.SpanReliability.redecide/4`: its extent first, by how far its
  readings' extents can be trusted, then its type among the hypotheses at
  that extent, with what the intent says about each of its hypothesis labels
  (`Brain.Analysis.SlotTypePreferences.intent_evidence/3`) as one more piece
  of evidence, under thresholds learned for deciding again. Nothing else
  decides it: no span is resolved because a source or the intent always wins.

  Returns `%{entities: entities, unresolved_spans: spans, outcomes: outcomes}`:
  the spans the intent's evidence decided, as decided entities whose
  `:decision` records `resolved_by: {:intent, intent}` and the evidence; every
  other span, unchanged; and one outcome per given span, in text order, saying
  what became of it and why:

  - `%{outcome: :resolved, entity_type, score, margin, evidence}`;
  - `%{outcome: :undecided, best_type, score, margin, evidence}` -- the
    evidence did not lift the span over the thresholds;
  - `%{outcome: :below_confidence_floor, entity_type, score, margin, evidence}`
    -- decided, but with a confidence below the floor `extract/2` applies, so
    the span stays unresolved as it was;
  - `%{outcome: :not_redecided, reason}` -- `:no_measured_reading` (no
    hypothesis to weigh), `:below_confidence_floor` (decided before, withheld
    by the floor), `:no_intent`, `:intent_without_domain` (an intent whose
    prefix `Brain.Lexicon.IntentDomains` does not declare, for which the slot
    type preferences hold no distribution), `:extent_undecided` (the readings'
    extents decide no extent, so the evidence is not weighed), or
    `:no_hypothesis_at_extent` (no measured reading gives a type at the
    extent decided).

  Each outcome also carries the span's `:value`, `:start_pos` and `:end_pos`.

  Options: `:language`, `:min_confidence` as for `extract/2`;
  `:span_reliability` and `:preferences`, models used in place of the ones
  loaded for the language.
  """
  @spec resolve_with_intent([map()], String.t() | atom() | nil, keyword()) :: %{
          entities: [map()],
          unresolved_spans: [map()],
          outcomes: [map()]
        }
  def resolve_with_intent(spans, intent, opts) when is_list(spans) and is_list(opts) do
    language = language!(opts)
    intent = if is_nil(intent), do: nil, else: to_string(intent)

    results =
      spans
      |> Enum.sort_by(&{&1.start_pos, &1.end_pos})
      |> Enum.map(&resolve_span(&1, intent, language, opts))

    %{
      entities: for({:entity, entity, _outcome} <- results, do: entity),
      unresolved_spans: for({:span, span, _outcome} <- results, do: span),
      outcomes: Enum.map(results, &elem(&1, 2))
    }
  end

  defp resolve_span(%{unresolved_reason: :insufficient_evidence} = span, intent, language, opts) do
    cond do
      intent in [nil, ""] ->
        {:span, span, outcome(span, %{outcome: :not_redecided, reason: :no_intent})}

      not Brain.Lexicon.IntentDomains.declared?(intent |> String.split(".") |> hd()) ->
        {:span, span, outcome(span, %{outcome: :not_redecided, reason: :intent_without_domain})}

      true ->
        redecide_span(span, intent, language, opts)
    end
  end

  defp resolve_span(%{unresolved_reason: reason} = span, _intent, _language, _opts)
       when reason in [:no_measured_reading, :below_confidence_floor] do
    {:span, span, outcome(span, %{outcome: :not_redecided, reason: reason})}
  end

  defp resolve_span(span, _intent, _language, _opts) do
    raise ArgumentError, "EntityExtractor.resolve_with_intent/3: not an unresolved span: #{inspect(span, limit: 8)}"
  end

  defp redecide_span(span, intent, language, opts) do
    reliability =
      case Keyword.fetch(opts, :span_reliability) do
        {:ok, %SpanReliability{} = model} -> model
        {:ok, other} -> raise ArgumentError, "EntityExtractor: :span_reliability must be a SpanReliability model, got #{inspect(other, limit: 3)}"
        :error -> SpanReliability.loaded!(language)
      end

    preferences =
      case Keyword.fetch(opts, :preferences) do
        {:ok, %SlotTypePreferences{} = model} -> model
        {:ok, other} -> raise ArgumentError, "EntityExtractor: :preferences must be a SlotTypePreferences model, got #{inspect(other, limit: 3)}"
        :error -> SlotTypePreferences.loaded!(language)
      end

    labels = span.readings |> Enum.filter(& &1.measured) |> Enum.map(& &1.canonical) |> Enum.uniq()
    evidence = SlotTypePreferences.intent_evidence(preferences, intent, labels)

    case SpanReliability.redecide(span, evidence, reliability, TypeHierarchy) do
      {:decided, entity} ->
        entity = %{entity | decision: Map.put(entity.decision, :resolved_by, {:intent, intent})}
        decided = %{entity_type: entity.entity_type, score: entity.decision.score, margin: entity.decision.margin, evidence: evidence}

        # The floor `extract/2` applies to every entity applies here too; a
        # decision under it is not acted on, and the span stays as it was.
        if entity.confidence >= min_confidence(opts) do
          {:entity, entity, outcome(span, Map.put(decided, :outcome, :resolved))}
        else
          {:span, span, outcome(span, Map.put(decided, :outcome, :below_confidence_floor))}
        end

      {:undecided, %Brain.Lattice{best: %{label: {_s, _e, best}} = top} = lattice} ->
        margin = if lattice.second, do: top.score - lattice.second.score, else: :infinity

        {:span, span,
         outcome(span, %{outcome: :undecided, best_type: best, score: top.score, margin: margin, evidence: evidence})}

      # The extent the readings' extent reliabilities decided carries no
      # measured reading, so there is no type there to weigh the evidence for.
      {:undecided, %Brain.Lattice{best: nil}} ->
        {:span, span, outcome(span, %{outcome: :not_redecided, reason: :no_hypothesis_at_extent})}

      # Evidence about types cannot say where the span starts and ends.
      {:extent_undecided, %Brain.Lattice{}} ->
        {:span, span, outcome(span, %{outcome: :not_redecided, reason: :extent_undecided})}
    end
  end

  defp outcome(span, fields), do: Map.merge(%{value: span.value, start_pos: span.start_pos, end_pos: span.end_pos}, fields)

  defp min_confidence(opts), do: Keyword.get(opts, :min_confidence) || get_min_confidence_threshold()

  @doc """
  Every source's reading of `text`, before any span is decided: the gazetteer,
  the system parsers (numbers, dates), the location and proper-noun hints, and
  the world's entity model. Each is an entity map with its `:source`, its
  `:entity_type`, its extent and the source's own `:confidence`, as
  `Brain.ML.SpanReliability` takes them.

  Takes `extract_entities/2`'s `:world_id`, `:domain` and `:intent`, and
  `:entity_model`, a model of `Brain.ML.EntityTrainer`'s shape used in place of
  the world's loaded one. Training passes each row's out-of-fold model.
  """
  @spec readings(String.t(), keyword()) :: [map()]
  def readings(text, opts) when is_binary(text) and is_list(opts) do
    collect_readings(text, Tokenizer.tokenize(text), opts, world_id!(opts))
  end

  defp collect_readings(text, tokens, opts, world_id) do
    gaz_context = Keyword.take(opts, [:domain, :intent, :world_id])
    tags = required_pos_tags(tokens)

    extract_gazetteer_entities(tokens, tags, text, gaz_context, language!(opts)) ++
      extract_system_entities(tokens, text) ++
      extract_location_hints(tokens, tags) ++
      extract_proper_noun_hints(tokens, tags) ++
      extract_model_entities(text, tokens, entity_model!(opts, world_id))
  end

  defp entity_model!(opts, world_id) do
    case Keyword.fetch(opts, :entity_model) do
      {:ok, %{tag_vocabulary: _} = model} -> model
      {:ok, other} -> raise ArgumentError, "EntityExtractor option :entity_model must be an entity model, got: #{inspect(other, limit: 3)}"
      :error -> EntityModel.loaded!(world_id)
    end
  end

  # The language whose span reliabilities decide the spans. Absent means the
  # caller named no language, which is English; an unsupported code raises.
  defp language!(opts) do
    case Keyword.fetch(opts, :language) do
      {:ok, code} -> Brain.Language.validate!(code)
      :error -> Brain.Language.resolve!(nil)
    end
  end

  # Entities are nomens, so extraction always runs in a world: the caller names
  # it, and there is no default.
  defp world_id!(opts) do
    case Keyword.fetch(opts, :world_id) do
      {:ok, world_id} when is_binary(world_id) and world_id != "" ->
        world_id

      {:ok, other} ->
        raise ArgumentError, "EntityExtractor option :world_id must be a world id, got: #{inspect(other)}"

      :error ->
        raise ArgumentError, "EntityExtractor needs :world_id, the world whose entities are extracted"
    end
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

          other ->
            raise "EntityExtractor: #{path} has no \"system_to_canonical\" object: #{inspect(other, limit: 5)}"
        end

      {:error, reason} ->
        raise "EntityExtractor: cannot read #{path}: #{inspect(reason)}"
    end
  end

  @doc """
  Refines entity types based on intent context and slot schemas.

  After intent classification, numeric entities can be refined to specific
  slot types (final_value, change_value, etc.) based on what the intent's
  slot schema expects. Person entities can be split into given_name/last_name.

  When the schema has several numeric slots, numbers fill them in the order
  they stand in the text. A number anaphora resolution supplied has no place
  in the text: it fills the slot of a schema with one numeric slot, and keeps
  its type when there are several.

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

  # Numbers fill several numeric slots in the order they stand in the text. A
  # referent anaphora resolution supplied (`source: :anaphora_resolution`)
  # stands nowhere in the text, so which slot it fills cannot be told from
  # that order: it keeps its type, and the numbers that have a position are
  # assigned without it. Any other number without a position raises.
  defp assign_numbers_to_slots(numbers, slots) do
    {referents, positioned} = Enum.split_with(numbers, &(Map.get(&1, :source) == :anaphora_resolution))

    assigned =
      positioned
      |> Enum.sort_by(& &1.start_pos)
      |> Enum.with_index()
      |> Enum.map(fn {entity, idx} ->
        slot = Enum.at(slots, idx) || List.last(slots)
        Map.put(entity, :entity_type, slot)
      end)

    assigned ++ referents
  end

  defp refine_person_subtypes(entities) do
    Enum.flat_map(entities, fn entity ->
      if entity[:entity_type] == "person" and entity[:source] == :pos_tagger_propn do
        value = entity[:value] || ""
        tokens = String.split(value, " ", trim: true)

        case tokens do
          [single] ->
            [Map.put(entity, :entity_type, "given_name") |> Map.put(:value, single)]

          [first | rest] ->
            last = List.last(rest)
            given = %{entity | entity_type: "given_name", value: first,
                       match: first, confidence: entity[:confidence]}
            family = %{entity | entity_type: "last_name", value: last,
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

        entity_type in ["date_time", "date", "sys_date", "relative_date"] and is_year_only?(value) ->
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
  Maps entity types onto the NER gold standard's labels, for evaluation only.

  Converts system types (e.g., "heating", "music_artist") to the labels the
  gold standard uses (e.g., "device", "artist"). The runtime keeps system
  types, which the intent registry and slot mappings are written in; only the
  NER evaluators call this, when comparing against the gold standard.
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

  # `{kept, withheld}`: a decided entity below the caller's floor is withheld
  # -- turned back into an unresolved span with its readings -- rather than
  # dropped.
  defp apply_confidence_floor(entities, min_confidence, text) when is_float(min_confidence) do
    {kept, below} = Enum.split_with(entities, &(&1.confidence >= min_confidence))
    {kept, Enum.map(below, &SpanReliability.withhold(&1, :below_confidence_floor, text, TypeHierarchy))}
  end

  defp apply_confidence_floor(entities, _min_confidence, _text) do
    {entities, []}
  end

  # Every candidate reading of a span is kept in :types and scored by
  # EntityTypeScorer; the entity takes the most probable reading. The intent,
  # when the caller knows it, supplies the types the context expects.
  # One UD tag per token, from the POS model. Entity scoring weighs a word by
  # its part of speech in the sentence, so a missing model is an error, not
  # a sentence of "X" tags.
  defp required_pos_tags([]), do: []

  defp required_pos_tags(tokens) do
    case POSTagger.load_model() do
      {:ok, model} ->
        tagged = POSTagger.predict(Enum.map(tokens, & &1.text), model)

        unless length(tagged) == length(tokens) do
          raise "EntityExtractor: POS tagger returned #{length(tagged)} tags for #{length(tokens)} tokens"
        end

        Enum.map(tagged, fn {_text, tag} -> tag end)

      {:error, reason} ->
        raise "EntityExtractor: the POS model is required to score entities: #{inspect(reason)}"
    end
  end

  defp extract_gazetteer_entities(tokens, tags, text, gaz_context, language) do
    token_texts =
      Enum.map(tokens, fn t ->
        t.text
        |> String.downcase()
        |> strip_non_word_chars()
        |> String.trim()
      end)

    scoring = [language: language]
    expected_types = EntityTypeScorer.expected_types(gaz_context[:intent], scoring)

    Gazetteer.lookup_spans(token_texts, gaz_context)
    |> Enum.sort_by(fn {start_idx, _end_idx, _info} -> start_idx end)
    |> attach_quantifiers(tokens)
    |> Enum.map(fn {start_idx, end_idx, entity_info, quantifier} ->
      start_token = Enum.at(tokens, start_idx)
      end_token = Enum.at(tokens, end_idx)

      matched_tokens = Enum.slice(tokens, start_idx..end_idx)
      match_text = Enum.map_join(matched_tokens, " ", & &1.text)

      candidates =
        case entity_info do
          [_ | _] = infos -> infos
          info when is_map(info) -> [info]
        end

      scored =
        %{
          entity_type: nil,
          value: match_text,
          match: match_text,
          start_pos: start_token.start_pos,
          end_pos: end_token.end_pos,
          confidence: nil,
          types: candidates,
          source: :gazetteer,
          # The head word's tag -- the last word of an English noun phrase.
          tag: Enum.at(tags, end_idx),
          # The tag of the word before the span, as the slot type preferences
          # were learned from it.
          preceding_tag: if(start_idx == 0, do: SlotTypePreferences.sentence_start(), else: Enum.at(tags, start_idx - 1))
        }
        |> EntityTypeScorer.mark_position(text)
        |> EntityTypeScorer.apply_to(expected_types, scoring)

      selected =
        Enum.find(candidates, &(&1[:entity_type] == scored.entity_type and &1[:value] == scored.value)) ||
          Enum.find(candidates, &(&1[:entity_type] == scored.entity_type))

      # The span's grammatical number, read from its last word -- the head
      # of an English noun phrase ("office lights"): :plural, :singular, or
      # nil when WordNet knows no noun it could be.
      number = end_token.text |> String.downcase() |> Brain.Lexicon.grammatical_number()

      scored
      |> Map.put(:number, number)
      |> Map.put(:quantifier, quantifier)
      |> extract_gazetteer_metadata(selected)
    end)
  end

  # Gives each span its quantifier: :total when a total quantifier ("all",
  # "every", "each") quantifies it -- "all the lights", "every room", "all of
  # the lights" -- so the action applies to each of them; nil otherwise.
  #
  # A quantifier stands among the function words before its noun phrase and
  # quantifies the phrase's head, which in English comes last. Spans that
  # directly follow one another form one phrase ("kitchen lights"), so in
  # "all the kitchen lights" the lights are quantified, not the kitchen.
  defp attach_quantifiers(spans, tokens) do
    spans
    |> Enum.chunk_while(
      [],
      fn {start_idx, _e, _i} = span, run ->
        case run do
          [{_s, prev_end, _} | _] when start_idx == prev_end + 1 -> {:cont, [span | run]}
          [] -> {:cont, [span]}
          _ -> {:cont, Enum.reverse(run), [span]}
        end
      end,
      fn
        [] -> {:cont, []}
        run -> {:cont, Enum.reverse(run), []}
      end
    )
    |> Enum.flat_map(fn [{first_start, _, _} | _] = run ->
      quantifier = quantifier_before(tokens, first_start)
      head = length(run) - 1

      run
      |> Enum.with_index()
      |> Enum.map(fn {{s, e, info}, i} -> {s, e, info, if(i == head, do: quantifier)} end)
    end)
  end

  # :total when a total quantifier stands among the function words directly
  # before token `start_idx`; the run stops at the first open-class word.
  defp quantifier_before(tokens, start_idx) do
    tokens
    |> Enum.take(start_idx)
    |> Enum.reverse()
    |> Enum.take_while(&Brain.Lexicon.ClosedClass.member?(&1.text))
    |> Enum.any?(&Brain.Lexicon.ClosedClass.total_quantifier?(&1.text))
    |> if(do: :total, else: nil)
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
          start = entity_idx + 1
          stop = min(entity_idx + 3, length(token_texts) - 1)

          if start <= stop do
            next_tokens = Enum.slice(token_texts, start..stop//1)
            Enum.any?(next_tokens, fn t -> t in age_indicators end)
          else
            false
          end
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

  # Reads the tags the sentence was already tagged with. Asking the tagger
  # about one word at a time cost 85-92% of extraction: a single-token call is
  # padded to the same batch as a 32-sentence one, so N words cost N times a
  # full sentence.
  defp extract_location_hints(tokens, tags) do
    prep_tag = TypeHierarchy.config(["pos_tag_roles", "preposition_tag"], "ADP")

    tags
    |> Enum.with_index()
    |> Enum.flat_map(fn {tag, idx} ->
      if tag == prep_tag do
        extract_following_location(tokens, tags, idx + 1)
      else
        []
      end
    end)
  end

  defp extract_following_location(tokens, tags, start_idx) do
    remaining = Enum.drop(tokens, start_idx)
    function_tags = TypeHierarchy.config(["pos_tag_roles", "function_word_tags"], [])

    location_tokens =
      remaining
      |> Enum.zip(Enum.drop(tags, start_idx))
      |> Enum.take_while(fn {token, tag} ->
        capitalized?(token.text) and tag not in function_tags
      end)
      |> Enum.map(&elem(&1, 0))

    if location_tokens != [] do
      location_text = Enum.map_join(location_tokens, " ", & &1.text)
      normalized = String.downcase(location_text)

      unless in_gazetteer?(normalized) do
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
            confidence: 0.7,
            source: :location_hint
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
      # A place name that is not a street address is typed as what a city is.
      case TypeHierarchy.parents_of("city") do
        [parent] -> parent
        parents -> raise "EntityExtractor: a place name needs the one parent type of city; the hierarchy gives #{inspect(parents)}"
      end
    end
  end

  defp extract_proper_noun_hints(tokens, tags) do
    extract_with_pos_tagger(tokens, tags)
  end

  # The world's BIO tagger over the same tokens. It was trained on
  # `Tokenizer.tokenize/1` tokens (`Brain.ML.NerGold`), so its token indexes map
  # straight back to positions in the text, and its confidence is the mean
  # posterior of the entity's tags. Its `:extent_cue` says how the decode
  # delimited it (`bio_extent_cue/2`), which `Brain.ML.SpanReliability` learns
  # how far to trust its extent by.
  defp extract_model_entities(text, tokens, model) do
    tagged = tokens |> Enum.map(& &1.text) |> EntityTrainer.tag(model)

    tagged
    |> EntityTrainer.extract_entities_from_bio()
    |> Enum.map(fn entity ->
      start_pos = Enum.at(tokens, entity.start_index).start_pos
      end_pos = Enum.at(tokens, entity.end_index).end_pos
      surface = String.slice(text, start_pos, end_pos - start_pos + 1)

      %{
        entity_type: entity.entity_type,
        value: surface,
        match: surface,
        start_pos: start_pos,
        end_pos: end_pos,
        confidence: entity.confidence,
        source: :entity_model,
        extent_cue: bio_extent_cue(tagged, entity)
      }
    end)
  end

  # `{opened, followed_by}`: `:begin` when the decode opened the entity at a
  # `B-` tag and `:inside` when at an `I-` tag (one that continues no entity of
  # its type); `:inside` when the token after the entity is tagged `I-` (the
  # decode ran on into another type) and `:outside` when it is tagged
  # otherwise or the sentence ends.
  defp bio_extent_cue(tagged, %{start_index: start_index, end_index: end_index}) do
    opened =
      case Enum.at(tagged, start_index) do
        {_token, "B-" <> _type, _posterior} -> :begin
        {_token, "I-" <> _type, _posterior} -> :inside
      end

    followed_by =
      case Enum.at(tagged, end_index + 1) do
        {_token, "I-" <> _type, _posterior} -> :inside
        {_token, "B-" <> _type, _posterior} -> :outside
        {_token, "O", _posterior} -> :outside
        nil -> :outside
      end

    {opened, followed_by}
  end

  # Takes the sentence's tags rather than tagging it a second time: this and
  # `required_pos_tags/1` were tagging the identical token list on every call.
  defp extract_with_pos_tagger(tokens, tags) do
    propn_tag = TypeHierarchy.config(["pos_tag_roles", "proper_noun"], "PROPN")
    default_type = TypeHierarchy.config("default_propn_type", "person")

    tokens
    |> Enum.map(& &1.text)
    |> Enum.zip(tags)
    |> Enum.with_index()
    |> Enum.reduce([], fn {{word, tag}, idx}, acc ->
      if tag == propn_tag and capitalized?(word) and not in_gazetteer?(word) do
        token = Enum.at(tokens, idx)

        case acc do
          [%{source: :pos_tagger_propn, _merge_end_idx: prev_idx} = prev | rest]
          when prev_idx == idx - 1 ->
            merged = %{
              prev
              | value: prev.value <> " " <> word,
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
  end

  defp in_gazetteer?(word) when is_binary(word) do
    Gazetteer.lookup(String.downcase(word)) != :not_found
  end

  defp extract_gazetteer_metadata(base, info) when is_map(info) do
    metadata_keys = [:ha_entity_id, :ha_domain, :state_code, :state_name, :region, :country, :county]

    metadata =
      metadata_keys
      |> Enum.reduce(%{}, fn key, acc ->
        case Map.get(info, key) do
          nil -> acc
          val -> Map.put(acc, key, val)
        end
      end)

    if map_size(metadata) > 0 do
      Map.put(base, :metadata, metadata)
    else
      base
    end
  end

  defp capitalized?(text) do
    first = String.first(text)
    first != nil and first == String.upcase(first) and first != String.downcase(first)
  end

  # Adjacent decided spans of compatible types become one entity carrying both
  # spans' readings.
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
            confidence: min(0.85, max(prev.confidence, entity.confidence) + 0.05),
            readings: prev.readings ++ entity.readings
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

  # The profile, when one is known, is the chunk's: the disambiguator ranks the
  # slot types of its domain, and `context_type/1` reads its domain.
  defp disambiguate_entities(entities, tokens, %{discourse: discourse, speech_act: speech_act, profile: profile}, original_text, world_id) do
    debug? = Application.get_env(:brain, :debug_pipeline_timing, false)
    t0 = if debug?, do: System.monotonic_time(:millisecond)

    classified_intent = extract_intent_from_speech_act(speech_act)

    context = %{
      discourse: discourse,
      speech_act: speech_act,
      profile: profile,
      intent: classified_intent,
      original_text: original_text,
      world_id: world_id
    }

    pos_tagged = get_pos_tags(tokens)

    if debug?, do: Logger.info("      disambig:pos_tags=#{System.monotonic_time(:millisecond) - t0}ms (#{length(entities)} entities, intent=#{classified_intent})")

    disambiguated =
      entities
      |> Enum.with_index()
      |> Enum.map(fn {entity, idx} ->
        types = get_entity_types(entity)
        entity_type = entity[:entity_type] || ""

        needs_disambiguation =
          length(types) > 1 or EntityDisambiguator.requires_inference?(entity_type)

        if needs_disambiguation do
          et0 = if debug?, do: System.monotonic_time(:millisecond)

          result = EntityDisambiguator.disambiguate_single(entity, pos_tagged, context)

          if debug?, do: Logger.info("      disambig:entity[#{idx}] disambiguate_single=#{System.monotonic_time(:millisecond) - et0}ms val=#{entity[:value]} type=#{entity_type} types=#{length(types)}")

          pt0 = if debug?, do: System.monotonic_time(:millisecond)

          result = refine_with_poincare(result, types, context)

          if debug?, do: Logger.info("      disambig:entity[#{idx}] poincare=#{System.monotonic_time(:millisecond) - pt0}ms types=#{length(types)}")

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
          if debug?, do: Logger.info("      disambig:entity[#{idx}] skip (no ambiguity) val=#{entity[:value]} type=#{entity_type}")
          entity
        end
      end)

    if debug?, do: Logger.info("      disambig:total=#{System.monotonic_time(:millisecond) - t0}ms")

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

  # Refines the type with the world's Poincare embeddings. A world with no copy
  # of the embeddings raises.
  defp refine_with_poincare(result, candidate_types, %{world_id: world_id} = context)
       when length(candidate_types) > 1 do
    debug? = Application.get_env(:brain, :debug_pipeline_timing, false)

    intent = context[:intent] || ""
    current_type = to_string(result[:entity_type] || "")
    pt0 = if debug?, do: System.monotonic_time(:millisecond)

    scored =
      candidate_types
      |> Enum.map(fn candidate ->
        type_str = to_string(candidate[:entity_type] || candidate[:type] || "")
        dist = poincare_distance(world_id, type_str, intent)
        {type_str, dist, candidate}
      end)
      |> Enum.filter(fn {_t, d, _c} -> d != nil end)
      |> Enum.sort_by(fn {_t, d, _c} -> d end)

    if debug?, do: Logger.info("        poincare:scored #{length(candidate_types)} candidates in #{System.monotonic_time(:millisecond) - pt0}ms")

    case scored do
      [{best_type, best_dist, _best_candidate} | _] when best_type != current_type ->
        current_dist = poincare_distance(world_id, current_type, intent)

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
  end

  defp refine_with_poincare(result, _types, _context), do: result

  # nil when either type has no embedding in the world's copy.
  defp poincare_distance(world_id, type1, type2) do
    case Brain.ML.Poincare.Embeddings.entity_distance(world_id, type1, type2) do
      {:ok, dist} -> dist
      {:error, :not_found} -> nil
    end
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
end
