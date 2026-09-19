defmodule Brain.Analysis.EntityTypeScorer do
  @moduledoc """
  Decides what a gazetteer match means, and how sure to be that it is an
  entity at all.

  A matched span such as "thursday" can have several readings: the day
  (`sys_date`), the band (`music_artist`), or no entity -- just an ordinary
  word, as "nice" is in "have a nice day". Each reading gets a posterior:

      P(reading | span, context)  ∝  prior × casing likelihood × context likelihood

  ## Prior -- how the word is used

  Read from the brain's own lexicon: `sense_usage` facts count how often the
  word is used as each entity type, `ordinary_usage` facts how often as a
  plain noun or as another part of speech. SemCor seeds them
  (`Brain.Lexicon.Seeder`); any other source writing the same facts --
  running statistics from conversations -- adds to the same counts.

  - An entity type counts its own uses. A candidate whose stored value is
    lowercase is a common word that a curated file lists in its everyday
    meaning -- a room file's "office", a date file's "later" -- so it also
    counts every ordinary use of the word, whatever its part of speech.
    Every type also gets the `prior_smoothing` pseudo-count, so an unseen
    type is unlikely, not impossible.
  - Ordinary word counts the word's uses as entity types no candidate
    offers, and -- when every candidate is a proper name (stored
    capitalized) -- its ordinary uses. It gets no pseudo-count: a name
    WordNet has never seen is not thereby half likely to be a plain word.

  ## Casing

  Uninformative on the first word of a sentence, and for common-word
  candidates. Elsewhere a proper name is typed capitalized with probability
  `proper_name_capitalized`, and an ordinary word with probability
  `ordinary_word_capitalized`.

  ## Context

  When the intent is known and its slot schema lists entity types, a reading
  whose type is listed has likelihood 1 and every other reading -- ordinary
  word included -- has `context_mismatch_likelihood`. Listed means listed:
  the schema names the types it accepts, and a subtype the hierarchy would
  allow (a `music_artist` is a `person`) is not thereby as likely to fill
  the slot. With no listed types, context says nothing.

  ## Result

  The selected type is the most probable entity reading. The confidence is
  the match's base confidence times the probability that the span is an
  entity at all, so two readings of one entity (a door as `device` or as
  `lock`) do not halve each other's confidence.

  The constants live in `config :brain, :entity_type_scoring`.
  """

  alias Brain.Analysis.{SlotDetector, TypeHierarchy}
  alias Brain.Lexicon.UserDefined

  @ordinary :ordinary

  @type evidence :: %{
          required(:match) => String.t(),
          required(:sentence_initial) => boolean(),
          optional(:expected_types) => [String.t()] | nil
        }

  @type result :: %{
          selected: map(),
          entity_type: String.t(),
          confidence: float(),
          p_ordinary: float(),
          posteriors: %{String.t() => float()}
        }

  @doc """
  Scores the candidate readings of one matched span.

  `candidates` are gazetteer infos, each with `:entity_type` and `:value`.
  `evidence` carries the matched text as typed, whether it starts a
  sentence, and the entity types the intent expects (`nil` or `[]` when
  there is no context).

  ## Options

  - `:usage` -- `%{types: %{type => count}, ordinary: %{"noun" | "other_pos"
    => count}}` to use instead of reading the lexicon.
  - `:lexicon` -- the `Brain.Lexicon.UserDefined` store to read usage from.
  - `:config` -- keyword list replacing `config :brain, :entity_type_scoring`.
  """
  @spec score([map()], evidence(), keyword()) :: result()
  def score(candidates, evidence, opts \\ [])

  def score([], evidence, _opts) do
    raise ArgumentError, "EntityTypeScorer: no candidates to score for #{inspect(evidence.match)}"
  end

  def score(candidates, %{match: match, sentence_initial: initial?} = evidence, opts)
      when is_binary(match) and is_boolean(initial?) do
    config = Keyword.get_lazy(opts, :config, &config!/0)
    alpha = Keyword.fetch!(config, :prior_smoothing)
    epsilon = Keyword.fetch!(config, :context_mismatch_likelihood)

    casing = %{
      name: Keyword.fetch!(config, :proper_name_capitalized),
      ordinary: Keyword.fetch!(config, :ordinary_word_capitalized)
    }

    surface = String.downcase(match)
    usage = Keyword.get_lazy(opts, :usage, fn -> usage(surface, Keyword.get(opts, :lexicon, UserDefined)) end)
    by_type = one_candidate_per_type(candidates, match)
    expected = Map.get(evidence, :expected_types) || []

    counts = prior_counts(by_type, usage)
    typed_capitalized? = capitalized?(match)

    weights =
      Map.new(counts, fn {reading, count} ->
        prior = if reading == @ordinary, do: count, else: count + alpha
        casing_l = casing_likelihood(reading, by_type, typed_capitalized?, initial?, casing)
        context_l = context_likelihood(reading, expected, epsilon)
        {reading, prior * casing_l * context_l}
      end)

    total = weights |> Map.values() |> Enum.sum()
    posteriors = Map.new(weights, fn {reading, w} -> {reading, w / total} end)
    p_ordinary = Map.fetch!(posteriors, @ordinary)

    {type, selected} = select(by_type, posteriors)

    %{
      selected: selected,
      entity_type: type,
      confidence: base_confidence(match, type) * (1.0 - p_ordinary),
      p_ordinary: p_ordinary,
      posteriors: Map.delete(posteriors, @ordinary)
    }
  end

  @doc """
  Records on the entity whether it starts a sentence of `text`, the text its
  `:start_pos` indexes into. Kept on the entity because later stages may
  hold a different text than the one it was extracted from. An entity that
  already records it is returned unchanged.
  """
  @spec mark_position(map(), String.t()) :: map()
  def mark_position(%{sentence_initial: initial?} = entity, _text) when is_boolean(initial?), do: entity

  def mark_position(%{start_pos: start_pos} = entity, text) when is_binary(text) do
    Map.put(entity, :sentence_initial, sentence_initial?(text, start_pos))
  end

  @doc """
  Scores an entity whose `:types` hold its candidates and applies the
  result: the entity takes the selected type, value and confidence, and
  records its posteriors. The candidates stay in `:types`, so the entity can
  be scored again once the intent is known. The entity must carry
  `:sentence_initial` (see `mark_position/2`).
  """
  @spec apply_to(map(), [String.t()] | nil, keyword()) :: map()
  def apply_to(entity, expected_types, opts \\ [])

  def apply_to(%{types: [_ | _] = candidates, match: match, sentence_initial: initial?} = entity, expected_types, opts)
      when is_boolean(initial?) do
    result =
      score(
        candidates,
        %{match: match, sentence_initial: initial?, expected_types: expected_types},
        opts
      )

    %{
      entity
      | entity_type: result.entity_type,
        value: Map.get(result.selected, :value, entity.value),
        confidence: result.confidence
    }
    |> Map.put(:type_posteriors, result.posteriors)
    |> Map.put(:p_ordinary, result.p_ordinary)
  end

  def apply_to(entity, _expected_types, _opts) do
    raise ArgumentError,
          "EntityTypeScorer: cannot score an entity without :types, :match and " <>
            ":sentence_initial: #{inspect(entity)}"
  end

  @doc """
  Scores again every entity that carries candidates, now that the intent is
  known. Entities without candidates -- numbers, dates parsed from text --
  are returned unchanged.
  """
  @spec rescore([map()], String.t() | atom() | nil, keyword()) :: [map()]
  def rescore(entities, intent, opts \\ []) when is_list(entities) do
    expected = expected_types(intent)

    Enum.map(entities, fn
      %{types: [_ | _]} = entity -> apply_to(entity, expected, opts)
      entity -> entity
    end)
  end

  @doc """
  Returns the entity types an intent's slot schema expects, sorted, or `nil`
  when there is no intent -- no context to weigh. An intent whose schema
  maps no entity types returns `[]`, which likewise says nothing.
  """
  @spec expected_types(String.t() | atom() | nil) :: [String.t()] | nil
  def expected_types(nil), do: nil
  def expected_types(""), do: nil

  def expected_types(intent) do
    intent
    |> to_string()
    |> SlotDetector.get_entity_types_for_intent()
    |> MapSet.to_list()
    |> Enum.sort()
  end

  @doc """
  True when the span starting at grapheme offset `start_pos` is the first
  word of a sentence in `text`.
  """
  @spec sentence_initial?(String.t(), non_neg_integer()) :: boolean()
  def sentence_initial?(text, start_pos) when is_binary(text) and is_integer(start_pos) do
    prefix = text |> String.slice(0, start_pos) |> String.trim_trailing()
    prefix == "" or String.ends_with?(prefix, [".", "!", "?"])
  end

  @doc """
  Base confidence of a match of `match_text` read as `entity_type`, before
  weighing whether it is an entity at all: longer matches are likelier to be
  deliberate, and `confidence_adjustments` in entity_types.json adjusts per
  type.
  """
  @spec base_confidence(String.t(), String.t()) :: float()
  def base_confidence(match_text, entity_type) do
    length_score = min(0.9, 0.5 + String.length(match_text) * 0.03)
    type_bonus = Map.get(confidence_adjustments(), entity_type, 0.0)
    min(0.95, max(0.1, length_score + type_bonus))
  end

  @doc """
  Reads a word's usage counts from the brain's lexicon, summed over every
  source that records them.

  A single word's usage is that of every lemma it can be a form of
  (`Brain.Lexicon.base_forms/1`): "lights" is used as "light" is, so a band
  called Lights competes with every ordinary use of the word. Usage is
  recorded per lemma, and an inflected form has none of its own. A span of
  several words is looked up as written.
  """
  @spec usage(String.t(), GenServer.server()) :: %{types: map(), ordinary: map()}
  def usage(surface, store \\ UserDefined) when is_binary(surface) do
    words = lexemes(String.downcase(surface))

    facts = fn key ->
      Enum.flat_map(words, &UserDefined.facts(&1, [kind: "property", key: key], store))
    end

    %{types: sum_counts(facts.("sense_usage")), ordinary: sum_counts(facts.("ordinary_usage"))}
  end

  defp lexemes(surface) do
    if String.contains?(surface, " ") do
      [surface]
    else
      bases = for {lemma, _pos} <- Brain.Lexicon.base_forms(surface), do: lemma
      Enum.uniq([surface | bases])
    end
  end

  defp sum_counts(facts) do
    Enum.reduce(facts, %{}, fn fact, acc ->
      count =
        case fact.value do
          %{"count" => n} when is_integer(n) and n >= 0 ->
            n

          other ->
            raise "EntityTypeScorer: #{fact.key} fact for #{inspect(fact.word)} " <>
                    "(#{fact.source}) has no count: #{inspect(other)}"
        end

      Map.update(acc, fact.ref, count, &(&1 + count))
    end)
  end

  # Several gazetteer entries can share a type ("Thursday" the band from two
  # artist lists). The reading is the type; the entry kept is the one whose
  # stored value matches how the span was typed, then the first listed.
  defp one_candidate_per_type(candidates, match) do
    candidates
    |> Enum.with_index()
    |> Enum.group_by(fn {c, _i} -> candidate_type!(c) end)
    |> Map.new(fn {type, indexed} ->
      {best, _i} = Enum.min_by(indexed, fn {c, i} -> {if(Map.get(c, :value) == match, do: 0, else: 1), i} end)
      {type, best}
    end)
  end

  defp candidate_type!(candidate) do
    case Map.get(candidate, :entity_type) do
      type when is_binary(type) and type != "" -> type
      other -> raise ArgumentError, "EntityTypeScorer: candidate has no entity type (#{inspect(other)}): #{inspect(candidate)}"
    end
  end

  defp prior_counts(by_type, usage) do
    type_counts = Map.get(usage, :types, %{})
    ordinary_counts = Map.get(usage, :ordinary, %{})
    ordinary_uses = ordinary_counts |> Map.values() |> Enum.sum()

    common_word_candidate? = Enum.any?(by_type, fn {_t, c} -> not proper_name?(c) end)

    entity_counts =
      Map.new(by_type, fn {type, candidate} ->
        own = Map.get(type_counts, type, 0)
        {type, if(proper_name?(candidate), do: own, else: own + ordinary_uses)}
      end)

    unoffered =
      type_counts
      |> Enum.reject(fn {type, _n} -> Map.has_key?(by_type, type) end)
      |> Enum.map(fn {_type, n} -> n end)
      |> Enum.sum()

    ordinary = unoffered + if(common_word_candidate?, do: 0, else: ordinary_uses)

    Map.put(entity_counts, @ordinary, ordinary)
  end

  defp casing_likelihood(_reading, _by_type, _typed_capitalized?, true = _initial?, _casing), do: 1.0

  defp casing_likelihood(@ordinary, _by_type, typed_capitalized?, false, casing),
    do: capitalization_probability(casing.ordinary, typed_capitalized?)

  defp casing_likelihood(type, by_type, typed_capitalized?, false, casing) do
    if proper_name?(Map.fetch!(by_type, type)),
      do: capitalization_probability(casing.name, typed_capitalized?),
      else: 1.0
  end

  defp capitalization_probability(p_capitalized, true), do: p_capitalized
  defp capitalization_probability(p_capitalized, false), do: 1.0 - p_capitalized

  defp context_likelihood(_reading, [], _epsilon), do: 1.0
  defp context_likelihood(@ordinary, _expected, epsilon), do: epsilon
  defp context_likelihood(type, expected, epsilon), do: if(type in expected, do: 1.0, else: epsilon)

  # Highest posterior wins. Ties go to the larger confidence adjustment, then
  # to the type name, so the choice never depends on map or list order.
  defp select(by_type, posteriors) do
    adjustments = confidence_adjustments()

    by_type
    |> Enum.min_by(fn {type, _c} ->
      {-Map.fetch!(posteriors, type), -Map.get(adjustments, type, 0.0), type}
    end)
  end

  defp proper_name?(candidate), do: capitalized?(to_string(Map.get(candidate, :value, "")))

  defp capitalized?(text) do
    first = String.first(text)
    first != nil and first == String.upcase(first) and first != String.downcase(first)
  end

  defp confidence_adjustments do
    case TypeHierarchy.config("confidence_adjustments") do
      adjustments when is_map(adjustments) ->
        adjustments

      other ->
        raise "EntityTypeScorer: entity_types.json config.confidence_adjustments " <>
                "is not loaded (got #{inspect(other)})"
    end
  end

  defp config! do
    case Application.fetch_env(:brain, :entity_type_scoring) do
      {:ok, config} -> config
      :error -> raise "EntityTypeScorer: config :brain, :entity_type_scoring is not set"
    end
  end
end
