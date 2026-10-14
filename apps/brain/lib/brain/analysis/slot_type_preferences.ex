defmodule Brain.Analysis.SlotTypePreferences do
  @moduledoc """
  Which entity types fill an intent's slots, how often a gazetteer match in
  the intent's utterances is no entity at all, and which part of speech comes
  before a span of each type, learned from the gold standards' training
  splits.

  ## The data

  The NER gold standard gives each row's entity spans with their gold labels,
  placed by `Brain.ML.NerGold.spans!/1` on `Brain.ML.Tokenizer` tokens. The
  intent gold standard gives each text its intent. Joined on the text, every
  gold span is a labeled `{intent, type}` pair. Every token is tagged as
  `Brain.ML.EntityExtractor` tags it (`Brain.ML.POSTagger` over the
  tokenizer's tokens), so each span also gives the UD tag of the token before
  it -- `:sentence_start` for a span that begins the text -- and every token
  outside the spans gives the tag before an ordinary word.

  Every row is also read by the gazetteer, as the extractor reads it (one
  `Brain.ML.EntityExtractor.gazetteer_key/1` per token, looked up with
  `Brain.ML.Gazetteer.lookup_spans/2`), before any reading is scored. A
  gazetteer span that no gold span overlaps -- `Brain.ML.SpanReliability`'s
  `:no_entity` outcome -- is an ordinary word the gazetteer happens to list;
  one that a gold span overlaps is an entity.

  Only training rows are read: a NER training row is joined to an intent only
  when the intent row is a training row too, the slot and gazetteer counts
  come from the joined rows, and the tag counts come from the NER training
  rows whether or not they join.

  ## Slot-fill preferences

  The label universe `U` holds every gold label, every type of the static
  hierarchy, and the gold label of every type a slot schema lists or the
  gazetteer can read a span as. For an
  intent `i` in consolidated domain `d` (`Brain.Lexicon.IntentDomains`), the
  probability that a gold span of the utterance has label `g` is smoothed
  through three levels:

      global:  P0(g) = (c(g)   + k0 / |U|)  / (N   + k0)
      domain:  Pd(g) = (c(d,g) + kd * P0(g)) / (n_d + kd)
      intent:  Pi(g) = (c(i,g) + ki * Pd(g)) / (n_i + ki)

  and then through the static parents of `Brain.Analysis.TypeHierarchy`. A
  parent's subtree mass `M(p)` is the sum of `Pi` over `p` and its
  descendants; a type with parents inherits from each an equal share of it,
  divided among the parent's children:

      H(g) = mean over the parents p of g of M(p) / |children(p)|   (Pi(g) for a root)
      P(g) proportional to (1 - lambda) * Pi(g) + lambda * H(g)

  so a child type the data rarely shows for the intent receives part of what
  its family receives there. A domain on its own (a `ChunkProfile.domain`)
  uses `Pd` where an intent uses `Pi`, and every utterance together (`:global`)
  uses `P0`.

  ## What an intent says about a span's label

  For a label `L`, the mass `M(L)` of a distribution is the probability of `L`
  and of every label that refines it. An intent's evidence about `L` is

      log( M_intent(L) / M_global(L) )

  how much likelier a gold span of the intent's utterances is to agree with
  `L` than a gold span of any utterance. Read as a likelihood ratio, it is what
  the intent adds, in log-odds, to evidence about the span that is
  independent of the intent (`Brain.ML.SpanReliability.redecide/4`).

  ## Ordinary words

  The probability that a gazetteer span in a context's utterances is no
  entity, smoothed through the same levels as the labels, from an even split
  of the two outcomes (`o(c)` counts the ordinary spans of `c`, `m(c)` all its
  gazetteer spans):

      global:  O0   = (o     + q0 / 2)       / (m     + q0)
      domain:  Od   = (o(d)  + qd * O0)      / (m(d)  + qd)
      intent:  Oi   = (o(i)  + qi * Od)      / (m(i)  + qi)

  ## The context's likelihood of a reading

  A gazetteer span has readings: each candidate entity type `t` and the
  ordinary word `o`. `EntityTypeScorer` weighs each by how the word is used
  (its lexicon prior, `P(r | word)`), which already holds how often readings
  of the word are entities at all. What the context adds must not count that
  base rate again. Taking the context `c` to bear on the span only through its
  reading,

      P(r | word, c)  ∝  P(r | word) × P(c | r)  ∝  P(r | word) × P(r | c) / P(r)

  so the context's factor is the probability of the reading in the context
  over its probability across every utterance (`reading_likelihoods/2`):

      ordinary:  K(o) = O(c) / O0
      type t:    K(t) = (1 - O(c)) × M_c(t) / ((1 - O0) × M_global(t))

  `M_c(t)` is the mass of the label of `t` under the context's distribution
  (the label and every label refining it), so a city reading takes what the
  data gives cities under the context, and `O(c)` is `Oi`, `Od` or `O0`.

  The slot schema splits that mass in two. With `A` the labels that agree with
  some type the schema lists (`weights/3`), the probability that a gold span of
  the context agrees with no listed type is

      u(c) = sum of P(g) over the labels g outside A          (`unlisted_rate/3`)

  and `M_c(t)` is the listed part plus the unlisted part,

      M_c(t) = (1 - u(c)) × P(t | listed, c) + u(c) × P(t | unlisted, c)

  with each part the context's distribution over `t`'s labels inside or outside
  `A`, renormalized. Both parts come from the smoothed distribution above, so a
  reading the schema does not list has the likelihood the data gives it -- a
  band under a weather intent is rare, and a city under one listing `location`
  is common -- not a fixed penalty.

  ## Weights for the types a schema lists

  A runtime type is compared to the gold labels through the NER type mappings
  (`canonical`). A listed type agrees with a gold label that is its label or
  refines it in the hierarchy, as `Brain.ML.SpanReliability` counts
  agreement. Each gold label's probability goes to the most specific listed
  labels that agree with it: with `location` and `room` both listed, a gold
  `room` goes to `room` and a gold `address` to `location`. A listed type's
  weight is what its label receives: the probability that a span filling the
  context's slots is read best as that type. Weights rank types; they are not
  normalized, and every one is positive.

  ## Preceding tag

  The likelihood of the tag before a span, for each label of `U` and for an
  ordinary word, smoothed toward the distribution over every observation (`T`
  is the tag vocabulary):

      global:   Pg(t)    = (c(t)    + kg / |T|)          / (N      + kg)
      label:    P(t | g) = (c*(g,t) + kt * Q(t | g))     / (n*(g)  + kt)
      ordinary: P(t | o) = (c(o,t)  + kt * Pg(t))        / (n(o)   + kt)

  `c*` pools a label's descendants, and `Q` is the mean of its parents'
  distributions (`Pg` for a root). A type outside `U` has no counts and no
  parents, so its distribution is `Pg`, as the formula gives for any root with
  no counts.

  ## Choosing the smoothing

  `select/2` chooses `k0`, `kd`, `ki` and `lambda` in that order, then `kg`
  and `kt`, then `q0`, `qd` and `qi`: each is the value of its grid with the
  highest #{5}-fold cross-validated log-likelihood of the held-out folds'
  observations (span labels given the intent; preceding tags given the label
  or the ordinary reading; whether each gazetteer span is ordinary given the
  intent), with the earlier choices fixed. Ties go to the smaller value. The
  held-out splits are never read.

  ## Language

  A model is fitted for one language and stored per language
  (`slot_type_preferences.<language>.term`, directly under the models
  directory, so run provenance hashes it). `Brain.Analysis.SlotTypePreferences.Loader`
  loads it at boot, and boot fails without it. The gold standards are English.
  """

  alias Brain.Lexicon.IntentDomains
  alias Brain.ML.{NerGold, Tokenizer}

  @enforce_keys [
    :language,
    :universe,
    :parents,
    :children,
    :ancestors,
    :descendants,
    :canonical,
    :tag_vocabulary,
    :slot_counts,
    :tag_counts,
    :ordinary_counts,
    :smoothing,
    :tag_likelihoods,
    :fitted
  ]
  defstruct @enforce_keys ++ [:cross_validation]

  @type context :: {:intent, String.t()} | {:domain, String.t()} | :global

  @type t :: %__MODULE__{
          language: String.t(),
          universe: [String.t()],
          parents: %{String.t() => [String.t()]},
          children: %{String.t() => [String.t()]},
          ancestors: %{String.t() => MapSet.t()},
          descendants: %{String.t() => MapSet.t()},
          canonical: %{String.t() => String.t()},
          tag_vocabulary: [String.t() | :sentence_start],
          slot_counts: map(),
          tag_counts: map(),
          ordinary_counts: map(),
          smoothing: map(),
          tag_likelihoods: map(),
          fitted: map(),
          cross_validation: map() | nil
        }

  @sentence_start :sentence_start

  # Pseudo-counts from far below one observation to far above the training
  # split's size, and hierarchy weights from none to all. Cross-validation
  # picks within them.
  @strengths [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000]
  @hierarchy_weights [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  @folds 5

  @smoothing_keys [
    :global,
    :domain,
    :intent,
    :hierarchy,
    :tag_global,
    :tag_type,
    :ordinary_global,
    :ordinary_domain,
    :ordinary_intent
  ]

  @ordinary_keys [:ordinary_global, :ordinary_domain, :ordinary_intent]

  @doc "Every smoothing value the model carries, in the order `select/2` chooses them."
  @spec smoothing_keys() :: [atom()]
  def smoothing_keys, do: @smoothing_keys

  @doc "The marker that stands for the tag before a span that begins the text."
  @spec sentence_start() :: :sentence_start
  def sentence_start, do: @sentence_start

  # -- building examples --------------------------------------------------------

  @doc """
  The intent of each text of the intent gold rows given. A text labeled with
  two different intents raises: joining it would be a guess.
  """
  @spec intents_by_text!([map()]) :: %{String.t() => String.t()}
  def intents_by_text!(rows) when is_list(rows) do
    Enum.reduce(rows, %{}, fn
      %{"text" => text, "intent" => intent}, acc when is_binary(text) and is_binary(intent) ->
        case Map.fetch(acc, text) do
          :error ->
            Map.put(acc, text, intent)

          {:ok, ^intent} ->
            acc

          {:ok, other} ->
            raise ArgumentError,
                  "SlotTypePreferences: #{inspect(text)} is labeled both #{other} and #{intent} in the intent gold standard"
        end

      row, _acc ->
        raise ArgumentError, "SlotTypePreferences: an intent gold row needs a text and an intent, got #{inspect(row)}"
    end)
  end

  @doc """
  One training or evaluation example from a NER gold row: its gold spans with
  the tag before each, the tags before every token outside the spans, the
  outcome of every gazetteer span (`:ordinary` when no gold span overlaps it,
  `:entity` when one does), and `intent` (`nil` when the row joins no usable
  intent row).

  `tag` takes the row's token texts and returns one UD tag per token, as
  `Brain.ML.EntityExtractor` tags them. `gazetteer` takes the same token texts
  and returns the gazetteer's spans over them as `{first, last}` token
  indexes, as `Brain.ML.Gazetteer.lookup_spans/2` finds them for the
  extractor.
  """
  @spec example!(map(), String.t() | nil, ([String.t()] -> [String.t()]), ([String.t()] -> [{integer(), integer()}])) ::
          map()
  def example!(%{"text" => text} = row, intent, tag, gazetteer)
      when (is_binary(intent) or is_nil(intent)) and is_function(tag, 1) and is_function(gazetteer, 1) do
    tokens = Tokenizer.tokenize(text)
    texts = Enum.map(tokens, & &1.text)
    tags = tag.(texts)

    unless length(tags) == length(tokens) do
      raise "SlotTypePreferences: #{length(tags)} tags for #{length(tokens)} tokens of #{inspect(text)}"
    end

    tags = List.to_tuple(tags)

    ranges =
      row
      |> NerGold.spans!()
      |> Enum.map(fn span -> {span.type, Tokenizer.token_range!(tokens, span.start_pos, span.end_pos)} end)

    covered = for {_type, {first, last}} <- ranges, index <- first..last, into: MapSet.new(), do: index

    %{
      key: text,
      intent: intent,
      spans: Enum.map(ranges, fn {type, {first, _last}} -> %{type: type, preceding_tag: preceding(tags, first)} end),
      ordinary: for(index <- 0..(tuple_size(tags) - 1)//1, not MapSet.member?(covered, index), do: preceding(tags, index)),
      gazetteer: texts |> gazetteer.() |> Enum.map(&gazetteer_outcome!(&1, covered, text, length(tokens)))
    }
  end

  defp gazetteer_outcome!({first, last}, covered, _text, count)
       when is_integer(first) and is_integer(last) and 0 <= first and first <= last and last < count do
    if Enum.any?(first..last, &MapSet.member?(covered, &1)), do: :entity, else: :ordinary
  end

  defp gazetteer_outcome!(other, _covered, text, count) do
    raise ArgumentError,
          "SlotTypePreferences: the gazetteer gave #{inspect(other)} for #{inspect(text)}, " <>
            "which is not a {first, last} range of its #{count} tokens"
  end

  defp preceding(_tags, 0), do: @sentence_start
  defp preceding(tags, index), do: elem(tags, index - 1)

  # -- fitting ------------------------------------------------------------------

  @doc """
  Fits the model to `examples` (`example!/4`'s shape) with the given smoothing.

  Options, all required: `:language`; `:canonical` (runtime type => gold
  label, as `Brain.ML.EntityExtractor.load_type_mappings/0` gives it);
  `:hierarchy`, a module with `all_types/0` and `parents_of/1` as
  `Brain.Analysis.TypeHierarchy`; `:listed`, every entity type a slot schema
  lists; `:reading_types`, every entity type the gazetteer can read a span as;
  `:tag_vocabulary`, every tag a preceding token can have; and `:smoothing`, a
  map of `:global`, `:domain`, `:intent`, `:tag_global`, `:tag_type`,
  `:ordinary_global`, `:ordinary_domain`, `:ordinary_intent` (positive
  pseudo-counts) and `:hierarchy` (a weight in 0..1).
  """
  @spec fit([map()], keyword()) :: t()
  def fit([_ | _] = examples, opts) do
    examples = Enum.map(examples, &example_shape!/1)
    smoothing = smoothing!(Keyword.fetch!(opts, :smoothing))

    examples
    |> base(opts)
    |> with_smoothing(smoothing)
  end

  def fit([], _opts), do: raise(ArgumentError, "SlotTypePreferences: no examples to fit")

  @doc """
  Chooses the smoothing by cross-validation on `examples` (see the moduledoc),
  then fits on all of them. Returns `{model, report}`; the report, also kept
  as the model's `cross_validation`, gives each stage's mean held-out
  log-likelihood per observation for every value tried.

  Takes `fit/2`'s options except `:smoothing`.
  """
  @spec select([map()], keyword()) :: {t(), map()}
  def select([_ | _] = examples, opts) do
    if Keyword.has_key?(opts, :smoothing) do
      raise ArgumentError, "SlotTypePreferences.select/2 chooses the smoothing; do not pass :smoothing"
    end

    examples = Enum.map(examples, &example_shape!/1)
    full = base(examples, opts)
    folds = Enum.map(examples, &:erlang.phash2(&1.key, @folds))

    fold_data =
      Enum.map(0..(@folds - 1), fn k ->
        {held, training} = split_fold(examples, folds, k)

        if training == [] or held == [] do
          raise ArgumentError, "SlotTypePreferences: fold #{k} leaves no training or no held-out examples; too few rows to cross-validate"
        end

        {counted(full, training), held}
      end)

    slot_observations = fold_data |> Enum.map(fn {_m, held} -> length(slot_observations(held)) end) |> Enum.sum()
    tag_observations = fold_data |> Enum.map(fn {_m, held} -> length(tag_observations(held)) end) |> Enum.sum()

    ordinary_observations =
      fold_data |> Enum.map(fn {_m, held} -> length(ordinary_observations(held)) end) |> Enum.sum()

    if slot_observations == 0 do
      raise ArgumentError, "SlotTypePreferences: no example joins an intent; the slot preferences cannot be cross-validated"
    end

    if ordinary_observations == 0 do
      raise ArgumentError,
            "SlotTypePreferences: no example joined to an intent has a gazetteer span; " <>
              "the ordinary-word rate cannot be cross-validated"
    end

    start = Map.new(@smoothing_keys, &{&1, nil})

    stages = [
      {:global, @strengths, &slot_loglik(&1, &2, :global)},
      {:domain, @strengths, &slot_loglik(&1, &2, :domain)},
      {:intent, @strengths, &slot_loglik(&1, &2, :intent)},
      {:hierarchy, @hierarchy_weights, &slot_loglik(&1, &2, :full)},
      {:tag_global, @strengths, &tag_loglik(&1, &2, :global)},
      {:tag_type, @strengths, &tag_loglik(&1, &2, :full)},
      {:ordinary_global, @strengths, &ordinary_loglik(&1, &2, :global)},
      {:ordinary_domain, @strengths, &ordinary_loglik(&1, &2, :domain)},
      {:ordinary_intent, @strengths, &ordinary_loglik(&1, &2, :intent)}
    ]

    {chosen, report} =
      Enum.reduce(stages, {start, %{}}, fn {key, grid, loglik}, {chosen, report} ->
        observations =
          cond do
            key in [:tag_global, :tag_type] -> tag_observations
            key in @ordinary_keys -> ordinary_observations
            true -> slot_observations
          end

        scores =
          Enum.map(grid, fn value ->
            smoothing = Map.put(chosen, key, value)

            total =
              fold_data
              |> Enum.map(fn {model, held} -> loglik.(%{model | smoothing: smoothing}, held) end)
              |> Enum.sum()

            {value, total / observations}
          end)

        # The grids ascend, and max_by keeps the first maximum: the smaller value.
        {best, _score} = Enum.max_by(scores, fn {_value, score} -> score end)
        {Map.put(chosen, key, best), Map.put(report, key, scores)}
      end)

    report =
      Map.merge(report, %{
        chosen: chosen,
        observations: %{slots: slot_observations, tags: tag_observations, gazetteer_spans: ordinary_observations}
      })

    model =
      full
      |> counted(examples)
      |> with_smoothing(chosen)
      |> Map.put(:cross_validation, report)

    {model, report}
  end

  def select([], _opts), do: raise(ArgumentError, "SlotTypePreferences: no examples to select from")

  # -- reading the model -------------------------------------------------------

  @doc """
  The probability of each label of the universe for an intent, a domain, or
  every utterance together (`:global`; see the moduledoc). Raises for a domain
  `Brain.Lexicon.IntentDomains` does not declare and for an intent whose prefix
  it does not map.
  """
  @spec distribution(t(), context()) :: %{String.t() => float()}
  def distribution(%__MODULE__{} = model, context), do: slot_distribution(model, context, :full)

  @doc """
  What `intent` says about each of `labels`, gold labels of the model's
  universe: `log(M_intent(L) / M_global(L))` (see the moduledoc). Positive
  when the intent's utterances name `L`, or a label refining it, more often
  than utterances do overall; negative when less. A label outside the universe
  raises, as does an intent `distribution/2` refuses.
  """
  @spec intent_evidence(t(), String.t(), [String.t()]) :: %{String.t() => float()}
  def intent_evidence(%__MODULE__{} = model, intent, labels) when is_binary(intent) and is_list(labels) do
    for_intent = distribution(model, {:intent, intent})
    overall = distribution(model, :global)

    Map.new(labels, fn label ->
      unless Map.has_key?(model.descendants, label) do
        raise ArgumentError,
              "SlotTypePreferences: #{inspect(label)} is not in the #{model.language} model's universe. " <>
                "Retrain it with `mix train_slot_type_preferences`."
      end

      {label, :math.log(mass(model, for_intent, label) / mass(model, overall, label))}
    end)
  end

  # A label's mass under a distribution: its probability and that of every
  # label refining it.
  defp mass(model, distribution, label) do
    model.descendants |> Map.fetch!(label) |> Enum.map(&Map.fetch!(distribution, &1)) |> Enum.sum()
  end

  @doc """
  The weight of each listed type for an intent or a domain: the probability
  that a span filling its slots is read best as that type (see the
  moduledoc). `%{}` when nothing is listed. A listed type whose gold label is
  outside the model's universe raises: the model was trained before the type
  was listed.
  """
  @spec weights(t(), context(), [String.t()]) :: %{String.t() => float()}
  def weights(%__MODULE__{}, _context, []), do: %{}

  def weights(%__MODULE__{} = model, context, listed) when is_list(listed) do
    labels = Map.new(listed, &{&1, label!(model, &1)})
    candidates = labels |> Map.values() |> Enum.uniq()

    received =
      model
      |> distribution(context)
      |> Enum.reduce(%{}, fn {gold, p}, acc ->
        agreeing = Enum.filter(candidates, &MapSet.member?(Map.fetch!(model.ancestors, gold), &1))

        agreeing
        |> Enum.reject(fn label ->
          Enum.any?(agreeing, &(&1 != label and MapSet.member?(Map.fetch!(model.ancestors, &1), label)))
        end)
        |> Enum.reduce(acc, fn label, acc -> Map.update(acc, label, p, &(&1 + p)) end)
      end)

    Map.new(labels, fn {type, label} -> {type, Map.fetch!(received, label)} end)
  end

  @doc """
  The probability that a gazetteer span in an intent's, a domain's or every
  utterance's text (`:global`) is no entity: `Oi`, `Od` or `O0` (see the
  moduledoc). Raises for a context `distribution/2` refuses.
  """
  @spec ordinary_rate(t(), context()) :: float()
  def ordinary_rate(%__MODULE__{} = model, context), do: ordinary_level(model, context, :intent)

  @doc """
  The probability that a gold span of the context's utterances agrees with no
  type of `listed` -- the mass of the context's distribution outside the
  labels the listed types agree with (`u(c)` in the moduledoc). Every label
  agrees with no type of an empty list, so with nothing listed it is 1.
  """
  @spec unlisted_rate(t(), context(), [String.t()]) :: float()
  def unlisted_rate(%__MODULE__{} = model, context, listed) when is_list(listed) do
    listed_labels = Enum.map(listed, &label!(model, &1))

    model
    |> distribution(context)
    |> Enum.reject(fn {gold, _p} -> Enum.any?(listed_labels, &MapSet.member?(Map.fetch!(model.ancestors, gold), &1)) end)
    |> Enum.map(&elem(&1, 1))
    |> Enum.sum()
  end

  @doc """
  The context's likelihood of every reading of a gazetteer span, relative to
  its likelihood across every utterance (`K` in the moduledoc): `:ordinary`
  for the ordinary word, and `:labels`, a likelihood for every label of the
  universe. Read one reading's with `reading_likelihood/2`. Raises for a
  context `distribution/2` refuses.
  """
  @spec reading_likelihoods(t(), context()) :: %{
          language: String.t(),
          canonical: map(),
          ordinary: float(),
          labels: %{String.t() => float()}
        }
  def reading_likelihoods(%__MODULE__{} = model, context) do
    in_context = distribution(model, context)
    overall = distribution(model, :global)
    o_context = ordinary_rate(model, context)
    o_overall = ordinary_rate(model, :global)

    %{
      language: model.language,
      canonical: model.canonical,
      ordinary: o_context / o_overall,
      labels:
        Map.new(model.universe, fn label ->
          {label, (1 - o_context) * mass(model, in_context, label) / ((1 - o_overall) * mass(model, overall, label))}
        end)
    }
  end

  @doc """
  One reading's likelihood from `reading_likelihoods/2`: `:ordinary`, or a
  runtime entity type, compared through the type mappings. A type whose gold
  label is outside the model's universe raises: the model was trained before
  the gazetteer could read a span as it.
  """
  @spec reading_likelihood(map(), String.t() | :ordinary) :: float()
  def reading_likelihood(%{ordinary: ordinary}, :ordinary), do: ordinary

  def reading_likelihood(%{labels: labels, canonical: mappings, language: language}, type) when is_binary(type) do
    label = canonical(mappings, type)

    case Map.fetch(labels, label) do
      {:ok, likelihood} ->
        likelihood

      :error ->
        raise ArgumentError,
              "SlotTypePreferences: #{inspect(type)} (gold label #{inspect(label)}) is not in the #{language} model's " <>
                "universe, so no context can weigh it. Retrain it with `mix train_slot_type_preferences`."
    end
  end

  @doc """
  The likelihood of `tag` before a span read as `type` (a runtime type,
  compared through the type mappings), or before an ordinary word when
  `type` is `:ordinary`. `tag` is a UD tag or `:sentence_start`; a tag outside
  the model's vocabulary raises.
  """
  @spec preceding_likelihood(t(), String.t() | :ordinary, String.t() | :sentence_start) :: float()
  def preceding_likelihood(%__MODULE__{tag_likelihoods: tables} = model, type, tag) do
    table =
      case type do
        :ordinary -> tables.ordinary
        type when is_binary(type) -> Map.get(tables.labels, canonical(model.canonical, type), tables.global)
      end

    case Map.fetch(table, tag) do
      {:ok, p} -> p
      :error -> raise ArgumentError, "SlotTypePreferences: #{inspect(tag)} is not a tag the model knows: #{inspect(model.tag_vocabulary)}"
    end
  end

  @doc """
  Raises unless every type in `listed` has its gold label in the model's
  universe -- the check the loader makes against the slot schemas at boot.
  """
  @spec check_listed!(t(), [String.t()]) :: :ok
  def check_listed!(%__MODULE__{} = model, listed) when is_list(listed) do
    universe = MapSet.new(model.universe)

    case listed |> Enum.reject(&MapSet.member?(universe, canonical(model.canonical, &1))) |> Enum.uniq() |> Enum.sort() do
      [] ->
        :ok

      missing ->
        raise "SlotTypePreferences: the slot schemas list #{inspect(missing)}, which the #{model.language} model " <>
                "was not trained with. Retrain it with `mix train_slot_type_preferences`."
    end
  end

  @doc "Every entity type a slot schema lists (`Brain.Analysis.SlotDetector.list_schemas/0`), sorted."
  @spec listed_types() :: [String.t()]
  def listed_types do
    Brain.Analysis.SlotDetector.list_schemas()
    |> Map.values()
    |> Enum.flat_map(fn schema -> schema |> Map.get("entity_mappings", %{}) |> Map.values() end)
    |> List.flatten()
    |> Enum.uniq()
    |> Enum.sort()
  end

  # -- persistence --------------------------------------------------------------

  @doc "Where the model for `language` is stored."
  @spec path(String.t()) :: String.t()
  def path(language) do
    Path.join(Brain.Analysis.RunProvenance.models_path!(), "slot_type_preferences.#{Brain.Language.validate!(language)}.term")
  end

  @doc "Writes `model` to `path/1` for its language."
  @spec save!(t()) :: :ok
  def save!(%__MODULE__{language: language} = model) do
    smoothing!(model.smoothing)
    File.write!(path(language), Brain.ML.ModelStore.serialize(model))
  end

  @doc """
  Reads the stored model for `language`. Raises when it is missing, is not a
  model, or lacks a field this version reads -- a model trained before the
  ordinary-word counts and their smoothing were learned is refused, not
  completed with defaults.
  """
  @spec read!(String.t()) :: t()
  def read!(language) do
    file = path(language)

    case File.read(file) do
      {:ok, binary} ->
        case :erlang.binary_to_term(binary) do
          %__MODULE__{language: ^language} = model ->
            fields!(model, file)
            model

          other ->
            raise "SlotTypePreferences: #{file} does not hold a #{language} model: #{inspect(other, limit: 3)}"
        end

      {:error, reason} ->
        raise "SlotTypePreferences: cannot read #{file} (#{inspect(reason)}). " <>
                "Train it with `mix train_slot_type_preferences`."
    end
  end

  @doc "The model `Brain.Analysis.SlotTypePreferences.Loader` loaded at boot for `language`."
  @spec loaded!(String.t()) :: t()
  def loaded!(language) do
    case :persistent_term.get({__MODULE__, language}, nil) do
      %__MODULE__{} = model -> model
      nil -> raise "SlotTypePreferences: no #{language} model is loaded; the loader did not run or the language is unsupported"
    end
  end

  @doc false
  def put_loaded(%__MODULE__{language: language} = model), do: :persistent_term.put({__MODULE__, language}, model)

  # -- structure and counts ----------------------------------------------------

  defp example_shape!(%{key: _key, intent: intent, spans: spans, ordinary: ordinary, gazetteer: gazetteer} = example)
       when (is_binary(intent) or is_nil(intent)) and is_list(spans) and is_list(ordinary) and is_list(gazetteer) do
    Enum.each(spans, fn
      %{type: type, preceding_tag: _tag} when is_binary(type) -> :ok
      other -> raise ArgumentError, "SlotTypePreferences: a span needs :type and :preceding_tag, got #{inspect(other)}"
    end)

    Enum.each(gazetteer, fn
      outcome when outcome in [:entity, :ordinary] -> :ok
      other -> raise ArgumentError, "SlotTypePreferences: a gazetteer span is :entity or :ordinary, got #{inspect(other)}"
    end)

    example
  end

  defp example_shape!(other) do
    raise ArgumentError,
          "SlotTypePreferences: an example needs :key, :intent, :spans, :ordinary and :gazetteer, got #{inspect(other, limit: 5)}"
  end

  # A stored model must carry every field this version reads, each in the
  # shape it is read in.
  defp fields!(model, file) do
    case Enum.reject(@enforce_keys, &(Map.has_key?(model, &1) and Map.fetch!(model, &1) != nil)) do
      [] ->
        :ok

      missing ->
        raise "SlotTypePreferences: #{file} has no #{inspect(missing)}; it was trained by an earlier version. " <>
                "Retrain it with `mix train_slot_type_preferences`."
    end

    ordinary_counts!(model.ordinary_counts, file)

    try do
      smoothing!(model.smoothing)
    rescue
      e in ArgumentError ->
        reraise "SlotTypePreferences: #{file}: #{Exception.message(e)}. Retrain it with `mix train_slot_type_preferences`.",
                __STACKTRACE__
    end
  end

  defp ordinary_counts!(%{global: {o, m}, domain: domain, intent: intent}, _file)
       when is_integer(o) and is_integer(m) and o <= m and is_map(domain) and is_map(intent),
       do: :ok

  defp ordinary_counts!(other, file) do
    raise "SlotTypePreferences: #{file} has malformed ordinary_counts #{inspect(other, limit: 5)}. " <>
            "Retrain it with `mix train_slot_type_preferences`."
  end

  defp smoothing!(%{} = smoothing) do
    Enum.each(@smoothing_keys, fn key ->
      case Map.fetch(smoothing, key) do
        {:ok, w} when key == :hierarchy and is_number(w) and w >= 0 and w <= 1 -> :ok
        {:ok, k} when key != :hierarchy and is_number(k) and k > 0 -> :ok
        other -> raise ArgumentError, "SlotTypePreferences: smoothing #{inspect(key)} is invalid: #{inspect(other)}"
      end
    end)

    smoothing
  end

  defp smoothing!(other), do: raise(ArgumentError, "SlotTypePreferences: smoothing must be a map, got #{inspect(other)}")

  # Everything that does not depend on which examples are counted: the label
  # universe and its hierarchy, the type mappings and the tag vocabulary.
  defp base(examples, opts) do
    language = Brain.Language.validate!(Keyword.fetch!(opts, :language))
    canonical = Keyword.fetch!(opts, :canonical)
    hierarchy = Keyword.fetch!(opts, :hierarchy)
    listed = Keyword.fetch!(opts, :listed)
    reading_types = Keyword.fetch!(opts, :reading_types)
    vocabulary = Keyword.fetch!(opts, :tag_vocabulary)

    unless is_map(canonical), do: raise(ArgumentError, "SlotTypePreferences: :canonical must be a map")
    unless is_list(listed), do: raise(ArgumentError, "SlotTypePreferences: :listed must be a list of types")

    unless match?([_ | _], reading_types) and Enum.all?(reading_types, &(is_binary(&1) and &1 != "")) do
      raise ArgumentError,
            "SlotTypePreferences: :reading_types must list the entity types the gazetteer reads, got #{inspect(reading_types)}"
    end

    gold = for example <- examples, span <- example.spans, do: span.type

    hierarchy_types =
      case hierarchy.all_types() do
        [_ | _] = types -> types
        [] -> raise ArgumentError, "SlotTypePreferences: #{inspect(hierarchy)} holds no types; is it loaded?"
      end

    seeds = Enum.uniq(hierarchy_types ++ gold ++ Enum.map(listed ++ reading_types, &canonical(canonical, &1)))
    parents = parent_closure(seeds, hierarchy, %{})
    universe = parents |> Map.keys() |> Enum.sort()
    ancestors = Map.new(universe, &{&1, ancestors(&1, parents, [])})

    children =
      Enum.reduce(parents, Map.new(universe, &{&1, []}), fn {child, ps}, acc ->
        Enum.reduce(ps, acc, fn p, acc -> Map.update!(acc, p, &[child | &1]) end)
      end)
      |> Map.new(fn {p, cs} -> {p, Enum.sort(cs)} end)

    descendants =
      Enum.reduce(ancestors, Map.new(universe, &{&1, MapSet.new()}), fn {label, ups}, acc ->
        Enum.reduce(ups, acc, fn up, acc -> Map.update!(acc, up, &MapSet.put(&1, label)) end)
      end)

    tags = MapSet.new(vocabulary)

    for example <- examples, tag <- Enum.map(example.spans, & &1.preceding_tag) ++ example.ordinary, not MapSet.member?(tags, tag) do
      raise ArgumentError, "SlotTypePreferences: #{inspect(tag)} in #{inspect(example.key)} is not in the tag vocabulary"
    end

    %__MODULE__{
      language: language,
      universe: universe,
      parents: parents,
      children: children,
      ancestors: ancestors,
      descendants: descendants,
      canonical: canonical,
      tag_vocabulary: vocabulary,
      slot_counts: nil,
      tag_counts: nil,
      ordinary_counts: nil,
      smoothing: nil,
      tag_likelihoods: nil,
      fitted: nil
    }
    |> counted(examples)
  end

  defp parent_closure([], _hierarchy, acc), do: acc

  defp parent_closure([label | rest], hierarchy, acc) do
    if Map.has_key?(acc, label) do
      parent_closure(rest, hierarchy, acc)
    else
      ps = label |> hierarchy.parents_of() |> Enum.uniq() |> Enum.sort()
      parent_closure(ps ++ rest, hierarchy, Map.put(acc, label, ps))
    end
  end

  # A label and every label above it.
  defp ancestors(label, parents, path) do
    if label in path do
      raise ArgumentError, "SlotTypePreferences: the type hierarchy has a cycle through #{inspect(Enum.reverse([label | path]))}"
    end

    parents
    |> Map.fetch!(label)
    |> Enum.reduce(MapSet.new([label]), fn p, acc -> MapSet.union(acc, ancestors(p, parents, [label | path])) end)
  end

  defp counted(model, examples) do
    slots = slot_observations(examples)
    span_tags = for example <- examples, span <- example.spans, do: {span.type, span.preceding_tag}
    ordinary = Enum.flat_map(examples, & &1.ordinary)
    gazetteer = ordinary_observations(examples)

    outcome_counts = fn outcomes -> {Enum.count(outcomes, &(&1 == :ordinary)), length(outcomes)} end

    outcomes_by = fn key ->
      gazetteer |> Enum.group_by(key, &elem(&1, 1)) |> Map.new(fn {k, outcomes} -> {k, outcome_counts.(outcomes)} end)
    end

    by_label =
      span_tags
      |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
      |> Map.new(fn {label, tags} -> {label, Enum.frequencies(tags)} end)

    %{
      model
      | slot_counts: %{
          global: slots |> Enum.map(&elem(&1, 1)) |> Enum.frequencies(),
          domain: count_by(slots, fn {intent, _label} -> IntentDomains.consolidate(intent) end),
          intent: count_by(slots, fn {intent, _label} -> intent end)
        },
        tag_counts: %{
          global: Enum.frequencies(Enum.map(span_tags, &elem(&1, 1)) ++ ordinary),
          labels: by_label,
          ordinary: Enum.frequencies(ordinary)
        },
        ordinary_counts: %{
          global: outcome_counts.(Enum.map(gazetteer, &elem(&1, 1))),
          domain: outcomes_by.(fn {intent, _outcome} -> IntentDomains.consolidate(intent) end),
          intent: outcomes_by.(fn {intent, _outcome} -> intent end)
        },
        fitted: %{
          rows: length(examples),
          joined_rows: Enum.count(examples, &(&1.intent != nil)),
          slot_observations: length(slots),
          span_tags: length(span_tags),
          ordinary_tags: length(ordinary),
          gazetteer_spans: length(gazetteer),
          ordinary_gazetteer_spans: Enum.count(gazetteer, &(elem(&1, 1) == :ordinary))
        }
    }
  end

  defp count_by(slots, key) do
    slots
    |> Enum.group_by(key, &elem(&1, 1))
    |> Map.new(fn {k, labels} -> {k, Enum.frequencies(labels)} end)
  end

  defp slot_observations(examples) do
    for %{intent: intent} = example <- examples, intent != nil, span <- example.spans, do: {intent, span.type}
  end

  # `{intent, :ordinary | :entity}` for every gazetteer span of the examples
  # joined to an intent.
  defp ordinary_observations(examples) do
    for %{intent: intent} = example <- examples, intent != nil, outcome <- example.gazetteer, do: {intent, outcome}
  end

  defp tag_observations(examples) do
    for(example <- examples, span <- example.spans, do: {span.type, span.preceding_tag}) ++
      for(example <- examples, tag <- example.ordinary, do: {:ordinary, tag})
  end

  defp split_fold(examples, folds, k) do
    examples
    |> Enum.zip(folds)
    |> Enum.split_with(fn {_example, fold} -> fold == k end)
    |> then(fn {held, training} -> {Enum.map(held, &elem(&1, 0)), Enum.map(training, &elem(&1, 0))} end)
  end

  # -- slot distribution --------------------------------------------------------

  defp slot_distribution(model, {:intent, intent}, depth) when is_binary(intent) do
    domain = IntentDomains.consolidate(intent)
    global = global_level(model)

    case depth do
      :global -> global
      :domain -> level(model, :domain, domain, global)
      _ -> model |> level(:intent, intent, level(model, :domain, domain, global)) |> with_hierarchy(model, depth)
    end
  end

  defp slot_distribution(model, {:domain, domain}, depth) when is_binary(domain) do
    unless domain in IntentDomains.domains() do
      raise ArgumentError, "SlotTypePreferences: #{inspect(domain)} is not a domain; the domains are #{inspect(IntentDomains.domains())}"
    end

    model |> level(:domain, domain, global_level(model)) |> with_hierarchy(model, depth)
  end

  defp slot_distribution(model, :global, depth), do: model |> global_level() |> with_hierarchy(model, depth)

  defp slot_distribution(_model, context, _depth) do
    raise ArgumentError, "SlotTypePreferences: a context is {:intent, name}, {:domain, name} or :global, got #{inspect(context)}"
  end

  defp global_level(%__MODULE__{universe: universe, slot_counts: %{global: counts}, smoothing: %{global: k}}) do
    n = counts |> Map.values() |> Enum.sum()
    uniform = 1 / length(universe)
    Map.new(universe, &{&1, (Map.get(counts, &1, 0) + k * uniform) / (n + k)})
  end

  defp level(%__MODULE__{universe: universe, slot_counts: counts, smoothing: smoothing}, key, name, prior) do
    k = Map.fetch!(smoothing, key)
    observed = counts |> Map.fetch!(key) |> Map.get(name, %{})
    n = observed |> Map.values() |> Enum.sum()
    Map.new(universe, &{&1, (Map.get(observed, &1, 0) + k * Map.fetch!(prior, &1)) / (n + k)})
  end

  defp with_hierarchy(base, _model, :intent), do: base

  defp with_hierarchy(base, %__MODULE__{smoothing: %{hierarchy: lambda}} = model, :full) do
    subtree = Map.new(model.descendants, fn {label, below} -> {label, below |> Enum.map(&Map.fetch!(base, &1)) |> Enum.sum()} end)

    raw =
      Map.new(base, fn {label, p} ->
        inherited =
          case Map.fetch!(model.parents, label) do
            [] -> p
            ps -> (ps |> Enum.map(&(Map.fetch!(subtree, &1) / length(Map.fetch!(model.children, &1)))) |> Enum.sum()) / length(ps)
          end

        {label, (1 - lambda) * p + lambda * inherited}
      end)

    total = raw |> Map.values() |> Enum.sum()
    Map.new(raw, fn {label, p} -> {label, p / total} end)
  end

  defp slot_loglik(model, held, depth) do
    held
    |> slot_observations()
    |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
    |> Enum.map(fn {intent, labels} ->
      distribution = slot_distribution(model, {:intent, intent}, depth)
      labels |> Enum.map(&:math.log(Map.fetch!(distribution, &1))) |> Enum.sum()
    end)
    |> Enum.sum()
  end

  # -- ordinary words -------------------------------------------------------------

  # O0, Od or Oi for a context, computed down to `depth` (:global, :domain or
  # :intent), so cross-validation can choose each level's pseudo-count with
  # only the levels above it chosen.
  defp ordinary_level(model, {:intent, intent}, depth) when is_binary(intent) do
    domain = IntentDomains.consolidate(intent)
    global = ordinary_global(model)

    case depth do
      :global -> global
      :domain -> ordinary_step(model, :domain, domain, global)
      :intent -> ordinary_step(model, :intent, intent, ordinary_step(model, :domain, domain, global))
    end
  end

  defp ordinary_level(model, {:domain, domain}, _depth) when is_binary(domain) do
    unless domain in IntentDomains.domains() do
      raise ArgumentError, "SlotTypePreferences: #{inspect(domain)} is not a domain; the domains are #{inspect(IntentDomains.domains())}"
    end

    ordinary_step(model, :domain, domain, ordinary_global(model))
  end

  defp ordinary_level(model, :global, _depth), do: ordinary_global(model)

  defp ordinary_level(_model, context, _depth) do
    raise ArgumentError, "SlotTypePreferences: a context is {:intent, name}, {:domain, name} or :global, got #{inspect(context)}"
  end

  defp ordinary_global(%__MODULE__{ordinary_counts: %{global: {o, m}}, smoothing: %{ordinary_global: q}}) do
    (o + q / 2) / (m + q)
  end

  defp ordinary_step(%__MODULE__{ordinary_counts: counts, smoothing: smoothing}, key, name, prior) do
    q = Map.fetch!(smoothing, %{domain: :ordinary_domain, intent: :ordinary_intent}[key])
    {o, m} = counts |> Map.fetch!(key) |> Map.get(name, {0, 0})
    (o + q * prior) / (m + q)
  end

  defp ordinary_loglik(model, held, depth) do
    held
    |> ordinary_observations()
    |> Enum.map(fn {intent, outcome} ->
      p = ordinary_level(model, {:intent, intent}, depth)
      :math.log(if outcome == :ordinary, do: p, else: 1 - p)
    end)
    |> Enum.sum()
  end

  # -- preceding tags -----------------------------------------------------------

  defp with_smoothing(model, smoothing) do
    model = %{model | smoothing: smoothing!(smoothing)}
    %{model | tag_likelihoods: tag_tables(model, :full)}
  end

  defp tag_tables(%__MODULE__{tag_vocabulary: vocabulary, tag_counts: counts, smoothing: smoothing} = model, depth) do
    kg = smoothing.tag_global
    n = counts.global |> Map.values() |> Enum.sum()
    global = Map.new(vocabulary, &{&1, (Map.get(counts.global, &1, 0) + kg / length(vocabulary)) / (n + kg)})

    case depth do
      :global ->
        %{global: global, ordinary: global, labels: Map.new(model.universe, &{&1, global})}

      :full ->
        kt = smoothing.tag_type

        # A parent has fewer ancestors than any of its children, so this order
        # computes every parent before the labels below it.
        labels =
          model.universe
          |> Enum.sort_by(&{MapSet.size(Map.fetch!(model.ancestors, &1)), &1})
          |> Enum.reduce(%{}, fn label, tables ->
            prior =
              case Map.fetch!(model.parents, label) do
                [] -> global
                ps -> Map.new(vocabulary, fn tag -> {tag, (ps |> Enum.map(&tables[&1][tag]) |> Enum.sum()) / length(ps)} end)
              end

            pooled = pooled(counts.labels, Map.fetch!(model.descendants, label))
            Map.put(tables, label, smooth(pooled, prior, kt, vocabulary))
          end)

        %{global: global, ordinary: smooth(counts.ordinary, global, kt, vocabulary), labels: labels}
    end
  end

  defp pooled(by_label, labels) do
    Enum.reduce(labels, %{}, fn label, acc ->
      by_label |> Map.get(label, %{}) |> Enum.reduce(acc, fn {tag, n}, acc -> Map.update(acc, tag, n, &(&1 + n)) end)
    end)
  end

  defp smooth(counts, prior, k, vocabulary) do
    n = counts |> Map.values() |> Enum.sum()
    Map.new(vocabulary, &{&1, (Map.get(counts, &1, 0) + k * Map.fetch!(prior, &1)) / (n + k)})
  end

  defp tag_loglik(model, held, depth) do
    tables = tag_tables(model, depth)

    held
    |> tag_observations()
    |> Enum.map(fn
      {:ordinary, tag} -> :math.log(Map.fetch!(tables.ordinary, tag))
      {label, tag} -> :math.log(Map.fetch!(Map.fetch!(tables.labels, label), tag))
    end)
    |> Enum.sum()
  end

  defp label!(model, type) do
    label = canonical(model.canonical, type)

    if Map.has_key?(model.parents, label) do
      label
    else
      raise ArgumentError,
            "SlotTypePreferences: #{inspect(type)} (gold label #{inspect(label)}) is not in the #{model.language} model's " <>
              "universe. Retrain it with `mix train_slot_type_preferences`."
    end
  end

  defp canonical(mappings, type), do: Map.get(mappings, type, type)
end
