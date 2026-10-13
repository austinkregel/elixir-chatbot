defmodule Brain.ML.SpanReliability do
  @moduledoc """
  How far each entity source can be trusted when it reads a span as a type,
  learned from the NER gold standard's training split, and the decision those
  reliabilities support when the sources read one span differently.

  ## Readings and spans

  `Brain.ML.EntityExtractor` runs several sources over a sentence -- the
  gazetteer, the system parsers for numbers and dates, the location and
  proper-noun hints, the world's entity model -- and keeps what each reports
  as a *reading*: its `:source`, the `:entity_type` it gave, its extent
  (`:start_pos`..`:end_pos`, graphemes, inclusive) and its own `:confidence`.
  Readings whose extents overlap form one span. No reading is dropped: a
  decided span and an unresolved one both carry every reading in `:readings`.

  ## Reliability

  For source `s` emitting type `t`, the reliability is P(the gold standard
  agrees | `s` emitted `t`), counted over the readings of the training rows.
  Gold spans are placed by `Brain.ML.NerGold.spans!/1`, and a runtime type is
  compared to the gold labels through the NER type mappings
  (`Brain.ML.EntityExtractor.load_type_mappings/0`). A reading's outcome is:

  - `:exact` -- a gold span has exactly its extent and the reading's type;
  - `:refined` -- a gold span has exactly its extent and a type the reading's
    type is refined by in `Brain.Analysis.TypeHierarchy` (a `number` reading of
    what the gold calls `final_value`). The reading is right at its own, less
    specific, level, so this counts as agreement, not a miss;
  - `{:wrong_type, gold_type}` -- a gold span has its extent and another type;
  - `:boundary` -- no gold span has its extent, but one overlaps it;
  - `:no_entity` -- no gold span overlaps it.

  Agreement is `:exact` plus `:refined`. The counts are smoothed in two levels
  by a Beta prior of strength `prior_strength`, a pseudo-count of readings: a
  source's rate is pulled toward the rate over every measured reading, and a
  `{source, type}` pair's toward its source's,

      source:  (agree_s + k * overall) / (n_s + k)
      pair:    (agree   + k * source)  / (n   + k)

  so a pair seen a few times says little beyond what its source does. The
  counts behind every rate are kept in the model (`pairs`, `sources`,
  `overall`).

  ## Measurable pairs

  A type is measurable when the gold labels can confirm it: some gold label of
  the training rows is the type, or refines it. A reading whose type is not
  measurable, or whose `{source, type}` pair the training rows never showed,
  has no reliability. It stays among the span's readings, but it is evidence
  for or against nothing, so a span read only by such readings stays
  unresolved. `unmeasurable` keeps how many training readings each such pair
  had.

  ## Combining and deciding

  A span's hypotheses are `{start_pos, end_pos, label}`, one for each extent
  and gold-space type its measured readings give. A measured reading supports
  every hypothesis at its extent whose label is its own type or refines it,
  and its reliability is added in log-odds (`Brain.Lattice.merge/3`,
  `:log_odds`); the span's readings are taken as independent evidence. The
  best hypothesis is decided when its summed log-odds is at least `min_score`
  and, when another hypothesis competes, its lead over the next is greater
  than zero and at least `min_margin` (`:infinity` when no contested span is
  decided). Equal evidence never decides.

  A span that is not decided is *unresolved*: `resolved: false`, with every
  reading and the candidate lattice, and typed at the most specific type every
  reading's type is or descends from in the hierarchy -- or `nil` when the
  readings share none, or more than one equally specific. An unresolved span
  fills no slot by itself.

  ## Deciding again with more evidence

  A span left unresolved for `:insufficient_evidence` can be decided again
  once something its readings did not know is known (`redecide/4`): a log-odds
  term per hypothesis label, added to the readings' sum as one more
  independent piece of evidence. The evidence decides nothing on its own -- a
  label no measured reading gives is no hypothesis -- and a span it does not
  lift over the thresholds stays unresolved.

  Those thresholds are the model's `redecision`, chosen by
  `select_redecision/3` as `select/2` chooses `min_score` and `min_margin`:
  #{5}-fold cross-validation on the training rows, each fold's spans decided
  by reliabilities fitted on the other folds and weighed again with evidence
  fitted on the other folds, maximizing the span-level F1 of both decisions
  together. Measured 2026-10-13 on the 1,146 NER training rows joined with
  intent gold, cross-validated F1 was 82.67 deciding nothing again, 84.21
  with the first decision's thresholds and 84.47 with thresholds of their
  own, so the second decision has its own.

  ## Choosing the prior strength and the thresholds

  `select/2` compares candidate prior strengths and thresholds by
  #{5}-fold cross-validation on the training rows: each fold's spans are
  decided with reliabilities fitted on the other folds, and the choice
  maximizes span-level F1 -- a decision is a true positive when a gold span has
  its extent and a type that is or refines its label, any other decision is a
  false positive, and every gold span not decided correctly is a false
  negative, so abstaining costs recall and a wrong decision costs both. The
  thresholds tried are the scores and margins the folds produced. The
  held-out split is never read.

  ## Language

  A model is fitted for one language and stored per language
  (`span_reliability.<language>.term`, directly under the models directory, so
  run provenance hashes it). `Brain.ML.SpanReliability.Loader` loads it at
  boot and boot fails without it. Tokens come from `Brain.ML.Tokenizer`, the
  English implementation, so English is the only language that can be fitted.
  """

  alias Brain.Lattice
  alias Brain.Lattice.Candidate

  @enforce_keys [:language, :prior_strength, :pairs, :unmeasurable, :sources, :overall, :gold_types, :fitted_rows]
  defstruct [
    :language,
    :prior_strength,
    :pairs,
    :unmeasurable,
    :sources,
    :overall,
    :gold_types,
    :fitted_rows,
    :min_score,
    :min_margin,
    :cross_validation,
    :redecision
  ]

  @type pair :: {atom(), String.t()}

  @type t :: %__MODULE__{
          language: String.t(),
          prior_strength: number(),
          pairs: %{pair() => map()},
          unmeasurable: %{pair() => %{n: pos_integer()}},
          sources: %{atom() => map()},
          overall: %{n: non_neg_integer(), agree: non_neg_integer(), rate: float()},
          gold_types: [String.t()],
          fitted_rows: non_neg_integer(),
          min_score: float() | nil,
          min_margin: float() | :infinity | nil,
          cross_validation: map() | nil,
          redecision:
            %{min_score: float() | :infinity, min_margin: float() | :infinity, cross_validation: map()} | nil
        }

  # Pseudo-counts of readings, from barely more than the counts to a prior
  # that outweighs any pair the training split shows. Cross-validation picks
  # one; ties go to the smaller strength. On the NER training split (1,688
  # rows, 2026-10-09) cross-validated F1 peaked at 1,000 (0.849) and fell to
  # 0.826 at 100,000, so the grid ends past that peak.
  @prior_strengths [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000, 100_000]
  @folds 5

  @doc """
  Fits the reliabilities from training examples, each `%{key: term, readings:
  [reading], gold: [%{type, start_pos, end_pos}]}`: a row's readings as
  `Brain.ML.EntityExtractor.readings/2` gives them and its gold spans as
  `Brain.ML.NerGold.spans!/1` places them. `key` identifies the row for fold
  assignment.

  Options: `:language`, `:prior_strength` (a positive pseudo-count),
  `:canonical` (runtime type => gold label, as `load_type_mappings/0` gives
  it) and `:hierarchy` (a module with `is_a?/2` and `parents_of/1`, as
  `Brain.Analysis.TypeHierarchy`), all required. The thresholds are left unset;
  `select/2` sets them.
  """
  @spec fit([map()], keyword()) :: t()
  def fit([_ | _] = examples, opts) do
    language = Brain.Language.validate!(Keyword.fetch!(opts, :language))
    prior_strength = prior_strength!(Keyword.fetch!(opts, :prior_strength))
    canonical = Keyword.fetch!(opts, :canonical)
    hierarchy = Keyword.fetch!(opts, :hierarchy)

    examples = Enum.map(examples, &example!/1)
    gold_types = examples |> Enum.flat_map(& &1.gold) |> Enum.map(& &1.type) |> Enum.uniq() |> Enum.sort()

    if gold_types == [] do
      raise ArgumentError, "SpanReliability: the training rows have no gold spans to measure readings against"
    end

    measurable =
      examples
      |> Enum.flat_map(& &1.readings)
      |> Enum.map(& &1.entity_type)
      |> Enum.uniq()
      |> Map.new(&{&1, measurable_type?(&1, gold_types, canonical, hierarchy)})

    {measured, unmeasured} =
      examples
      |> Enum.flat_map(fn example ->
        Enum.map(example.readings, fn reading ->
          {{reading.source, reading.entity_type}, outcome(reading, example.gold, canonical, hierarchy)}
        end)
      end)
      |> Enum.split_with(fn {{_source, type}, _outcome} -> Map.fetch!(measurable, type) end)

    tallies = measured |> Enum.group_by(&elem(&1, 0), &elem(&1, 1)) |> Map.new(fn {pair, outs} -> {pair, tally(outs)} end)
    overall = overall!(tallies)

    sources =
      tallies
      |> Enum.group_by(fn {{source, _type}, _tally} -> source end, fn {_pair, tally} -> tally end)
      |> Map.new(fn {source, source_tallies} ->
        n = source_tallies |> Enum.map(& &1.n) |> Enum.sum()
        agree = source_tallies |> Enum.map(& &1.agree) |> Enum.sum()
        {source, %{n: n, agree: agree, reliability: (agree + prior_strength * overall.rate) / (n + prior_strength)}}
      end)

    pairs =
      Map.new(tallies, fn {{source, _type} = pair, tally} ->
        prior = Map.fetch!(sources, source).reliability
        {pair, Map.put(tally, :reliability, (tally.agree + prior_strength * prior) / (tally.n + prior_strength))}
      end)

    %__MODULE__{
      language: language,
      prior_strength: prior_strength,
      pairs: pairs,
      unmeasurable: unmeasured |> Enum.frequencies_by(&elem(&1, 0)) |> Map.new(fn {pair, n} -> {pair, %{n: n}} end),
      sources: sources,
      overall: overall,
      gold_types: gold_types,
      fitted_rows: length(examples)
    }
  end

  def fit([], _opts), do: raise(ArgumentError, "SpanReliability: no training examples to fit")

  @doc """
  Picks the prior strength, `min_score` and `min_margin` by #{@folds}-fold
  cross-validation on `examples` (see the moduledoc), then fits on all of
  them. Returns `{model, report}`, where `report` maps each prior strength to
  the best cross-validated F1 it reached and the thresholds and counts behind
  it; the same report is kept in the model's `cross_validation`.

  Takes `fit/2`'s options except `:prior_strength`.
  """
  @spec select([map()], keyword()) :: {t(), map()}
  def select([_ | _] = examples, opts) do
    if Keyword.has_key?(opts, :prior_strength) do
      raise ArgumentError, "SpanReliability.select/2 chooses the prior strength; do not pass :prior_strength"
    end

    canonical = Keyword.fetch!(opts, :canonical)
    hierarchy = Keyword.fetch!(opts, :hierarchy)
    examples = Enum.map(examples, &example!/1)
    folds = Enum.map(examples, &:erlang.phash2(&1.key, @folds))
    gold_total = examples |> Enum.map(&length(&1.gold)) |> Enum.sum()

    report =
      Map.new(@prior_strengths, fn strength ->
        points =
          Enum.flat_map(0..(@folds - 1), fn k ->
            {held, training} = split_fold(examples, folds, k)

            case training do
              [] ->
                raise ArgumentError, "SpanReliability: fold #{k} leaves no training examples; too few rows to cross-validate"

              _ ->
                model = fit(training, Keyword.put(opts, :prior_strength, strength))
                Enum.flat_map(held, &decision_points(&1, model, canonical, hierarchy))
            end
          end)

        {strength, best_thresholds(points, gold_total)}
      end)

    {strength, chosen} =
      Enum.max_by(report, fn {strength, result} -> {result.f1, -strength} end)

    model =
      examples
      |> fit(Keyword.put(opts, :prior_strength, strength))
      |> Map.merge(%{min_score: chosen.min_score, min_margin: chosen.min_margin, cross_validation: report})

    {model, report}
  end

  def select([], _opts), do: raise(ArgumentError, "SpanReliability: no training examples to select from")

  @doc """
  Chooses the thresholds `redecide/4` decides with (see the moduledoc) and
  returns `model` with them as its `redecision`: `%{min_score, min_margin,
  cross_validation}`, the report giving the cross-validated F1 with no span
  decided again, with the first decision's thresholds, and with the ones
  chosen. Thresholds that decide nothing again are `:infinity`.

  `model` is `select/2`'s, fitted on `examples`: its prior strength and
  thresholds make each fold's first decision. `examples` take `fit/2`'s shape
  and are the rows the evidence is known for. Options: `:canonical` and
  `:hierarchy` as for `fit/2`, and `:evidence`, a function given a fold's
  training examples that returns the evidence fitted on them: a function of
  a held-out example and the labels of an unresolved span's hypotheses,
  returning a log-odds term for each label.
  """
  @spec select_redecision(t(), [map()], keyword()) :: t()
  def select_redecision(%__MODULE__{} = model, [_ | _] = examples, opts) do
    thresholds!(model)
    canonical = Keyword.fetch!(opts, :canonical)
    hierarchy = Keyword.fetch!(opts, :hierarchy)
    evidence_for = Keyword.fetch!(opts, :evidence)

    unless is_function(evidence_for, 1) do
      raise ArgumentError, "SpanReliability.select_redecision/3: :evidence must be a function of a fold's training examples"
    end

    examples = Enum.map(examples, &example!/1)
    folds = Enum.map(examples, &:erlang.phash2(&1.key, @folds))
    gold_total = examples |> Enum.map(&length(&1.gold)) |> Enum.sum()
    fit_opts = Keyword.merge(opts, language: model.language, prior_strength: model.prior_strength) |> Keyword.delete(:evidence)

    # Every unresolved span decided again with no threshold, so each point
    # holds the score and margin a threshold is compared with.
    open = %{min_score: -1.0e300, min_margin: 0.0}

    {first, points} =
      Enum.reduce(0..(@folds - 1), {%{tp: 0, fp: 0}, []}, fn k, {first, points} ->
        {held, training} = split_fold(examples, folds, k)

        if training == [] do
          raise ArgumentError, "SpanReliability: fold #{k} leaves no training examples; too few rows to cross-validate"
        end

        fold_model = training |> fit(fit_opts) |> Map.merge(%{min_score: model.min_score, min_margin: model.min_margin})
        evidence = evidence_for.(training)

        Enum.reduce(held, {first, points}, fn example, {first, points} ->
          spans = decide(example.readings, fold_model, text: "", canonical: canonical, hierarchy: hierarchy)
          {decided, unresolved} = Enum.split_with(spans, & &1.resolved)
          right = Enum.count(decided, &agrees?({&1.start_pos, &1.end_pos, canonical(&1.entity_type, canonical)}, example.gold, hierarchy))
          first = %{tp: first.tp + right, fp: first.fp + length(decided) - right}

          new_points =
            for %{unresolved_reason: :insufficient_evidence} = span <- unresolved,
                labels = span.readings |> Enum.filter(& &1.measured) |> Enum.map(& &1.canonical) |> Enum.uniq(),
                {:decided, entity} <- [redecide(span, evidence.(example, labels), %{fold_model | redecision: open}, hierarchy)] do
              label = {entity.start_pos, entity.end_pos, canonical(entity.entity_type, canonical)}
              %{score: entity.decision.score, margin: entity.decision.margin, correct: agrees?(label, example.gold, hierarchy)}
            end

          {first, new_points ++ points}
        end)
      end)

    offsets = {first.tp, first.fp}
    chosen = best_thresholds(points, gold_total, offsets)
    with_first = redecision_result(points, gold_total, offsets, model.min_score, model.min_margin)
    none = redecision_result([], gold_total, offsets, :infinity, :infinity)

    %{
      model
      | redecision: %{
          min_score: chosen.min_score,
          min_margin: chosen.min_margin,
          cross_validation: %{none: none, first_thresholds: with_first, chosen: chosen, points: length(points)}
        }
    }
  end

  def select_redecision(%__MODULE__{}, [], _opts),
    do: raise(ArgumentError, "SpanReliability: no examples to choose the redecision thresholds from")

  # The F1, decided and correct counts of deciding the points at the given
  # thresholds, beside the first decision's counts.
  defp redecision_result(points, gold_total, {tp0, fp0}, min_score, min_margin) do
    chosen = Enum.filter(points, &(score_ok?(&1.score, min_score) and margin_ok?(&1.margin, min_margin)))
    tp = tp0 + Enum.count(chosen, & &1.correct)
    fp = fp0 + length(chosen) - Enum.count(chosen, & &1.correct)

    %{
      f1: 2 * tp / max(2 * tp + fp + (gold_total - tp), 1),
      min_score: min_score,
      min_margin: min_margin,
      decided: tp + fp,
      correct: tp,
      gold: gold_total
    }
  end

  defp score_ok?(_score, :infinity), do: false
  defp score_ok?(score, min_score), do: score >= min_score

  @doc """
  Decides every span of a sentence's readings.

  Returns one map per span, in text order. A decided span is the reading of
  its winning hypothesis that the data trusts most (other readings of that
  hypothesis contribute the keys it lacks, such as a gazetteer reading's
  metadata), with `resolved: true`, `:confidence` the combined probability,
  `:readings`, `:candidates` (the `Brain.Lattice`) and `:decision`. An
  unresolved span is described in the moduledoc and carries
  `:unresolved_reason` (`:no_measured_reading` or `:insufficient_evidence`),
  `confidence: 0.0`, and `:raw_readings`, its readings as the sources gave
  them, from which `redecide/4` builds the entity it decides: it is no entity
  of any type until something decides it.

  Options: `:text` (the sentence the positions index), `:canonical` and
  `:hierarchy` as for `fit/2`. Raises when the model has no thresholds.
  """
  @spec decide([map()], t(), keyword()) :: [map()]
  def decide(readings, %__MODULE__{} = model, opts) when is_list(readings) do
    thresholds!(model)
    text = Keyword.fetch!(opts, :text)
    canonical = Keyword.fetch!(opts, :canonical)
    hierarchy = Keyword.fetch!(opts, :hierarchy)

    readings
    |> Enum.map(&reading!/1)
    |> clusters()
    |> Enum.map(fn cluster ->
      summaries = Enum.map(cluster, &summary(&1, model, canonical))
      lattice = hypotheses(summaries, hierarchy)

      case judge(lattice, model.min_score, model.min_margin) do
        {:decided, best, margin} ->
          decided(cluster, summaries, best, margin, lattice)

        {:undecided, reason} ->
          summaries |> unresolved(lattice, reason, text, hierarchy) |> Map.put(:raw_readings, cluster)
      end
    end)
  end

  @doc """
  Decides again a span `decide/3` left unresolved for `:insufficient_evidence`,
  with `evidence`: a log-odds term for each label (gold-space type) of the
  span's hypotheses, added to each hypothesis's summed reading log-odds (see
  the moduledoc). Every hypothesis label must have a term; a missing one
  raises, so no hypothesis is weighed with less evidence than the others.

  The model's `redecision` thresholds decide (`select_redecision/3`); a model
  without them raises.

  Returns `{:decided, entity}`, the span as `decide/3` returns a decided one,
  its `:candidates` the lattice with the evidence and its `:decision` also
  holding `:evidence`; or `{:undecided, lattice}`, the lattice with the
  evidence, when the thresholds still do not decide it. Any other span
  raises.
  """
  @spec redecide(map(), %{String.t() => float()}, t(), module()) :: {:decided, map()} | {:undecided, Lattice.t()}
  def redecide(
        %{resolved: false, unresolved_reason: :insufficient_evidence, readings: summaries, raw_readings: cluster} = span,
        evidence,
        %__MODULE__{} = model,
        hierarchy
      )
      when is_map(evidence) do
    %{min_score: min_score, min_margin: min_margin} = redecision!(model)

    labels =
      summaries
      |> Enum.filter(& &1.measured)
      |> Enum.map(& &1.canonical)
      |> Enum.uniq()

    case Enum.reject(labels, &is_number(Map.get(evidence, &1))) do
      [] ->
        :ok

      missing ->
        raise ArgumentError,
              "SpanReliability: no evidence for #{inspect(missing)}, labels of the hypotheses of " <>
                "#{inspect(Map.get(span, :value))}; every hypothesis is weighed with the same evidence"
    end

    lattice = hypotheses(summaries, hierarchy, evidence)

    case judge(lattice, min_score, min_margin) do
      {:decided, best, margin} ->
        entity = decided(cluster, summaries, best, margin, lattice)
        {:decided, %{entity | decision: Map.put(entity.decision, :evidence, Map.take(evidence, labels))}}

      {:undecided, _reason} ->
        {:undecided, lattice}
    end
  end

  def redecide(span, _evidence, _model, _hierarchy) do
    raise ArgumentError,
          "SpanReliability: only a span decide/3 left unresolved for :insufficient_evidence can be decided again; " <>
            "got #{inspect(span, limit: 8)}"
  end

  @doc """
  Turns a decided span back into an unresolved one, for a caller that will
  not act on the decision (a confidence floor). Its readings and candidates
  are kept and `reason` is recorded as `:unresolved_reason`.
  """
  @spec withhold(map(), atom(), String.t(), module()) :: map()
  def withhold(%{resolved: true, readings: summaries, candidates: %Lattice{} = lattice}, reason, text, hierarchy)
      when is_atom(reason) do
    unresolved(summaries, lattice, reason, text, hierarchy)
  end

  def withhold(entity, _reason, _text, _hierarchy) do
    raise ArgumentError, "SpanReliability: only a decided span can be withheld, got #{inspect(entity, limit: 8)}"
  end

  @doc """
  The most specific type each of `types` is or descends from in `hierarchy`'s
  static parents, or `nil` when they share none or more than one equally
  specific.
  """
  @spec common_type([String.t()], module()) :: String.t() | nil
  def common_type([_ | _] = types, hierarchy) do
    common =
      types
      |> Enum.uniq()
      |> Enum.map(&ancestors_or_self(&1, hierarchy))
      |> Enum.reduce(&MapSet.intersection/2)

    lowest =
      Enum.reject(common, fn candidate ->
        Enum.any?(common, &(&1 != candidate and MapSet.member?(ancestors_or_self(&1, hierarchy), candidate)))
      end)

    case lowest do
      [type] -> type
      _ -> nil
    end
  end

  @doc "Where the model for `language` is stored."
  @spec path(String.t()) :: String.t()
  def path(language) do
    Path.join(Brain.Analysis.RunProvenance.models_path!(), "span_reliability.#{Brain.Language.validate!(language)}.term")
  end

  @doc "Writes `model` to `path/1` for its language. Raises when it has no thresholds or no redecision thresholds."
  @spec save!(t()) :: :ok
  def save!(%__MODULE__{language: language} = model) do
    thresholds!(model)
    redecision!(model)
    File.write!(path(language), Brain.ML.ModelStore.serialize(model))
  end

  @doc "Reads the stored model for `language`. Raises when it is missing or is not a model."
  @spec read!(String.t()) :: t()
  def read!(language) do
    file = path(language)

    case File.read(file) do
      {:ok, binary} ->
        case :erlang.binary_to_term(binary) do
          %__MODULE__{language: ^language} = model ->
            thresholds!(model)
            redecision!(model)
            model

          other ->
            raise "SpanReliability: #{file} does not hold a #{language} model: #{inspect(other, limit: 3)}"
        end

      {:error, reason} ->
        raise "SpanReliability: cannot read #{file} (#{inspect(reason)}). " <>
                "Train it with `mix train_span_reliability --world <id>`."
    end
  end

  @doc "The model `Brain.ML.SpanReliability.Loader` loaded at boot for `language`."
  @spec loaded!(String.t()) :: t()
  def loaded!(language) do
    case :persistent_term.get({__MODULE__, language}, nil) do
      %__MODULE__{} = model -> model
      nil -> raise "SpanReliability: no #{language} model is loaded; the loader did not run or the language is unsupported"
    end
  end

  @doc false
  def put_loaded(%__MODULE__{language: language} = model), do: :persistent_term.put({__MODULE__, language}, model)

  # -- fitting ------------------------------------------------------------------

  defp prior_strength!(strength) when is_number(strength) and strength > 0, do: strength

  defp prior_strength!(other) do
    raise ArgumentError, "SpanReliability: :prior_strength must be a positive number, got #{inspect(other)}"
  end

  defp example!(%{key: _key, readings: readings, gold: gold} = example) when is_list(readings) and is_list(gold) do
    %{example | readings: Enum.map(readings, &reading!/1), gold: Enum.map(gold, &gold_span!/1)}
  end

  defp example!(other) do
    raise ArgumentError, "SpanReliability: an example needs :key, :readings and :gold, got #{inspect(other, limit: 5)}"
  end

  defp reading!(%{source: source, entity_type: type, start_pos: s, end_pos: e} = reading)
       when is_atom(source) and not is_nil(source) and is_binary(type) and is_integer(s) and is_integer(e) and s <= e,
       do: reading

  defp reading!(other) do
    raise ArgumentError,
          "SpanReliability: a reading needs a :source, an :entity_type, and :start_pos <= :end_pos; got #{inspect(other, limit: 8)}"
  end

  defp gold_span!(%{type: type, start_pos: s, end_pos: e} = span)
       when is_binary(type) and is_integer(s) and is_integer(e) and s <= e,
       do: span

  defp gold_span!(other) do
    raise ArgumentError, "SpanReliability: a gold span needs :type, :start_pos and :end_pos; got #{inspect(other)}"
  end

  defp measurable_type?(type, gold_types, canonical, hierarchy) do
    label = canonical(type, canonical)
    Enum.any?(gold_types, &(&1 == label or hierarchy.is_a?(&1, label)))
  end

  defp outcome(reading, gold, canonical, hierarchy) do
    label = canonical(reading.entity_type, canonical)

    case Enum.find(gold, &(&1.start_pos == reading.start_pos and &1.end_pos == reading.end_pos)) do
      %{type: ^label} ->
        :exact

      %{type: gold_type} ->
        if hierarchy.is_a?(gold_type, label), do: :refined, else: {:wrong_type, gold_type}

      nil ->
        if Enum.any?(gold, &overlaps?(&1, reading)), do: :boundary, else: :no_entity
    end
  end

  defp tally(outcomes) do
    base = %{n: 0, agree: 0, exact: 0, refined: 0, boundary: 0, no_entity: 0, wrong_type: %{}}

    Enum.reduce(outcomes, base, fn outcome, acc ->
      acc = %{acc | n: acc.n + 1}

      case outcome do
        :exact -> %{acc | exact: acc.exact + 1, agree: acc.agree + 1}
        :refined -> %{acc | refined: acc.refined + 1, agree: acc.agree + 1}
        :boundary -> %{acc | boundary: acc.boundary + 1}
        :no_entity -> %{acc | no_entity: acc.no_entity + 1}
        {:wrong_type, gold_type} -> %{acc | wrong_type: Map.update(acc.wrong_type, gold_type, 1, &(&1 + 1))}
      end
    end)
  end

  # The rate every source is pulled toward. It must lie strictly between 0 and
  # 1, or the log-odds of a smoothed reliability would be infinite.
  defp overall!(tallies) do
    n = tallies |> Map.values() |> Enum.map(& &1.n) |> Enum.sum()
    agree = tallies |> Map.values() |> Enum.map(& &1.agree) |> Enum.sum()

    if agree == 0 or agree == n do
      raise ArgumentError,
            "SpanReliability: #{agree} of #{n} measured readings agree with the gold standard; " <>
              "a rate of 0 or 1 cannot be learned from"
    end

    %{n: n, agree: agree, rate: agree / n}
  end

  defp split_fold(examples, folds, k) do
    examples
    |> Enum.zip(folds)
    |> Enum.split_with(fn {_example, fold} -> fold == k end)
    |> then(fn {held, training} -> {Enum.map(held, &elem(&1, 0)), Enum.map(training, &elem(&1, 0))} end)
  end

  # One point per span with a measured hypothesis: its best score, its margin
  # (`:infinity` without a competitor) and whether deciding it agrees with the
  # gold. A tie can never be decided, so it is no point.
  defp decision_points(example, model, canonical, hierarchy) do
    example.readings
    |> clusters()
    |> Enum.flat_map(fn cluster ->
      lattice = cluster |> Enum.map(&summary(&1, model, canonical)) |> hypotheses(hierarchy)

      case lattice do
        %Lattice{best: nil} ->
          []

        %Lattice{best: best} = lattice ->
          case margin(lattice) do
            margin when is_number(margin) and margin <= 0 -> []
            margin -> [%{score: best.score, margin: margin, correct: agrees?(best.label, example.gold, hierarchy)}]
          end
      end
    end)
  end

  defp agrees?({start_pos, end_pos, label}, gold, hierarchy) do
    Enum.any?(gold, fn span ->
      span.start_pos == start_pos and span.end_pos == end_pos and
        (span.type == label or hierarchy.is_a?(span.type, label))
    end)
  end

  # The (min_score, min_margin) with the best F1 over `points`, counted with
  # `offsets`, the `{correct, wrong}` decisions already made. Ties go to the
  # larger margin, then the larger score: when the data cannot tell two
  # thresholds apart, the one that decides less is kept.
  defp best_thresholds(points, gold_total, offsets \\ {0, 0}) do
    finite_margins = points |> Enum.map(& &1.margin) |> Enum.reject(&(&1 == :infinity)) |> Enum.uniq() |> Enum.sort()

    [0.0 | finite_margins]
    |> Enum.uniq()
    |> Kernel.++([:infinity])
    |> Enum.map(fn min_margin ->
      points
      |> Enum.filter(&margin_ok?(&1.margin, min_margin))
      |> best_score_threshold(gold_total, offsets)
      |> Map.put(:min_margin, min_margin)
    end)
    |> Enum.max_by(fn result -> {result.f1, rank(result.min_margin), rank(result.min_score)} end)
  end

  # Sweeps the score threshold down through the scores of `points`: deciding
  # every point at or above it, beside the decisions `offsets` counts.
  # Deciding none of them is the threshold `:infinity`.
  defp best_score_threshold(points, gold_total, {tp0, fp0}) do
    f1 = fn tp, fp -> 2 * tp / max(2 * tp + fp + (gold_total - tp), 1) end
    nothing = %{f1: f1.(tp0, fp0), min_score: :infinity, decided: tp0 + fp0, correct: tp0, gold: gold_total}

    points
    |> Enum.group_by(& &1.score)
    |> Enum.sort_by(&elem(&1, 0), :desc)
    |> Enum.reduce({nothing, tp0, fp0}, fn {score, group}, {best, tp, fp} ->
      correct = Enum.count(group, & &1.correct)
      tp = tp + correct
      fp = fp + length(group) - correct
      candidate = %{f1: f1.(tp, fp), min_score: score, decided: tp + fp, correct: tp, gold: gold_total}

      if {candidate.f1, rank(score)} > {best.f1, rank(best.min_score)},
        do: {candidate, tp, fp},
        else: {best, tp, fp}
    end)
    |> elem(0)
  end

  # Larger thresholds decide less. `:infinity` decides nothing.
  defp rank(:infinity), do: {1, 0.0}
  defp rank(value) when is_number(value), do: {0, value}

  defp margin_ok?(:infinity, _min_margin), do: true
  defp margin_ok?(_margin, :infinity), do: false
  defp margin_ok?(margin, min_margin), do: margin > 0.0 and margin >= min_margin

  # -- deciding -----------------------------------------------------------------

  defp thresholds!(%__MODULE__{min_score: score, min_margin: margin})
       when is_number(score) and (is_number(margin) or margin == :infinity),
       do: :ok

  defp thresholds!(%__MODULE__{min_score: score, min_margin: margin}) do
    raise ArgumentError,
          "SpanReliability: the model has no decision thresholds (min_score #{inspect(score)}, " <>
            "min_margin #{inspect(margin)}); fit it with select/2"
  end

  # The thresholds `redecide/4` decides with. A model read from a file written
  # before they existed has no `:redecision` key at all.
  defp redecision!(%__MODULE__{} = model) do
    case Map.get(model, :redecision) do
      %{min_score: score, min_margin: margin} = redecision
      when (is_number(score) or score == :infinity) and (is_number(margin) or margin == :infinity) ->
        redecision

      other ->
        raise ArgumentError,
              "SpanReliability: the model has no redecision thresholds (#{inspect(other, limit: 3)}); " <>
                "choose them with select_redecision/3 (`mix train_span_reliability`)"
    end
  end

  # Readings whose extents overlap, chained: a reading joins the span when it
  # starts at or before the furthest end so far.
  defp clusters(readings) do
    readings
    |> Enum.sort_by(&{&1.start_pos, &1.end_pos})
    |> Enum.chunk_while(
      {nil, []},
      fn reading, {max_end, acc} ->
        cond do
          acc == [] -> {:cont, {reading.end_pos, [reading]}}
          reading.start_pos <= max_end -> {:cont, {max(max_end, reading.end_pos), [reading | acc]}}
          true -> {:cont, Enum.reverse(acc), {reading.end_pos, [reading]}}
        end
      end,
      fn
        {_max_end, []} -> {:cont, {nil, []}}
        {_max_end, acc} -> {:cont, Enum.reverse(acc), {nil, []}}
      end
    )
  end

  defp summary(reading, model, canonical) do
    {reliability, evidence} =
      case Map.fetch(model.pairs, {reading.source, reading.entity_type}) do
        {:ok, pair} -> {pair.reliability, pair.n}
        :error -> {nil, 0}
      end

    %{
      source: reading.source,
      entity_type: reading.entity_type,
      canonical: canonical(reading.entity_type, canonical),
      start_pos: reading.start_pos,
      end_pos: reading.end_pos,
      raw_confidence: Map.get(reading, :confidence),
      measured: reliability != nil,
      reliability: reliability,
      evidence: evidence
    }
  end

  # One candidate per hypothesis a measured reading supports, with the
  # reliability as its confidence, and one per hypothesis whose label
  # `evidence` gives a log-odds term for, with that term as its log-odds; the
  # log-odds merge sums them per hypothesis.
  defp hypotheses(summaries, hierarchy, evidence \\ %{}) do
    measured = Enum.filter(summaries, & &1.measured)
    labels = measured |> Enum.map(&{&1.start_pos, &1.end_pos, &1.canonical}) |> Enum.uniq()

    candidates =
      for reading <- measured,
          {start_pos, end_pos, label} = hypothesis <- labels,
          start_pos == reading.start_pos and end_pos == reading.end_pos,
          reading.canonical == label or hierarchy.is_a?(label, reading.canonical) do
        %Candidate{label: hypothesis, score: 0.0, confidence: reading.reliability, source: reading.source, metadata: %{}}
      end

    evidenced =
      for {_start_pos, _end_pos, label} = hypothesis <- labels, Map.has_key?(evidence, label) do
        log_odds = Map.fetch!(evidence, label)
        %Candidate{label: hypothesis, score: 0.0, confidence: 1 / (1 + :math.exp(-log_odds)), source: :evidence, metadata: %{}}
      end

    Lattice.merge(Lattice.empty(:entity_span), %Lattice{candidates: candidates ++ evidenced}, :log_odds)
  end

  defp margin(%Lattice{second: nil}), do: :infinity
  defp margin(%Lattice{best: best, second: second}), do: best.score - second.score

  defp judge(%Lattice{best: nil}, _min_score, _min_margin), do: {:undecided, :no_measured_reading}

  defp judge(%Lattice{best: best} = lattice, min_score, min_margin) do
    margin = margin(lattice)

    if best.score >= min_score and margin_ok?(margin, min_margin),
      do: {:decided, best, margin},
      else: {:undecided, :insufficient_evidence}
  end

  defp decided(cluster, summaries, %Candidate{label: {start_pos, end_pos, label}} = best, margin, lattice) do
    [{base, _summary} | others] =
      cluster
      |> Enum.zip(summaries)
      |> Enum.filter(fn {_reading, summary} ->
        summary.measured and summary.start_pos == start_pos and summary.end_pos == end_pos and summary.canonical == label
      end)
      |> Enum.sort_by(fn {_reading, summary} -> {-summary.reliability, Atom.to_string(summary.source)} end)

    others
    |> Enum.reduce(%{}, fn {reading, _summary}, acc -> Map.merge(reading, acc) end)
    |> Map.merge(base)
    |> Map.merge(%{
      confidence: best.confidence,
      resolved: true,
      readings: summaries,
      candidates: lattice,
      decision: %{score: best.score, margin: margin}
    })
  end

  defp unresolved(summaries, lattice, reason, text, hierarchy) do
    start_pos = summaries |> Enum.map(& &1.start_pos) |> Enum.min()
    end_pos = summaries |> Enum.map(& &1.end_pos) |> Enum.max()
    surface = String.slice(text, start_pos, end_pos - start_pos + 1)

    %{
      entity_type: summaries |> Enum.map(& &1.entity_type) |> common_type(hierarchy),
      value: surface,
      match: surface,
      start_pos: start_pos,
      end_pos: end_pos,
      confidence: 0.0,
      resolved: false,
      unresolved_reason: reason,
      readings: summaries,
      candidates: lattice
    }
  end

  defp ancestors_or_self(type, hierarchy), do: walk_up([type], hierarchy, MapSet.new())

  defp walk_up([], _hierarchy, seen), do: seen

  defp walk_up([type | rest], hierarchy, seen) do
    if MapSet.member?(seen, type),
      do: walk_up(rest, hierarchy, seen),
      else: walk_up(hierarchy.parents_of(type) ++ rest, hierarchy, MapSet.put(seen, type))
  end

  defp canonical(type, mappings), do: Map.get(mappings, type, type)

  defp overlaps?(a, b), do: a.start_pos <= b.end_pos and b.start_pos <= a.end_pos
end
