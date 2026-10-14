defmodule Brain.Analysis.FreeTextCapture do
  @moduledoc """
  The value of a free-text slot -- a reminder's text, a calendar item's event --
  as a span of the utterance, learned from annotated intent examples.

  No entity type names what a reminder is about, so no entity can fill
  `reminder_text`: in "remind me to call mom at 5" the value is "call mom", the
  words the utterance says it about. This module learns where such a span starts
  and ends from examples, and `Brain.Analysis.SlotDetector` fills the slot with
  it.

  ## The data

  Free-text spans are annotated in the hand-written intent examples
  (`Brain.Corpus.Provenance.authored_dir/0`) as the Dialogflow export marks an
  any-text parameter: a `data` segment with `"meta": "@sys.any"` whose `"alias"`
  names the slot. The segments concatenate to the utterance, so the intent gold
  standard rebuilds from the files unchanged.

  An intent file with one such segment is annotated in full: each of its rows
  marks the span it holds, and a row with none says the utterance names no value
  for the slot ("Cancel the reminder for 3pm"). A span holds no date, time or
  duration: those fill slots of their own. Every annotated slot must be one the
  intent registry declares for the intent, a row holds at most one span per
  slot, and every row must be a row of the intent gold standard; anything else
  raises.

  Rows are split as the intent gold standard is: a row whose text and intent are
  a row of its `held_out.json` is held out, the rest train.

  ## The candidates

  Two models are fitted, and the one with the best exact-span F1 under
  #{5}-fold cross-validation of the training rows is kept (ties go to the first
  listed):

  - `:span_pos` -- each candidate span `(i, j)` of a row, and the outcome that
    the row names no span, is scored, and the scores' softmax is the posterior of
    each outcome. A span's score is `log pi` plus naive Bayes log-likelihood
    ratios of what is at its edges and inside it, against every token of the
    rows:

        start  the word and the UD tag before token i, and the tag of token i
        end    the word and the tag after token j, and the tag of token j
        inside the tag of every token from i to j

    For a feature template `t` with value `v`, a role `r` (start, end, inside)
    contributes `log P_r(v) - log q_t(v)`, where `q_t` is the template's
    distribution over every token and `P_r(v) = (n_r(v) + s q_t(v)) / (n_r + s)`.
    The no-span outcome scores `log (1 - pi)`, with
    `pi = (rows with a span + s / 2) / (rows + s)`. A value no training token had
    says nothing and contributes 0. The strength `s` is chosen by the same
    cross-validation.

  - `:bio_lexical` -- `Brain.ML.EntityTrainer`'s constrained BIO tagger, trained
    on the spans as entities of the slot's type, its smoothing chosen by its own
    cross-validation; a span's confidence is the mean posterior of its tokens.

  ## Deciding, and abstaining

  A span never holds a date, a time or a length: a token in a run of tokens
  `Brain.Temporal.Resolver` -- what the scheduler reads them with -- reads as
  one is never in a candidate span (`temporal_tokens/2`), in training, in
  cross-validation, on the held-out rows and in `capture/4` alike. Nor, in
  `capture/4`, does a span hold a token of an entity filling another of the
  intent's slots. An annotated span holding such a token raises: those words
  fill slots of their own. The best outcome is taken; when it is a
  span whose confidence is at least the slot's threshold, the slot is filled with
  the text of the span, and otherwise the slot is left for the clarification
  that asks for it.

  The threshold is calibrated on the held-out rows: of `0` and each confidence
  the model gives a held-out span, the one with the best exact-span F1 there,
  ties going to the higher threshold, since a slot left unfilled is asked about
  and a slot filled wrongly is acted on.

  ## Language

  A model is fitted for one language and stored per language
  (`free_text_capture.<language>.term`, directly under the models directory).
  `Brain.Analysis.FreeTextCapture.Loader` loads it at boot, and boot fails
  without it. `mix train_free_text_capture` trains it. The examples are English.
  """

  alias Brain.Corpus.{Export, Label, Provenance}
  alias Brain.ML.{EntityTrainer, POSTagger, Tokenizer}
  alias Brain.Temporal.{Resolver, Zone}

  @any_text "@sys.any"

  # The instant words are read against to ask whether they name a date, a time
  # or a length (`temporal_tokens/2`); any instant answers the same.
  @reading_instant ~U[1970-01-01 00:00:00Z]
  @folds 5
  @kinds [:span_pos, :bio_lexical]

  # Pseudo-counts from far below one observation to far above the number of
  # annotated rows. Cross-validation picks within them.
  @strengths [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

  @sentence_start :sentence_start
  @sentence_end :sentence_end

  @enforce_keys [:language, :slots, :kind, :smoothing, :parameters, :thresholds, :report]
  defstruct @enforce_keys

  @type outcome :: :none | {non_neg_integer(), non_neg_integer()}

  @type t :: %__MODULE__{
          language: String.t(),
          slots: %{String.t() => %{intents: [String.t()], counts: map()}},
          kind: :span_pos | :bio_lexical,
          smoothing: float() | nil,
          parameters: map(),
          thresholds: %{String.t() => float()},
          report: map()
        }

  @doc "The candidate models, in the order ties go."
  @spec kinds() :: [atom()]
  def kinds, do: @kinds

  @doc "The strengths `:span_pos` chooses among."
  @spec strengths() :: [float()]
  def strengths, do: @strengths

  # -- examples -----------------------------------------------------------------

  @doc """
  The annotated examples: one per row of every annotated intent file in `dir`
  and free-text slot of that file, as
  `%{text, intent, slot, tokens, tags, span, split}`. `span` is the slot's
  `{first, last}` token indexes, or `nil` when the row names no value for it;
  `split` is `:train` or `:held_out`.

  Options, all required but `:dir`:

  - `:registry` -- the intent registry (`Brain.Analysis.SlotDetector.registry/0`)
  - `:gold` -- the intent gold standard's rows
  - `:held_out` -- its held-out rows
  - `:tagger` -- a function from token texts to their UD tags
  - `:dir` -- the directory of usersays files (default `Provenance.authored_dir/0`)
  """
  @spec examples!(keyword()) :: [map()]
  def examples!(opts) do
    registry = Keyword.fetch!(opts, :registry)
    tag = Keyword.fetch!(opts, :tagger)
    gold = opts |> Keyword.fetch!(:gold) |> MapSet.new(&{&1["text"], &1["intent"]})
    held = opts |> Keyword.fetch!(:held_out) |> MapSet.new(&{&1["text"], &1["intent"]})
    dir = Keyword.get(opts, :dir, Provenance.authored_dir())

    examples =
      for path <- Export.usersays_files(dir),
          entries <- [decode!(path)],
          intent <- [path |> Export.label_of() |> Label.canonical()],
          slot <- free_text_slots(entries),
          entry <- entries do
        declared!(registry, intent, slot, path)
        example!(entry, intent, slot, path, tag, gold, held)
      end

    case examples do
      [] -> raise ArgumentError, "FreeTextCapture: no usersays file under #{dir} annotates a free-text slot (#{@any_text})"
      examples -> examples
    end
  end

  defp decode!(path) do
    case path |> File.read!() |> Jason.decode() do
      {:ok, entries} when is_list(entries) -> entries
      other -> raise ArgumentError, "FreeTextCapture: #{path} is not a JSON array of usersays entries: #{inspect(other, limit: 3)}"
    end
  end

  defp free_text_slots(entries) do
    entries
    |> Enum.flat_map(&Map.get(&1, "data", []))
    |> Enum.filter(&(Map.get(&1, "meta") == @any_text))
    |> Enum.map(&Map.get(&1, "alias"))
    |> Enum.uniq()
    |> Enum.sort()
  end

  defp declared!(registry, intent, slot, path) do
    schema =
      case Map.fetch(registry, intent) do
        {:ok, schema} -> schema
        :error -> raise ArgumentError, "FreeTextCapture: #{path} annotates #{inspect(slot)}, but the intent registry has no #{intent}"
      end

    unless is_binary(slot) and slot in (Map.get(schema, "required", []) ++ Map.get(schema, "optional", [])) do
      raise ArgumentError,
            "FreeTextCapture: #{path} annotates #{inspect(slot)}, which the intent registry does not declare for #{intent}"
    end
  end

  defp example!(entry, intent, slot, path, tag, gold, held) do
    text = Export.phrase_text(entry)
    key = {text, intent}

    unless MapSet.member?(gold, key) do
      raise ArgumentError,
            "FreeTextCapture: #{path}: #{inspect(text)} is not a #{intent} row of the intent gold standard; " <>
              "rebuild it with `mix rebuild_gold_standard`"
    end

    tokens = Tokenizer.tokenize(text)
    tags = tag.(Enum.map(tokens, & &1.text))

    if length(tags) != length(tokens) do
      raise ArgumentError, "FreeTextCapture: #{length(tags)} tags for #{length(tokens)} tokens of #{inspect(text)}"
    end

    temporal = temporal_tokens(text, tokens)

    span =
      case segment_ranges(entry, slot) do
        [] -> nil
        [{start_pos, end_pos}] -> Tokenizer.token_range!(tokens, start_pos, end_pos)
        many -> raise ArgumentError, "FreeTextCapture: #{path}: #{inspect(text)} marks #{length(many)} #{slot} spans; a row holds one"
      end

    with {i, j} <- span, true <- Enum.any?(i..j, &MapSet.member?(temporal, &1)) do
      raise ArgumentError,
            "FreeTextCapture: #{path}: the #{slot} span of #{inspect(text)} holds words Brain.Temporal.Resolver reads as " <>
              "a date, time or length; those fill slots of their own"
    end

    %{
      text: text,
      intent: intent,
      slot: slot,
      tokens: tokens,
      tags: tags,
      temporal: temporal,
      span: span,
      split: if(MapSet.member?(held, key), do: :held_out, else: :train)
    }
  end

  @doc """
  The indexes of the tokens of `text` (its `Brain.ML.Tokenizer` tokens) that
  lie in a run of tokens `Brain.Temporal.Resolver` reads as a date, a time or a
  length. Whether words name one does not depend on when they are said, so they
  are read against one fixed instant in UTC.
  """
  @spec temporal_tokens(String.t(), [map()]) :: MapSet.t()
  def temporal_tokens(text, tokens) do
    indexed = tokens |> Enum.with_index() |> List.to_tuple()
    n = tuple_size(indexed)

    for i <- 0..(n - 1)//1,
        j <- i..(n - 1)//1,
        {first, _} <- [elem(indexed, i)],
        {last, _} <- [elem(indexed, j)],
        match?({:ok, _}, Resolver.resolve(String.slice(text, first.start_pos, last.end_pos - first.start_pos + 1), @reading_instant, time_zone: Zone.utc())),
        k <- i..j,
        into: MapSet.new(),
        do: k
  end

  # The grapheme ranges (inclusive) of the entry's segments annotating `slot`.
  defp segment_ranges(entry, slot) do
    entry
    |> Map.get("data", [])
    |> Enum.reduce({0, []}, fn segment, {pos, ranges} ->
      length = segment |> Map.get("text", "") |> String.length()

      ranges =
        if Map.get(segment, "meta") == @any_text and Map.get(segment, "alias") == slot,
          do: [{pos, pos + length - 1} | ranges],
          else: ranges

      {pos + length, ranges}
    end)
    |> elem(1)
    |> Enum.reverse()
  end

  # -- fitting ------------------------------------------------------------------

  @doc """
  Chooses the model and its strength by cross-validation of the `:train`
  examples, fits it on all of them, and calibrates each slot's threshold on the
  `:held_out` examples. Returns the model; its `:report` holds the
  cross-validated counts of every candidate and the held-out results.
  """
  @spec select([map()], keyword()) :: t()
  def select([_ | _] = examples, opts) do
    language = Brain.Language.validate!(Keyword.fetch!(opts, :language))
    train = Enum.filter(examples, &(&1.split == :train))
    held = Enum.filter(examples, &(&1.split == :held_out))
    slots = examples |> Enum.map(& &1.slot) |> Enum.uniq() |> Enum.sort()

    for slot <- slots, {split, rows} <- [train: train, held_out: held] do
      if not Enum.any?(rows, &(&1.slot == slot and &1.span != nil)) do
        raise ArgumentError, "FreeTextCapture: no #{split} example holds a #{slot} span; the slot cannot be #{verb(split)}"
      end
    end

    cross_validation = cross_validate(train)

    {{kind, smoothing}, _counts} =
      Enum.max_by(cross_validation, fn {{kind, smoothing}, counts} ->
        {f1(counts.exact), -Enum.find_index(@kinds, &(&1 == kind)), -(smoothing || 0)}
      end)

    parameters = fit(kind, smoothing, train)
    decisions = Enum.map(held, &{&1, decide(kind, parameters, &1.slot, &1, &1.temporal)})

    thresholds =
      Map.new(slots, fn slot -> {slot, calibrate(Enum.filter(decisions, fn {example, _} -> example.slot == slot end))} end)

    held_out =
      Map.new(slots, fn slot ->
        rows = Enum.filter(decisions, fn {example, _} -> example.slot == slot end)
        threshold = Map.fetch!(thresholds, slot)

        {slot,
         %{
           decided: score(Enum.map(rows, fn {example, {outcome, _p}} -> {example, outcome} end)),
           thresholded: score(Enum.map(rows, fn {example, decision} -> {example, apply_threshold(decision, threshold)} end)),
           rows: Enum.map(rows, fn {example, {outcome, p}} -> %{text: example.text, gold: span_text(example, example.span), predicted: span_text(example, outcome), confidence: p} end)
         }}
      end)

    %__MODULE__{
      language: language,
      slots: Map.new(slots, &{&1, slot_summary(examples, &1)}),
      kind: kind,
      smoothing: smoothing,
      parameters: parameters,
      thresholds: thresholds,
      report: %{cross_validation: cross_validation, held_out: held_out}
    }
  end

  def select([], _opts), do: raise(ArgumentError, "FreeTextCapture: no examples to select from")

  defp verb(:train), do: "learned"
  defp verb(:held_out), do: "calibrated"

  defp slot_summary(examples, slot) do
    rows = Enum.filter(examples, &(&1.slot == slot))
    count = fn split, pred -> Enum.count(rows, &(&1.split == split and pred.(&1))) end

    %{
      intents: rows |> Enum.map(& &1.intent) |> Enum.uniq() |> Enum.sort(),
      counts: %{
        train_rows: count.(:train, fn _ -> true end),
        train_spans: count.(:train, &(&1.span != nil)),
        held_out_rows: count.(:held_out, fn _ -> true end),
        held_out_spans: count.(:held_out, &(&1.span != nil))
      }
    }
  end

  # `{kind, strength}` => %{exact: {tp, fp, fn}, tokens: {tp, fp, fn}} over the
  # held folds, each fold's rows decided by a model fitted on the other folds.
  defp cross_validate(train) do
    folds = Enum.map(train, &:erlang.phash2(&1.text, @folds))
    indexed = Enum.zip(train, folds)

    splits =
      for k <- 0..(@folds - 1) do
        {held, rest} = Enum.split_with(indexed, fn {_example, fold} -> fold == k end)
        {Enum.map(rest, &elem(&1, 0)), Enum.map(held, &elem(&1, 0))}
      end

    if Enum.any?(splits, fn {rest, held} -> held == [] or not Enum.any?(rest, &(&1.span != nil)) end) do
      raise ArgumentError, "FreeTextCapture: a fold leaves no held rows or no training span; too few rows to cross-validate"
    end

    candidates = [{:bio_lexical, nil} | Enum.map(@strengths, &{:span_pos, &1})]

    Map.new(candidates, fn {kind, smoothing} ->
      pairs =
        Enum.flat_map(splits, fn {rest, held} ->
          parameters = fit(kind, smoothing, rest)
          Enum.map(held, fn example -> {example, kind |> decide(parameters, example.slot, example, example.temporal) |> elem(0)} end)
        end)

      {{kind, smoothing}, score(pairs)}
    end)
  end

  defp fit(:span_pos, smoothing, examples) do
    examples
    |> Enum.group_by(& &1.slot)
    |> Map.new(fn {slot, rows} -> {slot, fit_span(rows, smoothing)} end)
  end

  defp fit(:bio_lexical, nil, examples) do
    {:ok, model} = EntityTrainer.train(Enum.map(examples, &ner_row/1))
    %{model: model}
  end

  defp ner_row(%{text: text, span: nil}), do: %{"text" => text, "expected" => []}
  defp ner_row(%{text: text, slot: slot} = example), do: %{"text" => text, "expected" => [%{"type" => slot, "value" => span_text(example, example.span)}]}

  # Counts per feature template and value: over every token (the background),
  # and at the start, the end and inside every span.
  defp fit_span(rows, strength) do
    empty = %{background: %{}, start: %{}, end: %{}, inside: %{}}

    counts =
      Enum.reduce(rows, empty, fn row, acc ->
        {words, tags} = features_input(row)
        n = tuple_size(words)

        acc =
          Enum.reduce(0..(n - 1)//1, acc, fn k, acc ->
            (start_features(words, tags, k) ++ end_features(words, tags, k) ++ inside_features(tags, k))
            |> Enum.reduce(acc, &bump(&2, :background, &1))
          end)

        case row.span do
          nil ->
            acc

          {i, j} ->
            acc = Enum.reduce(start_features(words, tags, i), acc, &bump(&2, :start, &1))
            acc = Enum.reduce(end_features(words, tags, j), acc, &bump(&2, :end, &1))
            Enum.reduce(i..j, acc, fn k, acc -> Enum.reduce(inside_features(tags, k), acc, &bump(&2, :inside, &1)) end)
        end
      end)

    tokens = rows |> Enum.map(&length(&1.tokens)) |> Enum.sum()
    spans = Enum.count(rows, &(&1.span != nil))
    inside = rows |> Enum.map(fn %{span: span} -> if span, do: elem(span, 1) - elem(span, 0) + 1, else: 0 end) |> Enum.sum()
    events = %{start: spans, end: spans, inside: inside}

    ratios =
      Map.new(counts.background, fn {{template, _value} = feature, n_bg} ->
        q = n_bg / tokens
        role = role_of(template)
        n_role = counts |> Map.fetch!(role) |> Map.get(feature, 0)
        {feature, :math.log((n_role + strength * q) / (Map.fetch!(events, role) + strength)) - :math.log(q)}
      end)

    prior = (spans + strength / 2) / (length(rows) + strength)
    %{log_prior: :math.log(prior), log_none: :math.log(1 - prior), ratios: ratios}
  end

  defp bump(acc, role, feature), do: Map.update!(acc, role, &Map.update(&1, feature, 1, fn n -> n + 1 end))

  defp role_of(template) when template in [:prev_word, :prev_tag, :first_tag], do: :start
  defp role_of(template) when template in [:next_word, :next_tag, :last_tag], do: :end
  defp role_of(:inside_tag), do: :inside

  defp features_input(%{tokens: tokens, tags: tags}),
    do: {tokens |> Enum.map(&String.downcase(&1.text)) |> List.to_tuple(), List.to_tuple(tags)}

  defp start_features(words, tags, i),
    do: [{:prev_word, at(words, i - 1)}, {:prev_tag, at(tags, i - 1)}, {:first_tag, elem(tags, i)}]

  defp end_features(words, tags, j),
    do: [{:next_word, at(words, j + 1)}, {:next_tag, at(tags, j + 1)}, {:last_tag, elem(tags, j)}]

  defp inside_features(tags, k), do: [{:inside_tag, elem(tags, k)}]

  defp at(_tuple, -1), do: @sentence_start
  defp at(tuple, k) when k == tuple_size(tuple), do: @sentence_end
  defp at(tuple, k), do: elem(tuple, k)

  # -- deciding -----------------------------------------------------------------

  # The best outcome for `example`'s tokens and tags and its posterior, no
  # candidate span holding a token index of `excluded`.
  defp decide(:span_pos, parameters, slot, example, excluded) do
    params = Map.fetch!(parameters, slot)
    {words, tags} = features_input(example)
    n = tuple_size(words)
    ratio = fn feature -> Map.get(params.ratios, feature, 0.0) end

    spans =
      for i <- 0..(n - 1)//1,
          j <- i..(n - 1)//1,
          not Enum.any?(i..j, &MapSet.member?(excluded, &1)) do
        score =
          params.log_prior +
            (start_features(words, tags, i) |> Enum.map(ratio) |> Enum.sum()) +
            (end_features(words, tags, j) |> Enum.map(ratio) |> Enum.sum()) +
            (i..j |> Enum.flat_map(&inside_features(tags, &1)) |> Enum.map(ratio) |> Enum.sum())

        {{i, j}, score}
      end

    outcomes = [{:none, params.log_none} | spans]
    log_z = outcomes |> Enum.map(&elem(&1, 1)) |> log_sum_exp()
    {best, score} = Enum.max_by(outcomes, &elem(&1, 1))
    {best, :math.exp(score - log_z)}
  end

  defp decide(:bio_lexical, %{model: model}, slot, example, excluded) do
    example.tokens
    |> Enum.map(& &1.text)
    |> EntityTrainer.tag(model)
    |> EntityTrainer.extract_entities_from_bio()
    |> Enum.filter(&(&1.entity_type == slot))
    |> Enum.reject(fn entity -> Enum.any?(entity.start_index..entity.end_index, &MapSet.member?(excluded, &1)) end)
    |> case do
      [] -> {:none, 1.0}
      entities -> entities |> Enum.max_by(& &1.confidence) |> then(&{{&1.start_index, &1.end_index}, &1.confidence})
    end
  end

  defp apply_threshold({{_i, _j} = span, p}, threshold) when p >= threshold, do: span
  defp apply_threshold(_decision, _threshold), do: :none

  defp calibrate(decisions) do
    candidates = [0.0 | for({_example, {{_i, _j}, p}} <- decisions, do: p)] |> Enum.uniq()

    Enum.max_by(candidates, fn threshold ->
      pairs = Enum.map(decisions, fn {example, decision} -> {example, apply_threshold(decision, threshold)} end)
      {f1(score(pairs).exact), threshold}
    end)
  end

  # Exact-span and token counts `{tp, fp, fn}` of `{example, outcome}` pairs.
  defp score(pairs) do
    Enum.reduce(pairs, %{exact: {0, 0, 0}, tokens: {0, 0, 0}}, fn {example, outcome}, acc ->
      gold = if example.span, do: example.span, else: :none

      exact =
        cond do
          outcome == gold and gold != :none -> {1, 0, 0}
          outcome != :none and gold != :none -> {0, 1, 1}
          outcome != :none -> {0, 1, 0}
          gold != :none -> {0, 0, 1}
          true -> {0, 0, 0}
        end

      gold_tokens = token_set(gold)
      predicted_tokens = token_set(outcome)
      hit = MapSet.size(MapSet.intersection(gold_tokens, predicted_tokens))
      tokens = {hit, MapSet.size(predicted_tokens) - hit, MapSet.size(gold_tokens) - hit}

      %{exact: add(acc.exact, exact), tokens: add(acc.tokens, tokens)}
    end)
  end

  defp token_set(:none), do: MapSet.new()
  defp token_set({i, j}), do: MapSet.new(i..j)

  defp add({a, b, c}, {d, e, f}), do: {a + d, b + e, c + f}

  @doc "Precision, recall and F1 of `{tp, fp, fn}` counts."
  @spec prf({non_neg_integer(), non_neg_integer(), non_neg_integer()}) :: {float(), float(), float()}
  def prf({tp, fp, fn_}) do
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn_, 1)
    {precision, recall, if(precision + recall == 0, do: 0.0, else: 2 * precision * recall / (precision + recall))}
  end

  defp f1(counts), do: counts |> prf() |> elem(2)

  defp log_sum_exp(values) do
    max = Enum.max(values)
    max + :math.log(Enum.sum(Enum.map(values, &:math.exp(&1 - max))))
  end

  defp span_text(_example, :none), do: nil
  defp span_text(_example, nil), do: nil

  defp span_text(%{text: text, tokens: tokens}, {i, j}) do
    first = Enum.at(tokens, i)
    last = Enum.at(tokens, j)
    String.slice(text, first.start_pos, last.end_pos - first.start_pos + 1)
  end

  # -- capturing ----------------------------------------------------------------

  @doc "The free-text slots `model` captures."
  @spec slots(t()) :: [String.t()]
  def slots(%__MODULE__{slots: slots}), do: slots |> Map.keys() |> Enum.sort()

  @doc """
  The value `model` captures for `slot` in `text`: `{:ok, %{value, confidence,
  start_pos, end_pos}}`, or `:abstain` when the best outcome is that `text`
  names none, or a span below the slot's threshold. `excluded` lists the
  grapheme ranges `{start_pos, end_pos}` (inclusive) of the entities that fill
  the intent's other slots; no span holds a token of one. Tags `text` with the
  loaded `Brain.ML.POSTagger` model, and raises when it cannot be loaded or
  `slot` is not one `model` captures.
  """
  @spec capture(t(), String.t(), String.t(), [{non_neg_integer(), non_neg_integer()}]) :: {:ok, map()} | :abstain
  def capture(%__MODULE__{} = model, slot, text, excluded) when is_binary(slot) and is_binary(text) and is_list(excluded) do
    threshold =
      case Map.fetch(model.thresholds, slot) do
        {:ok, threshold} -> threshold
        :error -> raise ArgumentError, "FreeTextCapture: the #{model.language} model captures #{inspect(slots(model))}, not #{inspect(slot)}"
      end

    tokens = Tokenizer.tokenize(text)
    example = %{text: text, tokens: tokens, tags: tag!(tokens)}

    excluded_tokens =
      for {token, k} <- Enum.with_index(tokens),
          {start_pos, end_pos} <- excluded,
          token.start_pos <= end_pos and token.end_pos >= start_pos,
          into: temporal_tokens(text, tokens),
          do: k

    case tokens do
      [] ->
        :abstain

      _ ->
        case model.kind |> decide(model.parameters, slot, example, excluded_tokens) |> apply_threshold_with(threshold) do
          {{i, j}, p} ->
            first = Enum.at(tokens, i)
            last = Enum.at(tokens, j)
            {:ok, %{value: span_text(example, {i, j}), confidence: p, start_pos: first.start_pos, end_pos: last.end_pos}}

          :none ->
            :abstain
        end
    end
  end

  defp apply_threshold_with({{_i, _j}, p} = decision, threshold) when p >= threshold, do: decision
  defp apply_threshold_with(_decision, _threshold), do: :none

  defp tag!(tokens) do
    case POSTagger.load_model() do
      {:ok, pos_model} -> POSTagger.predict_tags(Enum.map(tokens, & &1.text), pos_model)
      {:error, reason} -> raise "FreeTextCapture: the POS model is required to read a span's edges: #{inspect(reason)}"
    end
  end

  # -- persistence --------------------------------------------------------------

  @doc "Where the model for `language` is stored."
  @spec path(String.t()) :: String.t()
  def path(language) do
    Path.join(Brain.Analysis.RunProvenance.models_path!(), "free_text_capture.#{Brain.Language.validate!(language)}.term")
  end

  @doc "Writes `model` to `path/1` for its language."
  @spec save!(t()) :: :ok
  def save!(%__MODULE__{language: language} = model), do: File.write!(path(language), Brain.ML.ModelStore.serialize(model))

  @doc """
  Reads the stored model for `language`. Raises when it is missing or does not
  hold a model for the language with a threshold for every slot it captures.
  """
  @spec read!(String.t()) :: t()
  def read!(language) do
    file = path(language)

    case File.read(file) do
      {:ok, binary} ->
        case :erlang.binary_to_term(binary) do
          %__MODULE__{language: ^language, slots: slots, thresholds: thresholds} = model
          when map_size(slots) > 0 and map_size(thresholds) == map_size(slots) ->
            model

          other ->
            raise "FreeTextCapture: #{file} does not hold a #{language} model with a threshold for every slot: " <>
                    "#{inspect(other, limit: 3)}. Retrain it with `mix train_free_text_capture`."
        end

      {:error, reason} ->
        raise "FreeTextCapture: cannot read #{file} (#{inspect(reason)}). Train it with `mix train_free_text_capture`."
    end
  end

  @doc """
  Raises unless the intent registry declares every slot `model` captures for
  every intent it was annotated for.
  """
  @spec check_declared!(t(), map()) :: :ok
  def check_declared!(%__MODULE__{slots: slots, language: language}, registry) do
    for {slot, %{intents: intents}} <- slots, intent <- intents do
      schema = Map.get(registry, intent, %{})

      unless slot in (Map.get(schema, "required", []) ++ Map.get(schema, "optional", [])) do
        raise "FreeTextCapture: the #{language} model captures #{slot} for #{intent}, which the intent registry " <>
                "does not declare there. Retrain it with `mix train_free_text_capture`."
      end
    end

    :ok
  end

  @doc "The model `Brain.Analysis.FreeTextCapture.Loader` loaded at boot for `language`."
  @spec loaded!(String.t()) :: t()
  def loaded!(language) do
    case :persistent_term.get({__MODULE__, language}, nil) do
      %__MODULE__{} = model -> model
      nil -> raise "FreeTextCapture: no #{language} model is loaded; the loader did not run or the language is unsupported"
    end
  end

  @doc false
  def put_loaded(%__MODULE__{language: language} = model), do: :persistent_term.put({__MODULE__, language}, model)
end
