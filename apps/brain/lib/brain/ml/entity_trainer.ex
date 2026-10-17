defmodule Brain.ML.EntityTrainer do
  @moduledoc """
  Entity recognition by BIO tagging.

  Trained on the NER gold standard's training split
  (`EvaluationStore.load_gold_standard("ner", :train)`), each row turned into a
  token-aligned BIO sequence by `Brain.ML.NerGold`; the held-out split is left
  for `mix evaluate.ner`.

  BIO tagging:
  - B-{type}: beginning of an entity
  - I-{type}: inside (continuation) of an entity
  - O: outside any entity

  Example:
    "turn down heating until 40 percent"
    ["O", "O", "B-device", "O", "B-final_value", "O"]

  `tag/2` decodes the best tag sequence (Viterbi) and gives each tag its posterior
  probability under the model (forward-backward), and an extracted entity's
  confidence is the mean of its tokens' posteriors.

  ## Transitions

  An `I-x` tag continues an entity of type `x`, so it can only follow `B-x` or
  `I-x`; it cannot open a sentence or follow `O` or a tag of another type.
  Those transitions are not in the model at all, so neither the decode nor the
  posteriors can produce an invalid tag sequence. Every other transition
  (including from `<START>` and to `<END>`) has the probability the training
  sequences give it, smoothed with a Dirichlet of strength
  `transition_smoothing` toward how often each successor follows anything
  (restricted to the successors allowed after that tag): a transition the
  training rows never showed, such as `B-room` straight to `B-device`, stays
  possible. `transition_smoothing` is chosen by the same cross-validation as
  `smoothing`.

  ## Emission

  A token's features (its form, affixes, shape, neighbours) are combined as a
  naive Bayes product: `log P(tag) + Σ_f [log P(tag | f) - log P(tag)]`. A
  feature as common under every tag (`not_first`) adds nothing, and a decisive
  one (`token:kitchen`) dominates. `P(tag | f)` is smoothed toward the tag prior
  with a Dirichlet of strength `smoothing`, chosen by 5-fold cross-validation
  on the training rows. (An earlier version averaged
  `P(tag | f)` over the features, which let the shape features outvote every
  word, so it tagged every token `O`.)
  """

  # On the NER training split (1,688 rows, 2026-10-08) cross-validation preferred
  # ever less smoothing, with held-out exact-span F1 0.831 at 1e-3 and 0.843 at
  # 1e-6: an unseen (feature, tag) pair becomes close to a veto. Zero is not a
  # candidate, since it makes such a pair impossible.
  @smoothing_candidates [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 0.01, 0.1, 1.0]
  # On the same split (2026-10-13) the transition smoothing chosen from
  # [0.01 .. 100] was 0.01, the grid's smallest; from this grid it is 1.0e-4,
  # inside it. Training took 500s with this grid against 259-320s with that one.
  @transition_smoothing_candidates [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 0.01, 0.1, 1.0, 10.0, 100.0]
  @folds 5

  alias Brain.ML.{EvaluationStore, NerGold}
  require Logger

  @type bio_tag :: String.t()

  @type entity_model :: %{
          tag_vocabulary: %{bio_tag() => integer()},
          feature_weights: %{String.t() => %{bio_tag() => float()}},
          feature_totals: %{String.t() => pos_integer()},
          smoothing: float(),
          transition_weights: %{bio_tag() => %{bio_tag() => float()}},
          transition_smoothing: float(),
          tag_priors: %{bio_tag() => float()}
        }

  @doc """
  Trains the entity model. `rows` are NER gold rows (`%{"text", "expected"}`);
  they default to the gold standard's training split. Returns `{:ok, model}`, or
  raises when a row cannot be placed or there are no rows.
  """
  def train(rows \\ EvaluationStore.load_gold_standard("ner", :train))

  def train([_ | _] = rows) do
    start_time = System.monotonic_time(:millisecond)

    :telemetry.execute(
      [:chat_bot, :ml, :train, :start],
      %{sequence_count: 0},
      %{model: :entity_trainer, started_at: DateTime.utc_now()}
    )

    sequences = Enum.map(rows, &NerGold.sequence!/1)
    {smoothing, transition_smoothing} = select_smoothing(rows, sequences)
    model = train_sequence_model(sequences, smoothing, transition_smoothing)

    Logger.info("Entity model trained", %{
      sequence_count: length(sequences),
      tag_count: map_size(model.tag_vocabulary),
      feature_count: map_size(model.feature_weights)
    })

    :telemetry.execute(
      [:chat_bot, :ml, :train, :stop],
      %{
        duration_ms: System.monotonic_time(:millisecond) - start_time,
        sequence_count: length(sequences),
        tag_count: map_size(model.tag_vocabulary),
        feature_count: map_size(model.feature_weights)
      },
      %{model: :entity_trainer, success: true}
    )

    {:ok, model}
  end

  def train([]), do: raise(ArgumentError, "EntityTrainer: no NER rows to train on")

  @doc """
  Out-of-fold models for the rows `model` was trained on: `{fold_of_row,
  models}`, where a row's fold model was trained with `model`'s smoothing and
  transition smoothing on every other fold's rows. Tagging a training row with
  its fold model shows how the model reads a sentence it was not trained on,
  which is what `Brain.ML.SpanReliability` measures. Folds are assigned as
  `train/1`'s cross-validation assigns them.
  """
  @spec fold_models([map()], entity_model()) :: {[non_neg_integer()], %{non_neg_integer() => entity_model()}}
  def fold_models([_ | _] = rows, model) do
    smoothing = trained_strength!(model, :smoothing)
    transition_smoothing = trained_strength!(model, :transition_smoothing)

    folds = Enum.map(rows, &:erlang.phash2(&1, @folds))
    sequences = Enum.map(rows, &NerGold.sequence!/1)

    models =
      Map.new(0..(@folds - 1), fn k ->
        training =
          sequences
          |> Enum.zip(folds)
          |> Enum.reject(fn {_sequence, fold} -> fold == k end)
          |> Enum.map(&elem(&1, 0))

        {k, train_sequence_model(training, smoothing, transition_smoothing)}
      end)

    {folds, models}
  end

  def fold_models([], _model), do: raise(ArgumentError, "EntityTrainer: no NER rows to fold")

  defp trained_strength!(model, key) do
    case Map.fetch(model, key) do
      {:ok, strength} when is_number(strength) and strength > 0 ->
        strength

      other ->
        raise ArgumentError,
              "EntityTrainer: fold models need the trained model's #{key}, got #{inspect(other)}; " <>
                "retrain with `mix train --world <id>`"
    end
  end

  @doc "Train entity recognition model and save to disk.\n\n## Options\n  - models_path: Override the default models output path\n  - rows: The NER gold rows to train on (default: the gold standard's training split, as `train/1`)\n"
  def train_and_save(opts \\ []) do
    models_path =
      Keyword.get(opts, :models_path) ||
        Application.get_env(:brain, :ml)[:models_path] ||
        Brain.priv_path("ml_models")

    {:ok, model} =
      case Keyword.fetch(opts, :rows) do
        {:ok, rows} -> train(rows)
        :error -> train()
      end
    save_model(model, models_path)
  end

  @doc "Load a trained entity model from disk.\n"
  def load_model do
    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
    model_path = Path.join(models_path, "entity_model.term")

    Brain.Telemetry.measure_model_load("entity_trainer", fn ->
      case File.read(model_path) do
        {:ok, binary} ->
          try do
            model = :erlang.binary_to_term(binary)
            {:ok, model}
          rescue
            e -> {:error, "Failed to deserialize model: #{inspect(e)}"}
          end

        {:error, reason} ->
          {:error, "Failed to read model: #{reason}"}
      end
    end)
  end

  @doc "Predict entity tags for a sequence of tokens.\nReturns list of {token, predicted_tag} tuples.\n"
  def predict(tokens, model) when is_list(tokens) do
    predictions = viterbi_decode(tokens, model)
    Enum.zip(tokens, predictions)
  end

  @doc """
  The best tag sequence for `tokens`, each tag with its posterior probability
  under the model: `[{token, tag, probability}]`.
  """
  @spec tag([String.t()], entity_model()) :: [{String.t(), bio_tag(), float()}]
  def tag([], _model), do: []

  def tag(tokens, model) when is_list(tokens) do
    tags = viterbi_decode(tokens, model)
    marginals = posterior_marginals(tokens, model)

    tokens
    |> Enum.zip(tags)
    |> Enum.zip(marginals)
    |> Enum.map(fn {{token, tag}, posterior} -> {token, tag, Map.fetch!(posterior, tag)} end)
  end

  @doc """
  Entities from tagged tokens (`tag/2`'s output). Each is
  `%{entity_type, value, tokens, confidence, start_index, end_index}`, where the
  indexes are the first and last token positions in the input and `confidence`
  is the mean posterior of its tokens. An `I-` tag that continues no entity of
  its type is not a BIO sequence and raises.
  """
  def extract_entities_from_bio(tagged) when is_list(tagged) do
    tagged
    |> Enum.with_index()
    |> Enum.reduce({[], nil}, fn {{token, tag, probability}, index}, {done, current} ->
      case {tag, current} do
        {"B-" <> type, nil} ->
          {done, open(type, token, probability, index)}

        {"B-" <> type, open} ->
          {[close(open) | done], open(type, token, probability, index)}

        {"I-" <> type, %{type: type} = open} ->
          {done, extend(open, token, probability, index)}

        {"I-" <> type, open} ->
          raise ArgumentError,
                "EntityTrainer: #{inspect(tag)} at token #{index} (#{inspect(token)}) continues " <>
                  "#{if open, do: "an entity of type #{inspect(open.type)}", else: "no entity"}; " <>
                  "an I-#{type} tag can only follow B-#{type} or I-#{type}"

        {"O", nil} ->
          {done, nil}

        {"O", open} ->
          {[close(open) | done], nil}
      end
    end)
    |> then(fn
      {done, nil} -> Enum.reverse(done)
      {done, open} -> Enum.reverse([close(open) | done])
    end)
  end

  defp open(type, token, probability, index),
    do: %{type: type, tokens: [token], probabilities: [probability], start_index: index, end_index: index}

  defp extend(open, token, probability, index),
    do: %{open | tokens: open.tokens ++ [token], probabilities: open.probabilities ++ [probability], end_index: index}

  defp close(open) do
    %{
      entity_type: open.type,
      value: Enum.join(open.tokens, " "),
      tokens: open.tokens,
      confidence: Enum.sum(open.probabilities) / length(open.probabilities),
      start_index: open.start_index,
      end_index: open.end_index
    }
  end

  # The emission and transition smoothing strengths, `{smoothing,
  # transition_smoothing}`, with the best exact span-and-type F1 across the
  # folds of the training rows, every pair of candidates tried; ties go to the
  # smaller strengths.
  defp select_smoothing(rows, sequences) do
    folds = Enum.map(rows, &:erlang.phash2(&1, @folds))
    indexed = Enum.zip(sequences, folds)

    totals =
      Enum.reduce(0..(@folds - 1), %{}, fn k, acc ->
        {held, train} = Enum.split_with(indexed, fn {_sequence, fold} -> fold == k end)
        counts = train |> Enum.map(&elem(&1, 0)) |> sequence_counts()
        held = Enum.map(held, fn {seq, _fold} -> {seq.tokens, spans_of(Enum.zip(seq.tokens, seq.tags))} end)
        transitions = Map.new(@transition_smoothing_candidates, &{&1, transition_weights(counts, &1)})

        for smoothing <- @smoothing_candidates, reduce: acc do
          acc ->
            emission = emission_model(counts, smoothing)

            for {transition_smoothing, weights} <- transitions, reduce: acc do
              acc ->
                model = Map.merge(emission, %{transition_weights: weights, transition_smoothing: transition_smoothing})
                fold_counts = score_fold(held, model)
                Map.update(acc, {smoothing, transition_smoothing}, fold_counts, &add_counts(&1, fold_counts))
            end
        end
      end)

    totals
    |> Enum.max_by(fn {{smoothing, transition_smoothing}, counts} -> {f1(counts), -smoothing, -transition_smoothing} end)
    |> elem(0)
  end

  # `{tp, fp, fn}` of the exact spans `model` decodes for the held rows, each
  # `{tokens, gold spans}`.
  defp score_fold(held, model) do
    Enum.reduce(held, {0, 0, 0}, fn {tokens, gold}, {tp, fp, fn_} ->
      predicted = tokens |> predict(model) |> spans_of()
      hit = MapSet.size(MapSet.intersection(gold, predicted))
      {tp + hit, fp + MapSet.size(predicted) - hit, fn_ + MapSet.size(gold) - hit}
    end)
  end

  defp add_counts({tp, fp, fn_}, {tp2, fp2, fn2}), do: {tp + tp2, fp + fp2, fn_ + fn2}

  defp f1({tp, fp, fn_}) do
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn_, 1)
    if precision + recall == 0, do: 0.0, else: 2 * precision * recall / (precision + recall)
  end

  # Entity spans as `{type, first token index, last token index}`.
  defp spans_of(token_tags) do
    token_tags
    |> Enum.map(fn {token, tag} -> {token, tag, 1.0} end)
    |> extract_entities_from_bio()
    |> MapSet.new(&{&1.entity_type, &1.start_index, &1.end_index})
  end

  defp train_sequence_model(sequences, smoothing, transition_smoothing) do
    counts = sequence_counts(sequences)

    counts
    |> emission_model(smoothing)
    |> Map.merge(%{
      transition_weights: transition_weights(counts, transition_smoothing),
      transition_smoothing: transition_smoothing
    })
  end

  # What training counts from the sequences, before any smoothing: the tags,
  # how often each occurs, each transition and each (feature, tag) pair.
  defp sequence_counts(sequences) do
    %{
      tags: sequences |> Enum.flat_map(& &1.tags) |> Enum.uniq() |> Enum.sort(),
      tag_counts: sequences |> Enum.flat_map(& &1.tags) |> Enum.frequencies(),
      transition_counts: transition_counts(sequences),
      feature_tag_counts: feature_tag_counts(sequences)
    }
  end

  defp emission_model(counts, smoothing) do
    total_tags = Enum.sum(Map.values(counts.tag_counts))
    tag_priors = Map.new(counts.tag_counts, fn {tag, count} -> {tag, count / total_tags} end)
    {feature_weights, feature_totals} = feature_weights(counts.feature_tag_counts, tag_priors, smoothing)

    %{
      tag_vocabulary: counts.tags |> Enum.with_index() |> Map.new(),
      feature_weights: feature_weights,
      feature_totals: feature_totals,
      smoothing: smoothing,
      tag_priors: tag_priors
    }
  end

  # Whether BIO allows `curr` after `prev`: an `I-x` only continues `B-x` or
  # `I-x`; anything else may follow anything.
  defp bio_transition?(prev, "I-" <> type), do: prev in ["B-" <> type, "I-" <> type]
  defp bio_transition?(_prev, _curr), do: true

  defp transition_counts(sequences) do
    Enum.reduce(sequences, %{}, fn seq, acc ->
      (["<START>" | seq.tags] ++ ["<END>"])
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.reduce(acc, fn [prev, curr], inner_acc ->
        unless bio_transition?(prev, curr) do
          raise ArgumentError,
                "EntityTrainer: the training sequence #{inspect(seq.tokens)} has #{prev} followed by #{curr}, " <>
                  "which BIO does not allow: #{inspect(seq.tags)}"
        end

        Map.update(inner_acc, {prev, curr}, 1, &(&1 + 1))
      end)
    end)
  end

  # P(curr | prev) for every transition BIO allows, and no entry for one it
  # does not: (n(prev, curr) + s * q(curr | prev)) / (n(prev) + s), where q is
  # how often `curr` follows any tag, normalized over the successors allowed
  # after `prev`. Each row sums to 1, since the training sequences hold only
  # allowed transitions.
  defp transition_weights(counts, strength) do
    successors = counts.tags ++ ["<END>"]
    sum_by = fn key -> Enum.reduce(counts.transition_counts, %{}, fn {pair, n}, acc -> Map.update(acc, key.(pair), n, &(&1 + n)) end) end
    out_of = sum_by.(&elem(&1, 0))
    into = sum_by.(&elem(&1, 1))

    Map.new(["<START>" | counts.tags], fn prev ->
      allowed = Enum.filter(successors, &bio_transition?(prev, &1))
      allowed_total = allowed |> Enum.map(&Map.fetch!(into, &1)) |> Enum.sum()
      seen = Map.fetch!(out_of, prev)

      {prev,
       Map.new(allowed, fn curr ->
         base = Map.fetch!(into, curr) / allowed_total
         {curr, (Map.get(counts.transition_counts, {prev, curr}, 0) + strength * base) / (seen + strength)}
       end)}
    end)
  end

  defp feature_tag_counts(sequences) do
    Enum.reduce(sequences, %{}, fn seq, acc ->
      seq.tokens
      |> Enum.zip(seq.tags)
      |> Enum.with_index()
      |> Enum.reduce(acc, fn {{token, tag}, idx}, inner_acc ->
        features = extract_token_features(token, seq.tokens, idx)

        Enum.reduce(features, inner_acc, fn feature, feat_acc ->
          Map.update(feat_acc, feature, %{tag => 1}, fn tag_counts ->
            Map.update(tag_counts, tag, 1, &(&1 + 1))
          end)
        end)
      end)
    end)
  end

  # P(tag | feature), smoothed toward the tag prior: (n(f, tag) + s * P(tag)) /
  # (n(f) + s). Only observed (feature, tag) pairs are stored; the rest follow
  # from `feature_totals` at decode time.
  defp feature_weights(feature_tag_counts, tag_priors, smoothing) do
    totals = Map.new(feature_tag_counts, fn {feature, tag_counts} -> {feature, Enum.sum(Map.values(tag_counts))} end)

    weights =
      Map.new(feature_tag_counts, fn {feature, tag_counts} ->
        total = Map.fetch!(totals, feature)

        {feature,
         Map.new(tag_counts, fn {tag, count} ->
           {tag, (count + smoothing * Map.fetch!(tag_priors, tag)) / (total + smoothing)}
         end)}
      end)

    {weights, totals}
  end

  defp extract_token_features(token, all_tokens, idx) do
    features = [
      "token:#{String.downcase(token)}",
      "prefix2:#{String.slice(String.downcase(token), 0, 2)}",
      "prefix3:#{String.slice(String.downcase(token), 0, 3)}",
      "suffix2:#{String.slice(String.downcase(token), -2, 2)}",
      "suffix3:#{String.slice(String.downcase(token), -3, 3)}",
      if(capitalized?(token)) do
        "is_capitalized"
      else
        "not_capitalized"
      end,
      if(all_caps?(token)) do
        "is_all_caps"
      else
        "not_all_caps"
      end,
      if(has_digit?(token)) do
        "has_digit"
      else
        "no_digit"
      end,
      if(all_digits?(token)) do
        "is_number"
      else
        "not_number"
      end,
      if(idx == 0) do
        "is_first"
      else
        "not_first"
      end,
      if(idx == length(all_tokens) - 1) do
        "is_last"
      else
        "not_last"
      end
    ]

    prev_features =
      if idx > 0 do
        prev_token = Enum.at(all_tokens, idx - 1)
        ["prev_token:#{String.downcase(prev_token)}"]
      else
        ["prev_token:<START>"]
      end

    next_features =
      if idx < length(all_tokens) - 1 do
        next_token = Enum.at(all_tokens, idx + 1)
        ["next_token:#{String.downcase(next_token)}"]
      else
        ["next_token:<END>"]
      end

    Enum.filter(features ++ prev_features ++ next_features, &(&1 != nil))
  end

  defp capitalized?(token) do
    first = String.first(token) || ""
    first == String.upcase(first) and first != String.downcase(first)
  end

  defp all_caps?(token) do
    token == String.upcase(token) and token != String.downcase(token)
  end

  defp has_digit?(token) do
    Enum.any?(String.graphemes(token), fn g ->
      g >= "0" and g <= "9"
    end)
  end

  defp all_digits?(token) do
    token != "" and
      Enum.all?(String.graphemes(token), fn g ->
        g >= "0" and g <= "9"
      end)
  end

  defp model_tags(model), do: model.tag_vocabulary |> Map.keys() |> Enum.sort()

  # The log transition scores the decode reads, for the transitions BIO allows
  # only: `from_start` (tag => score), `incoming` and `outgoing` (tag => [{other
  # tag, score}]) and `to_end` (tag => score). A transition BIO does not allow
  # is in none of them, so no path takes it. A model without a probability for
  # an allowed transition raises.
  defp log_transitions(model, tags) do
    weights = model.transition_weights

    pairs =
      for prev <- tags, curr <- tags, bio_transition?(prev, curr) do
        {prev, curr, log_transition!(weights, prev, curr)}
      end

    %{
      from_start: for(tag <- tags, bio_transition?("<START>", tag), into: %{}, do: {tag, log_transition!(weights, "<START>", tag)}),
      incoming: Map.merge(Map.new(tags, &{&1, []}), Enum.group_by(pairs, &elem(&1, 1), fn {prev, _curr, score} -> {prev, score} end)),
      outgoing: Map.merge(Map.new(tags, &{&1, []}), Enum.group_by(pairs, &elem(&1, 0), fn {_prev, curr, score} -> {curr, score} end)),
      to_end: Map.new(tags, &{&1, log_transition!(weights, &1, "<END>")})
    }
  end

  defp log_transition!(weights, prev, curr) do
    case weights do
      %{^prev => %{^curr => p}} when is_float(p) and p > 0 ->
        :math.log(p)

      _ ->
        raise ArgumentError,
              "EntityTrainer: the model gives no probability to #{prev} -> #{curr}, a transition BIO allows; " <>
                "retrain with `mix train --world <id>`"
    end
  end

  # Each token's log emission score under every tag.
  defp emission_table(tokens, tags, model) do
    tokens
    |> Enum.with_index()
    |> Enum.map(fn {token, idx} ->
      features = extract_token_features(token, tokens, idx)
      Map.new(tags, &{&1, emission_log(features, &1, model)})
    end)
  end

  # The best tag sequence over the transitions BIO allows. A column holds only
  # the tags some allowed path reaches; ties go to the tag that sorts last.
  defp viterbi_decode([], _model), do: []

  defp viterbi_decode(tokens, model) do
    tags = model_tags(model)
    transitions = log_transitions(model, tags)
    [first | rest] = emission_table(tokens, tags, model)

    start = Map.new(transitions.from_start, fn {tag, score} -> {tag, score + Map.fetch!(first, tag)} end)

    {last, backpointers} =
      Enum.reduce(rest, {start, []}, fn emission, {previous, backpointers} ->
        {column, backpointer} = viterbi_column(previous, emission, transitions.incoming, tags)
        {column, [backpointer | backpointers]}
      end)

    {_score, best} = last |> Enum.map(fn {tag, score} -> {score + Map.fetch!(transitions.to_end, tag), tag} end) |> Enum.max()

    # Newest first: the walk starts from the last token's backpointers.
    Enum.reduce(backpointers, [best], fn backpointer, [tag | _] = path -> [Map.fetch!(backpointer, tag) | path] end)
  end

  defp viterbi_column(previous, emission, incoming, tags) do
    Enum.reduce(tags, {%{}, %{}}, fn curr, {column, backpointer} ->
      case for({prev, score} <- Map.fetch!(incoming, curr), Map.has_key?(previous, prev), do: {Map.fetch!(previous, prev) + score, prev}) do
        [] ->
          {column, backpointer}

        candidates ->
          {score, prev} = Enum.max(candidates)
          {Map.put(column, curr, score + Map.fetch!(emission, curr)), Map.put(backpointer, curr, prev)}
      end
    end)
  end

  # Forward-backward over the same emission and transition scores Viterbi uses:
  # for each token, the probability of each tag given the whole sequence. A tag
  # no allowed path reaches at a token has probability 0 there and no entry.
  defp posterior_marginals(tokens, model) do
    tags = model_tags(model)
    transitions = log_transitions(model, tags)
    [first | rest] = emission_table(tokens, tags, model)

    alpha0 = Map.new(transitions.from_start, fn {tag, score} -> {tag, score + Map.fetch!(first, tag)} end)

    alphas =
      Enum.scan(rest, alpha0, fn emission, previous ->
        Enum.reduce(tags, %{}, fn curr, column ->
          case for({prev, score} <- Map.fetch!(transitions.incoming, curr), Map.has_key?(previous, prev), do: Map.fetch!(previous, prev) + score) do
            [] -> column
            terms -> Map.put(column, curr, log_sum_exp(terms) + Map.fetch!(emission, curr))
          end
        end)
      end)
      |> then(&[alpha0 | &1])

    betas =
      rest
      |> Enum.reverse()
      |> Enum.scan(transitions.to_end, fn emission, next ->
        Map.new(tags, fn prev ->
          {prev, log_sum_exp(for {curr, score} <- Map.fetch!(transitions.outgoing, prev), do: score + Map.fetch!(emission, curr) + Map.fetch!(next, curr))}
        end)
      end)
      |> Enum.reverse()
      |> Kernel.++([transitions.to_end])

    # Each token's scores are normalized by their own log-sum, which is the
    # sequence's log-likelihood at every token; summing per token keeps every
    # probability at most 1 under floating-point rounding.
    Enum.zip_with(alphas, betas, fn alpha, beta ->
      scores = Map.new(alpha, fn {tag, a} -> {tag, a + Map.fetch!(beta, tag)} end)
      log_z = scores |> Map.values() |> log_sum_exp()
      Map.new(scores, fn {tag, score} -> {tag, :math.exp(score - log_z)} end)
    end)
  end

  defp log_sum_exp(values) do
    max = Enum.max(values)
    max + :math.log(Enum.sum(Enum.map(values, &:math.exp(&1 - max))))
  end

  # The log emission score of `tag`: naive Bayes over the token's features. A
  # feature never seen in training says nothing and contributes 0.
  defp emission_log(features, tag, model) do
    prior = Map.fetch!(model.tag_priors, tag)
    log_prior = :math.log(prior)

    Enum.reduce(features, log_prior, fn feature, acc ->
      case Map.get(model.feature_weights, feature) do
        nil ->
          acc

        tag_probs ->
          p =
            Map.get_lazy(tag_probs, tag, fn ->
              model.smoothing * prior / (Map.fetch!(model.feature_totals, feature) + model.smoothing)
            end)

          acc + :math.log(p) - log_prior
      end
    end)
  end

  defp save_model(model, models_path) do
    File.mkdir_p!(models_path)

    model_path = Path.join(models_path, "entity_model.term")
    binary = Brain.ML.ModelStore.serialize(model)

    case File.write(model_path, binary) do
      :ok ->
        Logger.info("Entity model saved", %{path: model_path})
        {:ok, model}

      {:error, reason} ->
        {:error, "Failed to save model: #{reason}"}
    end
  end
end
