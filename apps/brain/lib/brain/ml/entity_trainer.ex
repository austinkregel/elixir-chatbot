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
  @folds 5

  alias Brain.ML.{EvaluationStore, NerGold}
  require Logger

  @type bio_tag :: String.t()

  @type entity_model :: %{
          tag_vocabulary: %{bio_tag() => integer()},
          feature_weights: %{String.t() => %{bio_tag() => float()}},
          transition_weights: %{bio_tag() => %{bio_tag() => float()}},
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
    smoothing = select_smoothing(rows, sequences)
    model = train_sequence_model(sequences, smoothing)

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

  @doc "Train entity recognition model and save to disk.\n\n## Options\n  - models_path: Override the default models output path\n"
  def train_and_save(opts \\ []) do
    models_path =
      Keyword.get(opts, :models_path) ||
        Application.get_env(:brain, :ml)[:models_path] ||
        Brain.priv_path("ml_models")

    {:ok, model} = train()
    save_model(model, models_path)
  end

  @doc "Load a trained entity model from disk.\n"
  def load_model do
    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
    model_path = Path.join(models_path, "entity_model.term")

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
  is the mean posterior of its tokens.
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

        {"I-" <> type, nil} ->
          {done, open(type, token, probability, index)}

        {"I-" <> type, open} ->
          {[close(open) | done], open(type, token, probability, index)}

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

  # The smoothing strength with the best exact span-and-type F1 across the
  # folds of the training rows; ties go to the smaller strength.
  defp select_smoothing(rows, sequences) do
    folds = Enum.map(rows, &:erlang.phash2(&1, @folds))
    indexed = Enum.zip([rows, sequences, folds])

    @smoothing_candidates
    |> Enum.map(fn smoothing ->
      counts =
        Enum.reduce(0..(@folds - 1), {0, 0, 0}, fn k, acc ->
          {held, train} = Enum.split_with(indexed, fn {_r, _s, fold} -> fold == k end)
          model = train_sequence_model(Enum.map(train, &elem(&1, 1)), smoothing)

          Enum.reduce(held, acc, fn {_row, seq, _fold}, {tp, fp, fn_} ->
            gold = spans_of(Enum.zip(seq.tokens, seq.tags))
            predicted = seq.tokens |> predict(model) |> spans_of()
            hit = MapSet.size(MapSet.intersection(gold, predicted))
            {tp + hit, fp + MapSet.size(predicted) - hit, fn_ + MapSet.size(gold) - hit}
          end)
        end)

      {smoothing, f1(counts)}
    end)
    |> Enum.max_by(fn {smoothing, f1} -> {f1, -smoothing} end)
    |> elem(0)
  end

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

  defp train_sequence_model(sequences, smoothing) do
    all_tags =
      sequences
      |> Enum.flat_map(& &1.tags)
      |> Enum.uniq()
      |> Enum.sort()

    tag_vocabulary =
      all_tags
      |> Enum.with_index()
      |> Enum.into(%{})

    tag_counts =
      sequences
      |> Enum.flat_map(& &1.tags)
      |> Enum.frequencies()

    total_tags = Enum.sum(Map.values(tag_counts))

    tag_priors =
      Enum.into(tag_counts, %{}, fn {tag, count} ->
        {tag, count / total_tags}
      end)

    transition_counts = calculate_transition_counts(sequences)
    transition_weights = normalize_transition_counts(transition_counts, all_tags)
    {feature_weights, feature_totals} = calculate_feature_weights(sequences, tag_priors, smoothing)

    %{
      tag_vocabulary: tag_vocabulary,
      feature_weights: feature_weights,
      feature_totals: feature_totals,
      smoothing: smoothing,
      transition_weights: transition_weights,
      tag_priors: tag_priors
    }
  end

  defp calculate_transition_counts(sequences) do
    Enum.reduce(sequences, %{}, fn seq, acc ->
      tags = ["<START>" | seq.tags] ++ ["<END>"]

      tags
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.reduce(acc, fn [prev, curr], inner_acc ->
        key = {prev, curr}
        Map.update(inner_acc, key, 1, &(&1 + 1))
      end)
    end)
  end

  defp normalize_transition_counts(counts, all_tags) do
    all_tags_with_markers = ["<START>" | all_tags] ++ ["<END>"]
    grouped = Enum.group_by(counts, fn {{prev, _curr}, _count} -> prev end)

    Enum.reduce(all_tags_with_markers, %{}, fn prev_tag, acc ->
      transitions = Map.get(grouped, prev_tag, [])
      total = Enum.sum(Enum.map(transitions, fn {_, count} -> count end))

      if total > 0 do
        probs =
          Enum.into(transitions, %{}, fn {{_prev, curr}, count} ->
            {curr, count / total}
          end)

        Map.put(acc, prev_tag, probs)
      else
        uniform = 1.0 / length(all_tags_with_markers)
        probs = Enum.into(all_tags_with_markers, %{}, fn tag -> {tag, uniform} end)
        Map.put(acc, prev_tag, probs)
      end
    end)
  end

  # P(tag | feature), smoothed toward the tag prior: (n(f, tag) + s * P(tag)) /
  # (n(f) + s). Only observed (feature, tag) pairs are stored; the rest follow
  # from `feature_totals` at decode time.
  defp calculate_feature_weights(sequences, tag_priors, smoothing) do
    feature_tag_counts =
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

  defp model_tags(model), do: model.tag_vocabulary |> Map.keys() |> Enum.reject(&(&1 in ["<START>", "<END>"]))

  defp viterbi_decode(tokens, model) do
    if tokens == [] do
      []
    else
      tags = model_tags(model)

      {initial_viterbi, initial_backpointer} =
        initialize_viterbi(Enum.at(tokens, 0), tokens, 0, tags, model)

      {final_viterbi, backpointers} =
        tokens
        |> Enum.with_index()
        |> Enum.drop(1)
        |> Enum.reduce({initial_viterbi, [initial_backpointer]}, fn {token, idx},
                                                                    {prev_viterbi, bps} ->
          {new_viterbi, new_bp} = viterbi_step(token, tokens, idx, prev_viterbi, tags, model)
          {new_viterbi, [new_bp | bps]}
        end)

      # Newest first: the walk starts from the last token's backpointers.
      backtrack(final_viterbi, backpointers, tags)
    end
  end

  defp initialize_viterbi(token, all_tokens, idx, tags, model) do
    features = extract_token_features(token, all_tokens, idx)

    viterbi =
      Enum.into(tags, %{}, fn tag ->
        transition = get_transition_prob("<START>", tag, model)
        score = emission_log(features, tag, model) + :math.log(max(transition, 1.0e-10))
        {tag, score}
      end)

    backpointer = Enum.into(tags, %{}, fn tag -> {tag, nil} end)

    {viterbi, backpointer}
  end

  defp viterbi_step(token, all_tokens, idx, prev_viterbi, tags, model) do
    features = extract_token_features(token, all_tokens, idx)

    {viterbi, backpointer} =
      Enum.reduce(tags, {%{}, %{}}, fn curr_tag, {vit_acc, bp_acc} ->
        emission = emission_log(features, curr_tag, model)

        {best_score, best_prev_tag} =
          Enum.reduce(tags, {nil, nil}, fn prev_tag, {best, best_tag} ->
            prev_score = Map.get(prev_viterbi, prev_tag, -1000)
            transition = get_transition_prob(prev_tag, curr_tag, model)

            score = prev_score + :math.log(max(transition, 1.0e-10)) + emission

            if best == nil or score > best do
              {score, prev_tag}
            else
              {best, best_tag}
            end
          end)

        {Map.put(vit_acc, curr_tag, best_score), Map.put(bp_acc, curr_tag, best_prev_tag)}
      end)

    {viterbi, backpointer}
  end

  # Forward-backward over the same emission and transition scores Viterbi uses:
  # for each token, the probability of each tag given the whole sequence.
  defp posterior_marginals(tokens, model) do
    tags = model_tags(model)

    emissions =
      tokens
      |> Enum.with_index()
      |> Enum.map(fn {token, idx} ->
        features = extract_token_features(token, tokens, idx)
        Map.new(tags, fn tag -> {tag, emission_log(features, tag, model)} end)
      end)

    log_t = fn prev, curr -> :math.log(max(get_transition_prob(prev, curr, model), 1.0e-10)) end

    [first | rest] = emissions
    alpha0 = Map.new(tags, fn tag -> {tag, log_t.("<START>", tag) + first[tag]} end)

    alphas =
      Enum.scan(rest, alpha0, fn emission, prev ->
        Map.new(tags, fn curr ->
          {curr, log_sum_exp(Enum.map(tags, fn p -> prev[p] + log_t.(p, curr) end)) + emission[curr]}
        end)
      end)
      |> then(&[alpha0 | &1])

    last_beta = Map.new(tags, &{&1, 0.0})

    betas =
      rest
      |> Enum.reverse()
      |> Enum.scan(last_beta, fn emission, next ->
        Map.new(tags, fn prev ->
          {prev, log_sum_exp(Enum.map(tags, fn c -> log_t.(prev, c) + emission[c] + next[c] end))}
        end)
      end)
      |> Enum.reverse()
      |> Kernel.++([last_beta])

    log_z = alphas |> List.last() |> Map.values() |> log_sum_exp()

    Enum.zip_with(alphas, betas, fn alpha, beta ->
      Map.new(tags, fn tag -> {tag, :math.exp(alpha[tag] + beta[tag] - log_z)} end)
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

  defp get_transition_prob(prev_tag, curr_tag, model) do
    case Map.get(model.transition_weights, prev_tag) do
      nil -> 0.01
      probs -> Map.get(probs, curr_tag, 0.01)
    end
  end

  defp backtrack(final_viterbi, backpointers, tags) do
    {_best_score, best_tag} =
      Enum.reduce(tags, {nil, nil}, fn tag, {best, best_tag} ->
        score = Map.get(final_viterbi, tag, -1000)

        if best == nil or score > best do
          {score, tag}
        else
          {best, best_tag}
        end
      end)

    backtrack_impl(backpointers, best_tag, [best_tag])
  end

  defp backtrack_impl([], _current_tag, path) do
    path
  end

  defp backtrack_impl([bp | rest], current_tag, path) do
    prev_tag = Map.get(bp, current_tag)

    if prev_tag != nil do
      backtrack_impl(rest, prev_tag, [prev_tag | path])
    else
      backtrack_impl(rest, current_tag, path)
    end
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
