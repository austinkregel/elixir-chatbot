defmodule Brain.ML.POSTagger do
  @moduledoc """
  Part-of-speech tagging with a BiLSTM over words and characters.

  The architecture follows Plank, Søgaard & Goldberg (2016, "Multilingual
  Part-of-Speech Tagging with Bidirectional LSTM Models and Auxiliary Loss"):

  - each word's characters run through a character-level BiLSTM, so a word
    never seen in training still has a representation from its spelling;
  - that joins the word's embedding (lowercased) and a sentence-level BiLSTM
    reads the whole sentence in both directions;
  - a softmax predicts the tag, and an auxiliary head predicts the word's
    training-frequency bin, which the paper found improves tagging of rare
    and unseen words.

  Tags are Universal Dependencies v1 (`valid_tags/0`). Training reads
  fixtures in the task 086 format; `mix pos.train` trains on the UD English
  Web Treebank and records the model's accuracy and confusion matrix on the
  held-out test split alongside it.

  ## Usage

      {:ok, model} = POSTagger.load_model()
      POSTagger.predict(["I", "may", "go"], model)
      # => [{"I", "PRON"}, {"may", "AUX"}, {"go", "VERB"}]

  Hyperparameters are in `config :brain, :pos_tagger`. Training is
  deterministic from the training seed (`Brain.ML.TrainingSeed`).
  """

  require Logger

  alias Brain.ML.{ModelStore, TrainingSeed}

  @pos_tags ~w(
    NOUN PROPN VERB AUX ADJ ADV PRON DET ADP
    CONJ PART NUM INTJ PUNCT SYM X
  )

  # Version of the saved model's shape. A file of another version is refused.
  @format 2

  @pad 0
  @unk 1

  # Sentences are padded to one of these lengths, so the compiled network is
  # reused rather than recompiled for every distinct sentence length.
  @buckets [8, 16, 32, 64, 128, 256]

  @type pos_tag :: String.t()
  @type training_sequence :: %{tokens: [String.t()], tags: [pos_tag()]}

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------

  @doc "The tags the tagger can emit (Universal Dependencies v1)."
  def valid_tags, do: @pos_tags

  @doc """
  Trains a tagger on `sequences` (`%{tokens: [...], tags: [...]}`).

  ## Options

  - `:dev` -- sequences to stop on: after each epoch the tagger is measured
    on them, and training stops when accuracy has not improved for
    `patience` epochs; the best epoch's parameters are kept. With
    `patience: nil` training never stops early and runs `max_epochs`, still
    keeping the best epoch. Without `:dev`, training runs exactly
    `max_epochs` epochs.
  - `:config` -- keyword list overriding `config :brain, :pos_tagger`.
  - `:seed` -- training seed (default `Brain.ML.TrainingSeed.get!/0`).
  - `:inputs` -- provenance of the training data (task 086 `producer.inputs`
    entries), recorded in the model.
  - `:on_epoch` -- `fn progress, model_at -> any end`, called after every
    epoch. `progress` is `%{epoch, loss, dev_accuracy, best_epoch,
    best_dev_accuracy, improved?, elapsed_ms}`: `loss` is the epoch's mean
    training loss, `dev_accuracy` is nil without `:dev`, and `improved?` is
    true when this epoch is the new best. `model_at.()` builds the model as
    it stands after this epoch, in the shape `train/2` returns, for saving
    a snapshot; it copies the parameters, so call it only when needed.

  Every sequence must have as many tags as tokens, all in `valid_tags/0`;
  anything else raises rather than being filtered out.
  """
  @spec train([training_sequence()], keyword()) :: {:ok, map()} | {:error, String.t()}
  def train(sequences, opts \\ [])

  def train([], _opts), do: {:error, "No training sequences provided"}

  def train(sequences, opts) when is_list(sequences) do
    config = config(opts)
    seed = Keyword.get_lazy(opts, :seed, &TrainingSeed.get!/0)
    sequences = Enum.map(sequences, &validate_sequence!/1)
    dev = opts |> Keyword.get(:dev, []) |> Enum.map(&validate_sequence!/1)

    started = System.monotonic_time(:millisecond)
    :telemetry.execute([:chat_bot, :ml, :train, :start], %{sequence_count: length(sequences)}, %{model: :pos_tagger})

    vocab = build_vocab(sequences)
    network = build_network(vocab, config, seed)
    {_init_fn, predict_fn} = Axon.build(network, compiler: EXLA, mode: :inference)

    # The saved model for trained parameters and a training summary.
    assemble = fn model_state, training ->
      %{
        format: @format,
        config: Map.new(config),
        vocab: vocab,
        params: transfer(model_state, Nx.BinaryBackend),
        training:
          Map.merge(training, %{
            seed: seed,
            sequences: length(sequences),
            inputs: Keyword.get(opts, :inputs, [])
          }),
        evaluation: nil
      }
    end

    on_epoch = Keyword.get(opts, :on_epoch, fn _progress, _model_at -> :ok end)
    {model_state, training} = fit(network, predict_fn, sequences, dev, vocab, config, seed, on_epoch, assemble)
    model = assemble.(model_state, training)

    :telemetry.execute(
      [:chat_bot, :ml, :train, :stop],
      %{duration_ms: System.monotonic_time(:millisecond) - started, sequence_count: length(sequences)},
      %{model: :pos_tagger, success: true}
    )

    {:ok, model}
  end

  @doc "Tags `tokens`, returning `[{token, tag}]`."
  @spec predict([String.t()], map()) :: [{String.t(), pos_tag()}]
  def predict([], _model), do: []
  def predict(tokens, model) when is_list(tokens) and is_map(model), do: Enum.zip(tokens, predict_tags(tokens, model))

  @doc "Tags `tokens`, returning only the tags."
  @spec predict_tags([String.t()], map()) :: [pos_tag()]
  def predict_tags([], _model), do: []

  def predict_tags(tokens, model) when is_list(tokens) and is_map(model) do
    check_format!(model)
    [tags] = decode_batch(runtime(model), model, [tokens])
    tags
  end

  @doc """
  Measures `model` on tagged `sequences`: token accuracy, accuracy on words
  outside the training vocabulary, per-tag recall and precision, and the
  confusion matrix (`%{gold => %{predicted => count}}`).
  """
  @spec evaluate([training_sequence()], map()) :: map()
  def evaluate(sequences, model) do
    check_format!(model)
    sequences = Enum.map(sequences, &validate_sequence!/1)
    runtime = runtime(model)

    pairs =
      sequences
      |> Enum.chunk_every(model.config.batch_size)
      |> Enum.flat_map(fn batch ->
        predicted = decode_batch(runtime, model, Enum.map(batch, & &1.tokens))

        Enum.zip(batch, predicted)
        |> Enum.flat_map(fn {seq, tags} ->
          Enum.zip([seq.tokens, seq.tags, tags])
          |> Enum.map(fn {token, gold, pred} ->
            {gold, pred, not Map.has_key?(model.vocab.words, String.downcase(token))}
          end)
        end)
      end)

    report(pairs)
  end

  @doc "Saves `model` (optionally with its evaluation) to `path`."
  def save_model(model, path \\ nil) do
    check_format!(model)
    model_path = path || model_path()
    File.mkdir_p!(Path.dirname(model_path))

    case File.write(model_path, ModelStore.serialize(model)) do
      :ok ->
        Logger.info("POS model saved to #{model_path}")
        {:ok, model_path}

      {:error, reason} ->
        {:error, "Failed to save model: #{reason}"}
    end
  end

  @doc """
  Loads the tagger from `path` (default: the configured models path).

  The loaded model is cached by path and file modification time, so the
  many callers that load before every prediction read the file once.
  """
  def load_model(path \\ nil) do
    model_path = path || model_path()
    if is_nil(path), do: ModelStore.ensure_local("pos_model.term", model_path)

    with {:ok, %File.Stat{mtime: mtime, size: size}} <- stat(model_path) do
      key = {__MODULE__, :model, model_path}

      case :persistent_term.get(key, nil) do
        {{^mtime, ^size}, model} ->
          {:ok, model}

        _ ->
          with {:ok, binary} <- read(model_path),
               {:ok, model} <- decode(binary, model_path) do
            :persistent_term.put(key, {{mtime, size}, model})
            {:ok, model}
          end
      end
    end
  end

  @doc "The current model from the configured path. See `load_model/1`."
  def get_model, do: load_model()

  @doc "True when a model file exists at `path` (default: the configured path)."
  def model_exists?(path \\ nil), do: File.exists?(path || model_path())

  # ---------------------------------------------------------------------------
  # Data
  # ---------------------------------------------------------------------------

  defp config(opts) do
    base = Application.fetch_env!(:brain, :pos_tagger)
    Keyword.merge(base, Keyword.get(opts, :config, []))
  end

  defp validate_sequence!(seq) do
    tokens = Map.get(seq, :tokens) || Map.get(seq, "tokens")
    tags = Map.get(seq, :tags) || Map.get(seq, "tags")

    unless is_list(tokens) and tokens != [] and is_list(tags) and length(tokens) == length(tags) do
      raise ArgumentError, "POSTagger: a sequence needs as many tags as tokens: #{inspect(seq)}"
    end

    case Enum.reject(tags, &(&1 in @pos_tags)) do
      [] -> %{tokens: tokens, tags: tags}
      bad -> raise ArgumentError, "POSTagger: tags outside the tagset #{inspect(Enum.uniq(bad))}"
    end
  end

  defp build_vocab(sequences) do
    word_counts =
      sequences
      |> Enum.flat_map(& &1.tokens)
      |> Enum.frequencies_by(&String.downcase/1)

    chars =
      sequences
      |> Enum.flat_map(& &1.tokens)
      |> Enum.flat_map(&String.graphemes/1)
      |> Enum.uniq()
      |> Enum.sort()

    words = word_counts |> Map.keys() |> Enum.sort()

    # Frequency bin as in Plank et al.: floor of the natural log of the count.
    bins = Map.new(word_counts, fn {w, n} -> {w, floor(:math.log(n))} end)

    %{
      words: words |> Enum.with_index(2) |> Map.new(),
      chars: chars |> Enum.with_index(2) |> Map.new(),
      word_counts: word_counts,
      freq_bins: bins,
      freq_bin_count: (bins |> Map.values() |> Enum.max()) + 1,
      tags: @pos_tags,
      tag_index: @pos_tags |> Enum.with_index() |> Map.new()
    }
  end

  defp bucket(n) do
    Enum.find(@buckets, &(&1 >= n)) || div(n + 255, 256) * 256
  end

  # Keeps a word's first and last characters when it is longer than
  # `max_chars`: prefix and suffix carry most of what spelling says about
  # a word's part of speech.
  defp word_chars(word, max_chars) do
    graphemes = String.graphemes(word)

    if length(graphemes) <= max_chars do
      graphemes
    else
      half = div(max_chars, 2)
      Enum.take(graphemes, half) ++ Enum.take(graphemes, -(max_chars - half))
    end
  end

  # Encodes sentences into padded input tensors. `unk?` decides, per
  # training token, whether to replace the word with the unknown token.
  defp encode(sentences, vocab, config, rows, unk? \\ fn _ -> false end) do
    len = sentences |> Enum.map(&length/1) |> Enum.max() |> bucket()
    max_chars = config[:max_word_chars]
    padded_rows = sentences ++ List.duplicate([], rows - length(sentences))

    word_ids =
      Enum.map(padded_rows, fn tokens ->
        ids =
          Enum.map(tokens, fn t ->
            lower = String.downcase(t)
            if unk?.(lower), do: @unk, else: Map.get(vocab.words, lower, @unk)
          end)

        ids ++ List.duplicate(@pad, len - length(ids))
      end)

    char_ids =
      Enum.map(padded_rows, fn tokens ->
        words =
          Enum.map(tokens, fn t ->
            ids = t |> word_chars(max_chars) |> Enum.map(&Map.get(vocab.chars, &1, @unk))
            ids ++ List.duplicate(@pad, max_chars - length(ids))
          end)

        words ++ List.duplicate(List.duplicate(@pad, max_chars), len - length(words))
      end)

    words = Nx.tensor(word_ids, type: :s64)
    chars = Nx.tensor(char_ids, type: :s64)

    %{
      "words" => words,
      "word_pad" => Nx.equal(words, @pad),
      "chars" => chars,
      "char_pad" => Nx.equal(chars, @pad)
    }
  end

  defp targets(batch, vocab, len, rows) do
    n_tags = length(vocab.tags)
    n_bins = vocab.freq_bin_count
    padded = batch ++ List.duplicate(%{tokens: [], tags: []}, rows - length(batch))

    one_hot = fn index, size -> for i <- 0..(size - 1), do: if(i == index, do: 1.0, else: 0.0) end
    zeros = fn size -> List.duplicate(0.0, size) end

    tags =
      Enum.map(padded, fn seq ->
        rows = Enum.map(seq.tags, &one_hot.(Map.fetch!(vocab.tag_index, &1), n_tags))
        rows ++ List.duplicate(zeros.(n_tags), len - length(rows))
      end)

    freq =
      Enum.map(padded, fn seq ->
        rows = Enum.map(seq.tokens, &one_hot.(Map.fetch!(vocab.freq_bins, String.downcase(&1)), n_bins))
        rows ++ List.duplicate(zeros.(n_bins), len - length(rows))
      end)

    %{tags: Nx.tensor(tags, type: :f32), freq: Nx.tensor(freq, type: :f32)}
  end

  # ---------------------------------------------------------------------------
  # Network
  # ---------------------------------------------------------------------------

  defp build_network(vocab, config, seed) do
    word_vocab = map_size(vocab.words) + 2
    char_vocab = map_size(vocab.chars) + 2
    max_chars = config[:max_word_chars]

    words = Axon.input("words", shape: {nil, nil})
    word_pad = Axon.input("word_pad", shape: {nil, nil})
    chars = Axon.input("chars", shape: {nil, nil, max_chars})
    char_pad = Axon.input("char_pad", shape: {nil, nil, max_chars})

    word_vec = Axon.embedding(words, word_vocab, config[:word_dim], name: "word_embedding")

    flat = fn t ->
      {b, s, c} = Nx.shape(t)
      Nx.reshape(t, {b * s, c})
    end

    chars_flat = Axon.nx(chars, flat, name: "chars_flat")
    char_pad_flat = Axon.nx(char_pad, flat, name: "char_pad_flat")
    char_emb = Axon.embedding(chars_flat, char_vocab, config[:char_dim], name: "char_embedding")

    char_fw = final_state(char_emb, char_pad_flat, config[:char_hidden], "char_lstm_fw", seed)
    char_bw = final_state(reverse(char_emb), reverse(char_pad_flat), config[:char_hidden], "char_lstm_bw", seed)

    char_vec =
      Axon.layer(
        fn char_states, word_ids, _opts ->
          {b, s} = Nx.shape(word_ids)
          Nx.reshape(char_states, {b, s, Nx.axis_size(char_states, 1)})
        end,
        [Axon.concatenate(char_fw, char_bw, axis: -1), words],
        name: "char_vec"
      )

    x =
      Axon.concatenate(word_vec, char_vec, axis: -1)
      |> Axon.dropout(rate: config[:dropout], seed: seed, name: "input_dropout")

    {fw_seq, _} = Axon.lstm(x, config[:word_hidden], lstm_opts("word_lstm_fw", word_pad, seed))
    {bw_seq, _} = Axon.lstm(reverse(x), config[:word_hidden], lstm_opts("word_lstm_bw", reverse(word_pad), seed))

    h =
      Axon.concatenate(fw_seq, reverse(bw_seq), axis: -1)
      |> Axon.dropout(rate: config[:dropout], seed: seed, name: "output_dropout")

    Axon.container(%{
      tags: h |> Axon.dense(length(@pos_tags), name: "tag_output") |> Axon.softmax(name: "tag_softmax"),
      freq: h |> Axon.dense(vocab.freq_bin_count, name: "freq_output") |> Axon.softmax(name: "freq_softmax")
    })
  end

  # The last real step's hidden state: masked (padded) steps carry the
  # state through unchanged, so with right-padding the final state is the
  # state after the last real character.
  defp final_state(x, pad, units, name, seed) do
    {_seq, {_cell, hidden}} = Axon.lstm(x, units, lstm_opts(name, pad, seed))
    hidden
  end

  # Zero initial states, so a sentence is tagged the same way every time.
  defp lstm_opts(name, pad, seed),
    do: [name: name, mask: pad, seed: seed, recurrent_initializer: :zeros]

  defp reverse(x), do: Axon.nx(x, &Nx.reverse(&1, axes: [1]))

  defp loss(aux_weight) do
    fn y_true, y_pred ->
      Nx.add(masked_ce(y_true.tags, y_pred.tags), Nx.multiply(aux_weight, masked_ce(y_true.freq, y_pred.freq)))
    end
  end

  # Cross-entropy averaged over real tokens only: padded positions have an
  # all-zero target row and contribute nothing.
  defp masked_ce(y_true, y_pred) do
    total = Nx.sum(Nx.multiply(y_true, Nx.log(Nx.add(y_pred, 1.0e-9))))
    count = Nx.max(Nx.sum(y_true), 1.0)
    Nx.negate(Nx.divide(total, count))
  end

  # ---------------------------------------------------------------------------
  # Training
  # ---------------------------------------------------------------------------

  defp fit(network, predict_fn, sequences, dev, vocab, config, seed, on_epoch, assemble) do
    batch_size = config[:batch_size]
    epochs = config[:max_epochs]
    per_epoch = sequences |> batches(batch_size, seed, 0) |> length()

    data =
      Stream.flat_map(1..epochs, fn epoch ->
        rand = TrainingSeed.state(seed + epoch)

        sequences
        |> batches(batch_size, seed, epoch)
        |> Enum.map_reduce(rand, fn batch, rand ->
          {unk_draws, rand} = draws(batch, rand)
          unk? = singleton_unk(vocab, config[:singleton_unk_rate], unk_draws)
          inputs = encode(Enum.map(batch, & &1.tokens), vocab, config, batch_size, unk?)
          len = Nx.axis_size(inputs["words"], 1)
          {{inputs, targets(batch, vocab, len, batch_size)}, rand}
        end)
        |> elem(0)
      end)

    best_key = {__MODULE__, :best, make_ref()}

    Process.put(best_key, %{
      accuracy: -1.0,
      epoch: 0,
      state: nil,
      epochs_run: 0,
      batches: 0,
      loss_total: 0.0,
      started: System.monotonic_time(:millisecond)
    })

    epoch_end = %{
      per_epoch: per_epoch,
      predict_fn: predict_fn,
      dev: dev,
      vocab: vocab,
      config: config,
      on_epoch: on_epoch,
      assemble: assemble
    }

    optimizer = Polaris.Optimizers.adam(learning_rate: config[:learning_rate])

    # Runs after every batch and acts only when a whole epoch's batches are
    # done. Axon's `every: n` filter fires after batch 1, n + 1, 2n + 1 --
    # one batch into each epoch -- so the batches are counted here instead.
    loop =
      network
      |> Axon.Loop.trainer(loss(config[:aux_loss_weight]), optimizer, seed: seed, log: 0)
      |> Axon.Loop.handle_event(:iteration_completed, fn state ->
        best = Process.get(best_key)
        batches = best.batches + 1
        Process.put(best_key, %{best | batches: batches})

        if rem(batches, per_epoch) == 0,
          do: end_of_epoch(state, epoch_end, best_key),
          else: {:continue, state}
      end)

    # strict?: false compiles the step once per batch shape -- once per length
    # bucket -- instead of refusing every shape but the first.
    last_state =
      Axon.Loop.run(loop, data, Axon.ModelState.empty(), epochs: 1, strict?: false, compiler: EXLA)
    best = Process.delete(best_key)

    if dev == [] do
      {last_state, %{epochs_run: best.epochs_run, best_epoch: best.epochs_run, dev_accuracy: nil}}
    else
      {best.state, %{epochs_run: best.epochs_run, best_epoch: best.epoch, dev_accuracy: best.accuracy}}
    end
  end

  # Runs once per epoch's worth of batches; the epoch number is the count of
  # those runs, kept alongside the best result.
  defp end_of_epoch(state, e, best_key) do
    model_state = state.step_state.model_state
    best = Process.get(best_key)
    epoch = best.epochs_run + 1

    # The loop runs as one Axon epoch, so its "loss" metric is the running
    # mean over every batch so far; the difference of running totals is this
    # epoch's share. At :iteration_completed `state.iteration` is still the
    # zero-based index of the batch just run, so the mean covers one more.
    loss_total = Nx.to_number(state.metrics["loss"]) * (state.iteration + 1)
    loss = (loss_total - best.loss_total) / e.per_epoch

    {accuracy, best} =
      if e.dev == [] do
        {nil, %{best | epoch: epoch}}
      else
        model = %{format: @format, config: Map.new(e.config), vocab: e.vocab, params: model_state}
        accuracy = e.dev |> evaluate_with(model, e.predict_fn) |> Map.fetch!(:accuracy)
        Logger.info("POSTagger: epoch #{epoch} dev accuracy #{Float.round(accuracy * 100, 2)}%")

        best =
          if accuracy > best.accuracy,
            do: %{best | accuracy: accuracy, epoch: epoch, state: transfer(model_state, Nx.BinaryBackend)},
            else: best

        {accuracy, best}
      end

    best = %{best | epochs_run: epoch, loss_total: loss_total}
    Process.put(best_key, best)

    progress = %{
      epoch: epoch,
      loss: loss,
      dev_accuracy: accuracy,
      best_epoch: best.epoch,
      best_dev_accuracy: if(e.dev == [], do: nil, else: best.accuracy),
      improved?: best.epoch == epoch,
      elapsed_ms: System.monotonic_time(:millisecond) - best.started
    }

    model_at = fn ->
      e.assemble.(model_state, %{
        epochs_run: epoch,
        best_epoch: epoch,
        dev_accuracy: accuracy
      })
    end

    e.on_epoch.(progress, model_at)

    patience = e.config[:patience]

    if e.dev != [] and is_integer(patience) and epoch - best.epoch >= patience,
      do: {:halt_loop, state},
      else: {:continue, state}
  end

  # One epoch's batches: sentences grouped by padded length, shuffled within
  # each length and in batch order, deterministically from seed and epoch.
  defp batches(sequences, batch_size, seed, epoch) do
    rand = TrainingSeed.state(seed * 1_000 + epoch)

    {grouped, rand} =
      sequences
      |> Enum.group_by(&bucket(length(&1.tokens)))
      |> Enum.sort()
      |> Enum.map_reduce(rand, fn {_len, group}, rand ->
        {shuffled, rand} = TrainingSeed.shuffle(group, rand)
        {Enum.chunk_every(shuffled, batch_size), rand}
      end)

    {batches, _rand} = grouped |> Enum.concat() |> TrainingSeed.shuffle(rand)
    batches
  end

  defp draws(batch, rand) do
    count = batch |> Enum.map(&length(&1.tokens)) |> Enum.sum()

    {draws, rand} =
      Enum.map_reduce(1..max(count, 1), rand, fn _, rand -> :rand.uniform_s(rand) end)

    {draws, rand}
  end

  # Replaces a word seen once in training with the unknown token with
  # probability `rate`, drawing from the batch's precomputed draws in order.
  defp singleton_unk(vocab, rate, draws) do
    counter = :counters.new(1, [])
    draws = List.to_tuple(draws)

    fn lower ->
      i = :counters.get(counter, 1)
      :counters.add(counter, 1, 1)
      draw = if i < tuple_size(draws), do: elem(draws, i), else: 1.0
      Map.get(vocab.word_counts, lower) == 1 and draw < rate
    end
  end

  # ---------------------------------------------------------------------------
  # Inference and evaluation
  # ---------------------------------------------------------------------------

  # The compiled predict function for a model's shape, built once and cached.
  defp runtime(model) do
    key = {__MODULE__, :runtime, :erlang.phash2({model.config, map_size(model.vocab.words), map_size(model.vocab.chars), model.vocab.freq_bin_count})}

    case :persistent_term.get(key, nil) do
      nil ->
        network = build_network(model.vocab, Map.to_list(model.config), model_seed(model))
        {_init, predict_fn} = Axon.build(network, compiler: EXLA, mode: :inference)
        :persistent_term.put(key, predict_fn)
        predict_fn

      predict_fn ->
        predict_fn
    end
  end

  defp model_seed(%{training: %{seed: seed}}), do: seed
  defp model_seed(_), do: 0

  defp evaluate_with(sequences, model, predict_fn) do
    pairs =
      sequences
      |> Enum.chunk_every(model.config.batch_size)
      |> Enum.flat_map(fn batch ->
        predicted = decode_batch(predict_fn, model, Enum.map(batch, & &1.tokens))

        Enum.zip(batch, predicted)
        |> Enum.flat_map(fn {seq, tags} ->
          Enum.zip(seq.tags, tags) |> Enum.map(fn {g, p} -> {g, p, false} end)
        end)
      end)

    report(pairs)
  end

  defp decode_batch(predict_fn, model, token_lists) do
    config = Map.to_list(model.config)
    rows = model.config.batch_size
    inputs = encode(token_lists, model.vocab, config, max(rows, length(token_lists)))
    %{tags: probs} = predict_fn.(model.params, inputs)
    best = probs |> Nx.argmax(axis: -1) |> Nx.to_list()

    token_lists
    |> Enum.zip(best)
    |> Enum.map(fn {tokens, ids} ->
      ids |> Enum.take(length(tokens)) |> Enum.map(&Enum.at(model.vocab.tags, &1))
    end)
  end

  defp report(pairs) do
    total = length(pairs)
    correct = Enum.count(pairs, fn {g, p, _} -> g == p end)
    oov = Enum.filter(pairs, fn {_, _, oov?} -> oov? end)

    confusion =
      Enum.reduce(pairs, %{}, fn {g, p, _}, acc ->
        Map.update(acc, g, %{p => 1}, &Map.update(&1, p, 1, fn n -> n + 1 end))
      end)

    predicted_counts = Enum.frequencies_by(pairs, fn {_, p, _} -> p end)

    per_tag =
      Map.new(confusion, fn {tag, row} ->
        n = row |> Map.values() |> Enum.sum()
        hit = Map.get(row, tag, 0)
        predicted = Map.get(predicted_counts, tag, 0)
        {tag, %{count: n, recall: hit / n, precision: if(predicted > 0, do: hit / predicted, else: 0.0)}}
      end)

    %{
      tokens: total,
      accuracy: if(total > 0, do: correct / total, else: 0.0),
      oov_tokens: length(oov),
      oov_accuracy:
        if(oov == [], do: nil, else: Enum.count(oov, fn {g, p, _} -> g == p end) / length(oov)),
      per_tag: per_tag,
      confusion: confusion
    }
  end

  # ---------------------------------------------------------------------------
  # Persistence
  # ---------------------------------------------------------------------------

  defp check_format!(%{format: @format}), do: :ok

  defp check_format!(model) do
    raise ArgumentError,
          "POSTagger: model format #{inspect(Map.get(model, :format))} is not #{@format}; " <>
            "retrain with mix pos.train"
  end

  # A copy, not a transfer: a transfer frees the source buffers, which the
  # training loop is still using when a best-epoch snapshot is taken.
  defp transfer(%Axon.ModelState{} = state, backend) do
    %{state | data: Nx.backend_copy(state.data, backend)}
  end

  defp stat(path) do
    case File.stat(path) do
      {:ok, stat} -> {:ok, stat}
      {:error, reason} -> {:error, "Failed to read model: #{reason}"}
    end
  end

  defp read(path) do
    case File.read(path) do
      {:ok, binary} -> {:ok, binary}
      {:error, reason} -> {:error, "Failed to read model: #{reason}"}
    end
  end

  defp decode(binary, path) do
    case :erlang.binary_to_term(binary) do
      %{format: @format} = model -> {:ok, model}
      other -> {:error, "#{path} is not a POS model of format #{@format} (got #{inspect(Map.get(other, :format))})"}
    end
  rescue
    e -> {:error, "Failed to deserialize model: #{Exception.message(e)}"}
  end

  @doc "Where the app's tagger is saved and loaded: `pos_model.term` under the configured models path."
  def model_path do
    case Application.get_env(:brain, :ml)[:models_path] do
      nil -> Brain.priv_path("ml_models/pos_model.term")
      models_path -> Path.join(models_path, "pos_model.term")
    end
  end
end
