defmodule Brain.ML.ModelPreflight do
  @moduledoc """
  Validates stored ML models match the current code's expected dimensions.

  Reads .term files directly (no GenServer calls) and compares stored metadata
  against what the code expects. Missing files are fine (graceful degradation
  handles absent models), but dimension/config mismatches are hard failures
  because they mean the model exists but is incompatible.

  ## Usage

  Call `validate_all!/0` at test suite startup to halt early on mismatches:

      Brain.ML.ModelPreflight.validate_all!()

  Call `validate_all/0` for a list of results without raising.
  """

  require Logger

  @doc "Validate all models and raise if any are incompatible."
  def validate_all! do
    results = validate_all()
    failures = Enum.filter(results, &match?({_, {:error, _}}, &1))

    if failures != [] do
      msg = format_failures(failures)
      IO.puts(:stderr, "\n" <> msg)
      raise "Model preflight failed. #{length(failures)} model(s) incompatible. Run `mix train` to retrain."
    end

    :ok
  end

  @doc "Validate all models. Returns a keyword list of `{name, :ok | {:error, reason}}`."
  def validate_all do
    [
      {:intent_arbitrator, validate_intent_arbitrator()},
      {:unified_model, validate_unified_model()},
      {:response_scorer, validate_response_scorer()},
      {:gcn, validate_gcn()},
      {:poincare, validate_poincare()},
      {:triple_scorer, validate_triple_scorer()}
    ]
  end

  # -- IntentArbitrator --

  defp validate_intent_arbitrator do
    path = model_file("intent_arbitrator.term")

    with_term_file(path, fn data ->
      stored_fc = Map.get(data, :feature_count, 0)
      expected = Brain.ML.IntentArbitrator.feature_count()

      if stored_fc != expected do
        {:error,
         "IntentArbitrator feature count mismatch: stored=#{stored_fc}, expected=#{expected}"}
      else
        :ok
      end
    end)
  end

  # -- UnifiedModel --

  defp validate_unified_model do
    path = model_file("lstm/unified_model.term")

    with_term_file(path, fn data ->
      cond do
        not is_map_key(data, :config) ->
          {:error, "UnifiedModel: missing :config key"}

        not is_map_key(data, :vocabularies) ->
          {:error, "UnifiedModel: missing :vocabularies key"}

        not is_map_key(data, :params) ->
          {:error, "UnifiedModel: missing :params key"}

        not is_map(data.vocabularies) or not is_map_key(data.vocabularies, :token_vocab) ->
          {:error, "UnifiedModel: missing :token_vocab in vocabularies"}

        map_size(data.vocabularies.token_vocab) == 0 ->
          {:error, "UnifiedModel: empty token vocabulary"}

        true ->
          :ok
      end
    end)
  end

  # -- LSTMResponse (Response Scorer) --

  defp validate_response_scorer do
    path = model_file("lstm/response_scorer.term")

    with_term_file(path, fn data ->
      cond do
        not is_map_key(data, :config) ->
          {:error, "ResponseScorer: missing :config key"}

        not is_map_key(data, :vocabularies) ->
          {:error, "ResponseScorer: missing :vocabularies key"}

        not is_map_key(data, :scorer_params) ->
          {:error, "ResponseScorer: missing :scorer_params key"}

        not is_map(data.vocabularies) or not is_map_key(data.vocabularies, :token_vocab) ->
          {:error, "ResponseScorer: missing :token_vocab in vocabularies"}

        map_size(data.vocabularies.token_vocab) == 0 ->
          {:error, "ResponseScorer: empty token vocabulary"}

        true ->
          :ok
      end
    end)
  end

  # -- GCN Model --

  defp validate_gcn do
    path = model_file("default/gcn/model.term")

    with_term_file(path, fn data ->
      config = Map.get(data, :config, %{})
      features = Map.get(data, :features)

      cond do
        not is_map_key(config, :num_features) ->
          {:error, "GCN: missing :num_features in config"}

        not is_map_key(config, :num_classes) ->
          {:error, "GCN: missing :num_classes in config"}

        is_struct(features, Nx.Tensor) and
            Nx.axis_size(features, 1) != config.num_features ->
          {:error,
           "GCN: features shape mismatch: tensor has #{Nx.axis_size(features, 1)} features, " <>
             "config says #{config.num_features}"}

        true ->
          :ok
      end
    end)
  end

  # -- Poincare Embeddings --

  defp validate_poincare do
    path = model_file("default/poincare/embeddings.term")

    with_term_file(path, fn data ->
      dim = Map.get(data, :dim)
      embeddings = Map.get(data, :embeddings)

      cond do
        is_nil(dim) ->
          {:error, "Poincare: missing :dim field"}

        is_struct(embeddings, Nx.Tensor) and Nx.axis_size(embeddings, 1) != dim ->
          {:error,
           "Poincare: dim mismatch: embeddings have #{Nx.axis_size(embeddings, 1)} dims, " <>
             "stored dim=#{dim}"}

        true ->
          :ok
      end
    end)
  end

  # -- TripleScorer --

  defp validate_triple_scorer do
    path = model_file("default/kg_lstm/triple_scorer.term")

    with_term_file(path, fn data ->
      config = Map.get(data, :config, %{})
      vocab = Map.get(data, :vocab, %{})

      cond do
        not is_map_key(config, :vocab_size) ->
          {:error, "TripleScorer: missing :vocab_size in config"}

        map_size(vocab) == 0 ->
          {:error, "TripleScorer: empty vocab"}

        config.vocab_size != map_size(vocab) ->
          {:error,
           "TripleScorer: vocab size mismatch: config says #{config.vocab_size}, " <>
             "vocab has #{map_size(vocab)} entries"}

        true ->
          :ok
      end
    end)
  end

  # -- Helpers --

  defp models_path do
    Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
  end

  defp model_file(relative) do
    Path.join(models_path(), relative)
  end

  defp with_term_file(path, validate_fn) do
    case File.read(path) do
      {:ok, binary} ->
        try do
          data = :erlang.binary_to_term(binary)
          validate_fn.(data)
        rescue
          e -> {:error, "Failed to deserialize #{Path.basename(path)}: #{Exception.message(e)}"}
        end

      {:error, :enoent} ->
        :ok

      {:error, reason} ->
        {:error, "Failed to read #{Path.basename(path)}: #{inspect(reason)}"}
    end
  end

  defp format_failures(failures) do
    header = """
    ╔══════════════════════════════════════════════════════════════╗
    ║  MODEL PREFLIGHT CHECK FAILED                              ║
    ║  Stored models are incompatible with current code.         ║
    ║  Run `mix train` to retrain all models.                    ║
    ╚══════════════════════════════════════════════════════════════╝
    """

    details =
      failures
      |> Enum.map(fn {name, {:error, reason}} ->
        "  ✗ #{name}: #{reason}"
      end)
      |> Enum.join("\n")

    header <> "\n" <> details <> "\n"
  end
end
