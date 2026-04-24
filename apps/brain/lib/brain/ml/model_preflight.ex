defmodule Brain.ML.ModelPreflight do
  @moduledoc """
  Validates stored ML models: every required file must exist, deserialize, and
  satisfy basic structural checks.

  Missing files are **failures** (no silent skips). Call `validate_all!/0` at
  test suite startup after models have been trained to disk.
  """

  require Logger

  # Keep in sync with `Brain.ML.MicroClassifiers` `@classifier_names`.
  @micro_classifier_names [
    :personal_question,
    :clarification_response,
    :modal_directive,
    :fallback_response,
    :goal_type,
    :entity_type,
    :user_fact_type,
    :directed_at_bot,
    :event_argument_role,
    :intent_full,
    :intent_domain,
    :tense_class,
    :aspect_class,
    :urgency,
    :certainty_level,
    :coarse_semantic_class,
    :framing_class
  ]

  @doc "Validate all models and raise if any are incompatible or missing."
  def validate_all! do
    results = validate_all()
    failures = Enum.filter(results, &match?({_, {:error, _}}, &1))

    if failures != [] do
      msg = format_failures(failures)
      IO.puts(:stderr, "\n" <> msg)
      raise "Model preflight failed. #{length(failures)} model(s) missing or incompatible. Run `mix train` or test `ModelFactory` setup."
    end

    :ok
  end

  @doc "Validate all models. Returns a keyword list of `{name, :ok | {:error, reason}}`."
  def validate_all do
    core =
      [
        {:embedder, validate_embedder()},
        {:gazetteer, validate_gazetteer()},
        {:pos_model, validate_pos_model()},
        {:sentiment_classifier, validate_simple_classifier_root("sentiment_classifier.term")},
        {:speech_act_classifier, validate_simple_classifier_root("speech_act_classifier.term")},
        {:framing_neutral_centroid, validate_framing_neutral_centroid()}
      ]

    micro =
      Enum.map(@micro_classifier_names, fn name ->
        {:"micro_#{name}", validate_micro_classifier(name)}
      end)

    graph =
      [
        {:poincare, validate_poincare()},
        {:triple_scorer, validate_triple_scorer()}
      ]

    core ++ micro ++ graph
  end

  defp validate_embedder do
    path = model_file("embedder.term")

    with_term_file(path, fn data ->
      cond do
        not is_map(data) ->
          {:error, "Embedder: expected a map"}

        not is_map_key(data, :vocabulary) or not is_map_key(data, :idf_weights) ->
          {:error, "Embedder: missing :vocabulary or :idf_weights"}

        map_size(data.vocabulary) == 0 ->
          {:error, "Embedder: empty vocabulary"}

        true ->
          :ok
      end
    end)
  end

  defp validate_gazetteer do
    path = model_file("gazetteer.term")

    with_term_file(path, fn data ->
      if is_map(data) and map_size(data) > 0 do
        :ok
      else
        {:error, "Gazetteer: expected non-empty map"}
      end
    end)
  end

  defp validate_pos_model do
    path = model_file("pos_model.term")

    with_term_file(path, fn data ->
      if is_map(data) and map_size(data) > 0 do
        :ok
      else
        {:error, "POS model: expected non-empty map"}
      end
    end)
  end

  defp validate_simple_classifier_root(filename) do
    path = model_file(filename)

    with_term_file(path, fn data ->
      cond do
        not is_map(data) ->
          {:error, "#{filename}: expected map"}

        not Map.has_key?(data, :vocabulary) or not Map.has_key?(data, :label_centroids) ->
          {:error, "#{filename}: not a SimpleClassifier-style map"}

        map_size(data.vocabulary) == 0 ->
          {:error, "#{filename}: empty vocabulary"}

        true ->
          :ok
      end
    end)
  end

  defp validate_micro_classifier(name) do
    path = model_file("micro/#{name}.term")

    with_term_file(path, fn data ->
      cond do
        not is_map(data) ->
          {:error, "micro/#{name}.term: expected map"}

        Map.has_key?(data, :vocabulary) and Map.has_key?(data, :label_centroids) ->
          if map_size(data.vocabulary) == 0 do
            {:error, "micro/#{name}.term: empty vocabulary"}
          else
            :ok
          end

        Map.has_key?(data, :label_centroids) ->
          # Feature-vector centroid model
          if map_size(data.label_centroids) == 0 do
            {:error, "micro/#{name}.term: empty label_centroids"}
          else
            :ok
          end

        true ->
          {:error, "micro/#{name}.term: unrecognized model shape"}
      end
    end)
  end

  defp validate_framing_neutral_centroid do
    path = model_file("micro/framing_neutral_centroid.term")

    with_term_file(path, fn data ->
      if is_list(data) and data != [] and Enum.all?(data, &is_float/1) do
        :ok
      else
        {:error, "framing_neutral_centroid: expected non-empty list of floats"}
      end
    end)
  end

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
        {:error, "Missing model file: #{path}"}

      {:error, reason} ->
        {:error, "Failed to read #{Path.basename(path)}: #{inspect(reason)}"}
    end
  end

  defp format_failures(failures) do
    header = """
    ╔══════════════════════════════════════════════════════════════╗
    ║  MODEL PREFLIGHT CHECK FAILED                              ║
    ║  Stored models are missing or incompatible with code.      ║
    ║  Run `mix train` (prod) or `ModelFactory` (test).          ║
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
