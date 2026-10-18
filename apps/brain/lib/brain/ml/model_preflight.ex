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
      raise "Model preflight failed. #{length(failures)} model(s) missing or incompatible. Run `mix train --world <id>` or test `ModelFactory` setup."
    end

    :ok
  end

  @doc "Validate all models. Returns a keyword list of `{name, :ok | {:error, reason}}`."
  def validate_all do
    core =
      [
        {:embedder, validate_embedder()},
        {:pos_model, validate_pos_model()},
        {:sentiment_classifier, validate_simple_classifier_root("sentiment_classifier.term")},
        {:speech_act_classifier, validate_simple_classifier_root("speech_act_classifier.term")},
        {:framing_neutral_centroid, validate_framing_neutral_centroid()},
        {:entity_model, validate_entity_model()}
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

    language =
      Enum.flat_map(Brain.Language.supported(), fn code ->
        [
          {:"lexical_evidence_#{code}", validate_language_artifact(Brain.Analysis.LexicalEvidence, code)},
          {:"speech_act_table_#{code}", validate_language_artifact(Brain.Analysis.SpeechActTable, code)},
          {:"span_reliability_#{code}", validate_language_artifact(Brain.ML.SpanReliability, code)},
          {:"slot_type_preferences_#{code}", validate_language_artifact(Brain.Analysis.SlotTypePreferences, code)},
          {:"free_text_capture_#{code}", validate_language_artifact(Brain.Analysis.FreeTextCapture, code)},
          {:"intent_arbitration_#{code}", validate_language_artifact(Brain.Analysis.IntentArbitration, code)}
        ]
      end)

    core ++ micro ++ graph ++ language
  end

  # `read!/1` checks the file holds that language's struct; the entries checked
  # here are what the feature vector, the speech-act refinement, entity span
  # decisions, slot-type weighting and free-text slot capture read.
  defp validate_language_artifact(module, code) do
    case module.read!(code) do
      %Brain.Analysis.LexicalEvidence{intents: [_ | _], scores: scores} when map_size(scores) > 0 -> :ok
      %Brain.Analysis.SpeechActTable{entries: entries} when map_size(entries) > 0 -> :ok
      %Brain.ML.SpanReliability{pairs: pairs} when map_size(pairs) > 0 -> :ok
      %Brain.Analysis.SlotTypePreferences{slot_counts: %{global: global}} when map_size(global) > 0 -> :ok
      %Brain.Analysis.FreeTextCapture{parameters: parameters} when map_size(parameters) > 0 -> :ok
      %Brain.Analysis.IntentArbitration{overall: %{n: n}} when n > 0 -> :ok
      other -> {:error, "#{inspect(module)} #{code}: empty #{inspect(other.__struct__)}"}
    end
  rescue
    e in RuntimeError -> {:error, Exception.message(e)}
  end

  # The trained model every world is seeded from. Each world's own copy is
  # checked when the world first extracts (`Brain.ML.EntityModel.loaded!/1`).
  defp validate_entity_model do
    model = Brain.ML.EntityModel.read!(Brain.ML.EntityModel.trained_path(), :trained)

    if map_size(model.tag_vocabulary) > 1 do
      :ok
    else
      {:error, "entity_model.term: tags only #{inspect(Map.keys(model.tag_vocabulary))}"}
    end
  rescue
    e in RuntimeError -> {:error, Exception.message(e)}
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
            validate_micro_provenance(data, name, path)
          end

        Map.has_key?(data, :label_centroids) ->
          # Feature-vector centroid model
          if map_size(data.label_centroids) == 0 do
            {:error, "micro/#{name}.term: empty label_centroids"}
          else
            validate_micro_provenance(data, name, path)
          end

        true ->
          {:error, "micro/#{name}.term: unrecognized model shape"}
      end
    end)
  end

  # Both halves of the provenance check. This is the only place the feature-schema
  # half can run: it needs Brain.Analysis.TypeHierarchy, which starts after
  # Brain.ML.MicroClassifiers, so that GenServer can check the training data at
  # load but not the schema.
  defp validate_micro_provenance(model, name, path) do
    Brain.ML.MicroProvenance.check_current!(model, name, path)
    :ok
  rescue
    e in RuntimeError -> {:error, Exception.message(e)}
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

  # The trained embeddings every world is seeded from. Each world's own copy is
  # checked when the world first uses it (`Brain.ML.Poincare.Embeddings.loaded!/1`).
  defp validate_poincare do
    alias Brain.ML.Poincare.Embeddings

    Embeddings.read!(Embeddings.trained_path(), :trained)
    :ok
  rescue
    e in RuntimeError -> {:error, Exception.message(e)}
  end

  # The trained scorer every world is seeded from. Each world's own copy is
  # checked when the world first scores (`Brain.ML.KnowledgeGraph.TripleScorer.loaded!/1`).
  defp validate_triple_scorer do
    alias Brain.ML.KnowledgeGraph.TripleScorer

    TripleScorer.read!(TripleScorer.trained_path(), :trained)
    :ok
  rescue
    e in RuntimeError -> {:error, Exception.message(e)}
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
    ║  Prod: `mix train --world <id>`. Test: `ModelFactory`.     ║
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
