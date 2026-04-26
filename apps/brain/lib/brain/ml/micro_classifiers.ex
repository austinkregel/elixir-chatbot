defmodule Brain.ML.MicroClassifiers do
  @moduledoc """
  GenServer that hosts multiple small TF-IDF classifiers for lightweight
  NLP classification tasks.

  Each classifier is trained from a JSON data file in `data/classifiers/`
  and loaded at startup from `priv/ml_models/micro/`. If a trained model
  file doesn't exist, the classifier logs an error and remains unavailable.

  ## Available Classifiers

  - `:personal_question` - Detects personal questions about the bot
  - `:clarification_response` - Detects clarification/disambiguation responses
  - `:modal_directive` - Detects "can you / would you" directive patterns
  - `:fallback_response` - Detects generic fallback/error responses
  - `:goal_type` - Classifies research goal type (reasoning/sentiment/factual/general)
  - `:entity_type` - Infers entity type from entity name + category
  - `:user_fact_type` - Classifies whether entity/fact is user-specific or general knowledge
  - `:directed_at_bot` - Detects whether text is addressed to the bot
  - `:intent_full` - Fine-grained intent classification (e.g. `weather.query`, `code.explain`)
  - `:intent_domain` - Consolidated topical domain for `ChunkProfile.domain`
  - `:tense_class` - Temporal tense (`past`, `present`, `future`, `atemporal`)
  - `:aspect_class` - Grammatical aspect (`simple`, `progressive`, `perfect`, `perfect_progressive`)
  - `:urgency` - Urgency (`low`, `normal`, `high`, `critical`)
  - `:certainty_level` - Epistemic stance (`committed`, `tentative`, `hedged`, `speculative`)
  - `:coarse_semantic_class` - OOV Tier-1 coarse class (`person`, `place`, `thing`, …)

  Axis training JSON is produced by `mix gen_micro_data`; models are trained with
  `mix train_micro` (also stage 9 of `mix train`).

  ## Usage

      MicroClassifiers.classify(:personal_question, "what is your name")
      # => {:ok, "personal", 0.87}

      MicroClassifiers.classify(:fallback_response, "The weather is sunny.")
      # => {:ok, "not_fallback", 0.92}

      # Incremental learning from usage
      MicroClassifiers.incremental_update(:user_fact_type, [
        {"my favorite color blue", "user_specific"},
        {"photosynthesis plants energy", "general_knowledge"}
      ])
      # => {:ok, 1}
  """

  use GenServer
  require Logger

  alias Brain.Lattice
  alias Brain.ML.FeatureVectorClassifier
  alias Brain.ML.SimpleClassifier

  @classifier_names [
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

  # --- Client API ---

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Classify text using the named micro-classifier."
  @spec classify(atom(), String.t()) :: {:ok, String.t(), float()} | {:error, :not_loaded}
  def classify(name, text) do
    if ready?() do
      GenServer.call(__MODULE__, {:classify, name, text}, 5_000)
    else
      {:error, :not_loaded}
    end
  end

  @doc """
  Classify a dense feature vector using the named micro-classifier.

  Only valid for classifiers whose loaded model is of `kind: :feature_vector`
  (see `Brain.ML.FeatureVectorClassifier`). Returns `{:error, :wrong_kind}`
  if the named classifier is text-based, and `{:error, :not_loaded}` if the
  GenServer or the named model isn't available.
  """
  @spec classify_vector(atom(), list(float())) ::
          {:ok, String.t(), float()}
          | {:error, :not_loaded | :wrong_kind | :classification_failed}
  def classify_vector(name, feature_vector) when is_atom(name) and is_list(feature_vector) do
    if ready?() do
      GenServer.call(__MODULE__, {:classify_vector, name, feature_vector}, 5_000)
    else
      {:error, :not_loaded}
    end
  end

  @doc """
  Classify text and return a `Brain.Lattice` with full `top_k` / margin semantics
  (TF-IDF path via `SimpleClassifier.classify_with_details/3`).
  """
  @spec classify_detailed(atom(), String.t()) ::
          {:ok, Lattice.t()}
          | {:error, :not_loaded | :wrong_kind | :classification_failed}
  def classify_detailed(name, text) when is_atom(name) and is_binary(text) do
    if ready?() do
      GenServer.call(__MODULE__, {:classify_detailed, name, text}, 5_000)
    else
      {:error, :not_loaded}
    end
  end

  @doc """
  Classify a feature vector and return a `Brain.Lattice` with full `top_k`
  from `FeatureVectorClassifier` details.
  """
  @spec classify_vector_detailed(atom(), list(float())) ::
          {:ok, Lattice.t()}
          | {:error, :not_loaded | :wrong_kind | :classification_failed}
  def classify_vector_detailed(name, feature_vector)
      when is_atom(name) and is_list(feature_vector) do
    if ready?() do
      GenServer.call(__MODULE__, {:classify_vector_detailed, name, feature_vector}, 5_000)
    else
      {:error, :not_loaded}
    end
  end

  @doc """
  Return the declared input dimensionality of the named feature-vector
  classifier.

  Returns `{:ok, integer}` for loaded feature-vector models, or an error
  tuple otherwise. Used to assert at boot / test time that the feature
  extractor and the trained model agree on vector shape.
  """
  @spec input_dim(atom()) ::
          {:ok, non_neg_integer()} | {:error, :not_loaded | :not_trained | :wrong_kind}
  def input_dim(name) when is_atom(name) do
    if ready?() do
      GenServer.call(__MODULE__, {:input_dim, name}, 1_000)
    else
      {:error, :not_loaded}
    end
  end

  @doc """
  Return the kind of the named classifier's loaded model.

  Returns `{:ok, :feature_vector}` or `{:ok, :text}` for loaded models,
  or `{:error, :not_loaded | :not_trained}` otherwise.
  """
  @spec kind(atom()) ::
          {:ok, :feature_vector | :text} | {:error, :not_loaded | :not_trained}
  def kind(name) when is_atom(name) do
    if ready?() do
      GenServer.call(__MODULE__, {:kind, name}, 1_000)
    else
      {:error, :not_loaded}
    end
  end

  @doc "Check if the MicroClassifiers server is ready."
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc """
  Block until MicroClassifiers is ready, polling every 250ms.

  Use `:infinity` for batch/CLI contexts (evaluation tasks, mix commands)
  where there is no caller to time out for. Use a finite timeout at
  runtime where degradation matters.

  Returns `:ok` or raises after `timeout_ms` milliseconds.
  """
  @spec await_ready(non_neg_integer() | :infinity) :: :ok
  def await_ready(timeout_ms \\ :infinity) do
    deadline =
      case timeout_ms do
        :infinity -> :infinity
        ms when is_integer(ms) and ms > 0 -> System.monotonic_time(:millisecond) + ms
      end

    do_await_ready(deadline)
  end

  defp do_await_ready(deadline) do
    if ready?() do
      :ok
    else
      if deadline != :infinity and System.monotonic_time(:millisecond) >= deadline do
        raise "MicroClassifiers failed to become ready within timeout. " <>
              "Ensure models are trained: mix train_micro"
      end

      Process.sleep(250)
      do_await_ready(deadline)
    end
  end

  @doc "Hot-reload all models from disk or retrain from data files."
  def reload do
    GenServer.call(__MODULE__, :reload, 30_000)
  end

  @doc "Incrementally update a named micro-classifier with new examples.\n\nExamples should be a list of `{text, label}` tuples. Uses\n`SimpleClassifier.update_model/3` to blend new examples into\nthe existing model without a full retrain. After 200 incremental\nupdates, the model is persisted to disk.\n\nReturns `{:ok, incremental_count}` or `{:error, reason}`.\n"
  @spec incremental_update(atom(), [{String.t(), String.t()}]) ::
          {:ok, non_neg_integer()} | {:error, atom()}
  def incremental_update(name, examples) when is_atom(name) and is_list(examples) do
    if ready?() do
      GenServer.call(__MODULE__, {:incremental_update, name, examples}, 10_000)
    else
      {:error, :not_loaded}
    end
  end

  @doc "Get status of all loaded classifiers."
  def status do
    if ready?() do
      GenServer.call(__MODULE__, :status, 5_000)
    else
      %{ready: false, classifiers: %{}}
    end
  end

  # --- Server Callbacks ---

  @impl true
  def init(_opts) do
    models = load_all_models()

    {:ok, %{models: models}}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, all_models_loaded?(state.models), state}
  end

  @impl true
  def handle_call({:classify, name, text}, _from, state) do
    case Map.get(state.models, name) do
      nil ->
        {:reply, {:error, :not_loaded}, state}

      %{kind: :feature_vector} ->
        {:reply, {:error, :wrong_kind}, state}

      model ->
        case SimpleClassifier.classify(text, model) do
          {:ok, label, score, _details} ->
            {:reply, {:ok, label, score}, state}

          _ ->
            {:reply, {:error, :classification_failed}, state}
        end
    end
  end

  @impl true
  def handle_call({:classify_vector, name, feature_vector}, _from, state) do
    case Map.get(state.models, name) do
      nil ->
        {:reply, {:error, :not_loaded}, state}

      %{kind: :feature_vector} = model ->
        case FeatureVectorClassifier.classify(feature_vector, model) do
          {:ok, label, score, _details} ->
            {:reply, {:ok, label, score}, state}

          _ ->
            {:reply, {:error, :classification_failed}, state}
        end

      _other ->
        {:reply, {:error, :wrong_kind}, state}
    end
  end

  @impl true
  def handle_call({:classify_detailed, name, text}, _from, state) do
    case Map.get(state.models, name) do
      nil ->
        {:reply, {:error, :not_loaded}, state}

      %{kind: :feature_vector} ->
        {:reply, {:error, :wrong_kind}, state}

      model ->
        case SimpleClassifier.classify_with_details(text, model, top_k: 5) do
          {:ok, label, score, details} ->
            lattice =
              Lattice.from_classifier({:ok, label, score, details},
                stage: name,
                source: :tf_idf
              )

            {:reply, {:ok, lattice}, state}

          _ ->
            {:reply, {:error, :classification_failed}, state}
        end
    end
  end

  @impl true
  def handle_call({:classify_vector_detailed, name, feature_vector}, _from, state) do
    case Map.get(state.models, name) do
      nil ->
        {:reply, {:error, :not_loaded}, state}

      %{kind: :feature_vector} = model ->
        case FeatureVectorClassifier.classify(feature_vector, model) do
          {:ok, label, score, details} ->
            lattice =
              Lattice.from_classifier({:ok, label, score, details},
                stage: name,
                source: :feature_vector
              )

            {:reply, {:ok, lattice}, state}

          _ ->
            {:reply, {:error, :classification_failed}, state}
        end

      _other ->
        {:reply, {:error, :wrong_kind}, state}
    end
  end

  @impl true
  def handle_call({:input_dim, name}, _from, state) do
    case Map.get(state.models, name) do
      nil ->
        {:reply, {:error, :not_trained}, state}

      %{kind: :feature_vector} = model ->
        {:reply, {:ok, FeatureVectorClassifier.input_dim(model)}, state}

      _other ->
        {:reply, {:error, :wrong_kind}, state}
    end
  end

  @impl true
  def handle_call({:kind, name}, _from, state) do
    case Map.get(state.models, name) do
      nil ->
        {:reply, {:error, :not_trained}, state}

      %{kind: :feature_vector} ->
        {:reply, {:ok, :feature_vector}, state}

      _other ->
        {:reply, {:ok, :text}, state}
    end
  end

  @impl true
  def handle_call({:incremental_update, name, examples}, _from, state) do
    case Map.get(state.models, name) do
      nil ->
        {:reply, {:error, :not_loaded}, state}

      %{kind: :feature_vector} ->
        # Feature-vector classifiers are retrained via `mix train_micro`;
        # they don't currently support incremental updates.
        {:reply, {:error, :incremental_not_supported}, state}

      model ->
        incremental_count = Map.get(model, :incremental_update_count, 0)
        {updated_model, new_count} = SimpleClassifier.update_model(model, examples, incremental_count)
        updated_model = Map.put(updated_model, :incremental_update_count, new_count)

        if rem(new_count, 200) == 0 and new_count > 0 do
          persist_model(name, updated_model)
        end

        Logger.debug("MicroClassifiers: incremental update for #{name}",
          examples: length(examples),
          incremental_count: new_count
        )

        {:reply, {:ok, new_count}, %{state | models: Map.put(state.models, name, updated_model)}}
    end
  end

  @impl true
  def handle_call(:reload, _from, _state) do
    models = load_all_models()
    {:reply, :ok, %{models: models}}
  end

  @impl true
  def handle_call({:load_trained_models, models_map}, _from, state) when is_map(models_map) do
    merged = Map.merge(state.models, models_map)
    {:reply, :ok, %{state | models: merged}}
  end

  @impl true
  def handle_call(:status, _from, state) do
    ready = all_models_loaded?(state.models)

    status =
      Enum.into(@classifier_names, %{}, fn name ->
        case Map.get(state.models, name) do
          nil ->
            {name, %{loaded: false, incremental_updates: 0}}

          model ->
            count = Map.get(model, :incremental_update_count, 0)
            {name, %{loaded: true, incremental_updates: count}}
        end
      end)

    {:reply, %{ready: ready, classifiers: status}, state}
  end

  @impl true
  def handle_info(_msg, state) do
    {:noreply, state}
  end

  # --- Private ---

  defp all_models_loaded?(models) do
    Enum.all?(@classifier_names, &Map.has_key?(models, &1))
  end

  defp load_all_models do
    Enum.reduce(@classifier_names, %{}, fn name, acc ->
      case load_model(name) do
        {:ok, model} ->
          Map.put(acc, name, model)

        {:error, reason} ->
          Logger.warning("MicroClassifiers: failed to load #{name}: #{inspect(reason)}")
          acc
      end
    end)
  end

  defp load_model(name) do
    model_path = model_file_path(name)
    Brain.ML.ModelStore.ensure_local("micro/#{name}.term", model_path)

    case File.read(model_path) do
      {:ok, binary} ->
        try do
          model = :erlang.binary_to_term(binary)
          {:ok, model}
        rescue
          _ ->
            Logger.error("MicroClassifiers: corrupt model file for #{name}. Run `mix train_micro` to retrain.")
            {:error, :corrupted_model}
        end

      {:error, _} ->
        Logger.error("MicroClassifiers: no model file for #{name} at #{model_file_path(name)}. Run `mix train_micro` to train.")
        {:error, :no_model_file}
    end
  end

  defp persist_model(name, model) do
    path = model_file_path(name)
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, :erlang.term_to_binary(model))
    Logger.info("MicroClassifiers: persisted #{name} to #{path}")
  rescue
    e -> Logger.warning("MicroClassifiers: failed to persist #{name}: #{inspect(e)}")
  end

  defp model_file_path(name) do
    base =
      case Application.get_env(:brain, :ml, [])[:models_path] do
        nil -> Brain.priv_path("ml_models")
        path -> path
      end

    Path.join([base, "micro", "#{name}.term"])
  end

end
