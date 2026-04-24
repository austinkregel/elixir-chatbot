defmodule Brain.Analysis.FramingDetector do
  @moduledoc """
  Document-level framing detector built on the FeatureVector pipeline.

  Combines a trained `:framing_class` centroid classifier (loaded via
  `Brain.ML.MicroClassifiers`) with deviation scoring against a
  neutral-framing centroid, and per-source drift detection via an
  ETS-backed exponential moving average cache.

  ## Public API

  - `assess_document/1` — classify a list of `ChunkProfile`s into a framing
    assessment (frame label, confidence, deviation from neutral, evidence).
  - `assess_entity_framing/2` — same assessment scoped to chunks mentioning
    a particular entity.
  - `detect_drift/2` — compare a new `DocumentProfile` against the running
    per-source EMA and report dimensional drift.

  ## Integration seam

  The JTMS plan's `TruthAssessor.sincerity` axis will eventually call
  `assess_document/1` and fold the `deviation_from_neutral` score into its
  overall assessment. This module has no dependency on the epistemic layer.
  """

  use GenServer
  require Logger

  alias Brain.Analysis.{ChunkProfile, DocumentProfile}
  alias Brain.ML.MicroClassifiers

  @type framing_assessment :: %{
          primary_frame: atom(),
          confidence: float(),
          secondary_frames: [{atom(), float()}],
          deviation_from_neutral: float(),
          rhetorical_mode: atom(),
          evidence: %{
            sentiment_skew: float(),
            modality_skew: float(),
            causal_attribution: %{agent_bias: float(), patient_bias: float()},
            dominant_lexical_domains: [{atom(), float()}]
          }
        }

  @type drift_report :: %{
          source_id: String.t(),
          drifted: boolean(),
          similarity_to_ema: float(),
          top_changed_dimensions: [{non_neg_integer(), float()}],
          documents_seen: non_neg_integer()
        }

  @ema_alpha 0.1
  @drift_threshold 0.85
  @ets_table :framing_drift_cache

  # -- Client API --------------------------------------------------------

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Classifies a list of ChunkProfiles into a framing assessment.

  Aggregates profiles into a DocumentProfile, classifies the document-level
  mean vector via the `:framing_class` micro-classifier, and computes
  deviation from the neutral centroid.

  Options:
  - `:entity_lists` — list of entity lists per chunk (for entity slicing)
  - `:token_counts` — list of token counts per chunk (for weighted averaging)
  """
  @spec assess_document([ChunkProfile.t()], keyword()) ::
          {:ok, framing_assessment()} | {:error, term()}
  def assess_document(profiles, opts \\ []) when is_list(profiles) do
    doc_profile =
      DocumentProfile.aggregate(profiles,
        entity_lists: Keyword.get(opts, :entity_lists),
        token_counts: Keyword.get(opts, :token_counts)
      )

    assess_from_profile(doc_profile)
  end

  @doc """
  Assesses framing for a specific entity within a document.

  Uses the entity-sliced mean vector from the DocumentProfile. Falls back
  to the full document assessment if the entity is not found in slices.
  """
  @spec assess_entity_framing(DocumentProfile.t(), String.t()) ::
          {:ok, framing_assessment()} | {:error, term()}
  def assess_entity_framing(%DocumentProfile{} = doc_profile, entity) when is_binary(entity) do
    normalized = String.downcase(entity)

    case Map.get(doc_profile.entity_slices, normalized) do
      nil ->
        assess_from_profile(doc_profile)

      entity_vector ->
        entity_doc = %DocumentProfile{
          doc_profile
          | mean_vector: entity_vector,
            doc_id: "#{doc_profile.doc_id || "doc"}:entity:#{normalized}"
        }

        assess_from_profile(entity_doc)
    end
  end

  @doc """
  Compares a document profile against the per-source running EMA.

  Returns a drift report indicating whether the source's framing has
  shifted significantly. Updates the EMA cache as a side effect.

  Requires the GenServer to be running. Falls back to a no-drift baseline
  if the GenServer is unavailable.
  """
  @spec detect_drift(String.t(), DocumentProfile.t()) :: {:ok, drift_report()}
  def detect_drift(source_id, %DocumentProfile{} = doc_profile) when is_binary(source_id) do
    if ready?() do
      GenServer.call(__MODULE__, {:detect_drift, source_id, doc_profile}, 5_000)
    else
      {:ok, no_drift_report(source_id)}
    end
  catch
    :exit, _ -> {:ok, no_drift_report(source_id)}
  end

  @doc "Check if the FramingDetector GenServer is running."
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc """
  Gets the current EMA vector for a source, if one exists.
  """
  @spec get_source_ema(String.t()) :: {:ok, list(float()), non_neg_integer()} | :not_found
  def get_source_ema(source_id) when is_binary(source_id) do
    case :ets.lookup(@ets_table, source_id) do
      [{^source_id, ema, count}] -> {:ok, ema, count}
      [] -> :not_found
    end
  rescue
    ArgumentError -> :not_found
  end

  # -- GenServer callbacks -----------------------------------------------

  @impl true
  def init(_opts) do
    table = :ets.new(@ets_table, [:named_table, :set, :public, read_concurrency: true])
    neutral = load_neutral_centroid()

    {:ok, %{table: table, neutral_centroid: neutral}}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  @impl true
  def handle_call({:detect_drift, source_id, doc_profile}, _from, state) do
    report = compute_drift(source_id, doc_profile)
    {:reply, {:ok, report}, state}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  # -- Core assessment logic ---------------------------------------------

  defp assess_from_profile(%DocumentProfile{mean_vector: mean_vec} = doc_profile)
       when is_list(mean_vec) and mean_vec != [] do
    {primary_frame, confidence, secondary_frames} = classify_frame(mean_vec)
    deviation = compute_deviation(mean_vec)
    rhetorical = dominant_rhetorical_mode(doc_profile.rhetorical_mode)
    evidence = extract_evidence(mean_vec, doc_profile)

    {:ok,
     %{
       primary_frame: primary_frame,
       confidence: confidence,
       secondary_frames: secondary_frames,
       deviation_from_neutral: deviation,
       rhetorical_mode: rhetorical,
       evidence: evidence
     }}
  end

  defp assess_from_profile(_) do
    {:error, :empty_document_profile}
  end

  defp classify_frame(mean_vector) do
    case MicroClassifiers.classify_vector(:framing_class, mean_vector) do
      {:ok, label, confidence} ->
        frame = safe_to_atom(label)
        {frame, confidence, []}

      {:error, _reason} ->
        {:unknown, 0.0, []}
    end
  end

  defp compute_deviation(mean_vector) do
    case load_neutral_centroid() do
      centroid when is_list(centroid) and centroid != [] ->
        1.0 - cosine_similarity(mean_vector, centroid)

      _ ->
        0.5
    end
  end

  defp dominant_rhetorical_mode(mode) when is_map(mode) do
    mode
    |> Enum.max_by(fn {_k, v} -> v end, fn -> {:mixed, 0.0} end)
    |> elem(0)
    |> case do
      :assertion_share -> :assertion
      :question_share -> :question
      :imperative_share -> :imperative
      :expressive_share -> :expressive
      other -> other
    end
  end

  defp dominant_rhetorical_mode(_), do: :mixed

  # Feature vector group offsets (from ChunkFeatures):
  # Group 5 (modality/certainty) starts at dim 46 (12+16+10+8=46), 8 dims
  # Group 8 (sentiment) starts at dim 65, 5 dims
  # Group 12 (SRL frames) starts ~dim 106, 10 dims
  # Group 10 (lexical-semantic fingerprint) starts ~dim 82, variable dims
  #
  # These are approximate; the exact offsets depend on runtime Lexicon.domain_atoms().
  # We extract evidence from the raw vector positions defensively.

  defp extract_evidence(mean_vec, _doc_profile) do
    modality_dims = safe_slice(mean_vec, 46, 8)
    sentiment_dims = safe_slice(mean_vec, 65, 5)

    sentiment_skew =
      case sentiment_dims do
        [pos, neg | _] -> pos - neg
        _ -> 0.0
      end

    modality_skew =
      case modality_dims do
        dims when length(dims) >= 4 ->
          hedge = Enum.at(dims, 6, 0.0)
          certainty = Enum.at(dims, 7, 0.0)
          certainty - hedge

        _ ->
          0.0
      end

    %{
      sentiment_skew: sentiment_skew,
      modality_skew: modality_skew,
      causal_attribution: %{agent_bias: 0.0, patient_bias: 0.0},
      dominant_lexical_domains: []
    }
  end

  # -- Drift detection ---------------------------------------------------

  defp compute_drift(source_id, %DocumentProfile{mean_vector: vec})
       when is_list(vec) and vec != [] do
    case :ets.lookup(@ets_table, source_id) do
      [{^source_id, ema, count}] ->
        similarity = cosine_similarity(vec, ema)
        drifted = similarity < @drift_threshold

        top_changed =
          vec
          |> Enum.zip(ema)
          |> Enum.with_index()
          |> Enum.map(fn {{v, e}, idx} -> {idx, abs(v - e)} end)
          |> Enum.sort_by(fn {_idx, diff} -> -diff end)
          |> Enum.take(10)

        new_ema = update_ema(ema, vec)
        :ets.insert(@ets_table, {source_id, new_ema, count + 1})

        %{
          source_id: source_id,
          drifted: drifted,
          similarity_to_ema: similarity,
          top_changed_dimensions: top_changed,
          documents_seen: count + 1
        }

      [] ->
        :ets.insert(@ets_table, {source_id, vec, 1})

        %{
          source_id: source_id,
          drifted: false,
          similarity_to_ema: 1.0,
          top_changed_dimensions: [],
          documents_seen: 1
        }
    end
  end

  defp compute_drift(source_id, _), do: no_drift_report(source_id)

  defp update_ema(ema, vec) do
    Enum.zip_with(ema, vec, fn e, v -> e * (1.0 - @ema_alpha) + v * @ema_alpha end)
  end

  defp no_drift_report(source_id) do
    %{
      source_id: source_id,
      drifted: false,
      similarity_to_ema: 1.0,
      top_changed_dimensions: [],
      documents_seen: 0
    }
  end

  # -- Neutral centroid loading ------------------------------------------

  defp load_neutral_centroid do
    path = neutral_centroid_path()

    case File.read(path) do
      {:ok, bin} ->
        try do
          :erlang.binary_to_term(bin)
        rescue
          _ -> nil
        end

      _ ->
        nil
    end
  end

  defp neutral_centroid_path do
    base =
      case Application.get_env(:brain, :ml, [])[:models_path] do
        nil -> Brain.priv_path("ml_models")
        path -> path
      end

    Path.join([base, "micro", "framing_neutral_centroid.term"])
  end

  # -- Math helpers ------------------------------------------------------

  defp cosine_similarity(a, b) when is_list(a) and is_list(b) and length(a) == length(b) do
    {dot, mag_a_sq, mag_b_sq} =
      Enum.zip_reduce(a, b, {0.0, 0.0, 0.0}, fn x, y, {d, ma, mb} ->
        {d + x * y, ma + x * x, mb + y * y}
      end)

    mag_a = :math.sqrt(mag_a_sq)
    mag_b = :math.sqrt(mag_b_sq)

    if mag_a == 0.0 or mag_b == 0.0, do: 0.0, else: dot / (mag_a * mag_b)
  end

  defp cosine_similarity(_, _), do: 0.0

  defp safe_slice(list, start, count) when is_list(list) do
    Enum.slice(list, start, count)
  end

  defp safe_slice(_, _, _), do: []

  defp safe_to_atom(label) when is_binary(label) do
    String.to_existing_atom(label)
  rescue
    ArgumentError -> String.to_atom(label)
  end
end
