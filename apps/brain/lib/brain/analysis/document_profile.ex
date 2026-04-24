defmodule Brain.Analysis.DocumentProfile do
  @moduledoc """
  Aggregates per-chunk feature vectors into a document-level profile.

  Given a list of `%ChunkProfile{}` structs (each carrying a `feature_vector`),
  produces mean, variance, and skew vectors in the **same dimensional space** as
  `Brain.Analysis.FeatureExtractor.ChunkFeatures.vector_dimension/0`. This means
  a centroid classifier trained on document-level mean vectors can compare
  directly to chunk-level evidence.

  ## Entity slices

  `entity_slices` averages only the feature vectors of chunks that mention a
  given entity, giving a per-entity "how does this document talk about X"
  fingerprint. Entities are harvested from `ChunkProfile` metadata or from a
  companion `ChunkAnalysis` entity list passed alongside.

  ## Rhetorical mode

  Derived from the distribution of `speech_act_category` across chunks:
  what share of chunks are assertions, questions, imperatives, or expressives.
  """

  alias Brain.Analysis.ChunkProfile

  @type entity_token :: String.t()

  @type t :: %__MODULE__{
          doc_id: String.t() | nil,
          chunk_count: non_neg_integer(),
          mean_vector: list(float()),
          variance_vector: list(float()),
          skew_vector: list(float()),
          entity_slices: %{entity_token() => list(float())},
          dominant_pos_distribution: list(float()),
          dominant_lexical_domains: list({atom(), float()}),
          rhetorical_mode: %{
            assertion_share: float(),
            question_share: float(),
            imperative_share: float(),
            expressive_share: float()
          }
        }

  defstruct doc_id: nil,
            chunk_count: 0,
            mean_vector: [],
            variance_vector: [],
            skew_vector: [],
            entity_slices: %{},
            dominant_pos_distribution: [],
            dominant_lexical_domains: [],
            rhetorical_mode: %{
              assertion_share: 0.0,
              question_share: 0.0,
              imperative_share: 0.0,
              expressive_share: 0.0
            }

  @max_entity_slices 50

  @doc """
  Aggregates a list of ChunkProfiles into a DocumentProfile.

  Options:
  - `:doc_id` — optional document identifier
  - `:entity_lists` — optional list of entity lists (one per chunk, aligned by
    index) where each entry is a list of entity maps with `:value` or `:name`
    keys. When absent, entity slicing is skipped.
  - `:token_counts` — optional list of integers (one per chunk) used to
    weight chunks by length when averaging. When absent, uniform weighting.
  """
  @spec aggregate([ChunkProfile.t()], keyword()) :: t()
  def aggregate(profiles, opts \\ [])

  def aggregate([], opts) do
    %__MODULE__{doc_id: Keyword.get(opts, :doc_id)}
  end

  def aggregate(profiles, opts) when is_list(profiles) do
    doc_id = Keyword.get(opts, :doc_id)
    entity_lists = Keyword.get(opts, :entity_lists)
    token_counts = Keyword.get(opts, :token_counts)

    vectors =
      profiles
      |> Enum.map(fn
        %ChunkProfile{feature_vector: v} when is_list(v) and v != [] -> v
        _ -> nil
      end)

    valid_indices =
      vectors
      |> Enum.with_index()
      |> Enum.reject(fn {v, _i} -> is_nil(v) end)

    valid_vectors = Enum.map(valid_indices, fn {v, _} -> v end)
    valid_profiles = Enum.map(valid_indices, fn {_, i} -> Enum.at(profiles, i) end)

    if valid_vectors == [] do
      %__MODULE__{doc_id: doc_id, chunk_count: length(profiles)}
    else
      weights = build_weights(valid_indices, token_counts)

      mean = weighted_mean(valid_vectors, weights)
      variance = weighted_variance(valid_vectors, mean, weights)
      skew = weighted_skew(valid_vectors, mean, variance, weights)

      entity_slices = build_entity_slices(valid_indices, entity_lists, weights)

      rhetorical = compute_rhetorical_mode(valid_profiles)

      %__MODULE__{
        doc_id: doc_id,
        chunk_count: length(profiles),
        mean_vector: mean,
        variance_vector: variance,
        skew_vector: skew,
        entity_slices: entity_slices,
        dominant_pos_distribution: [],
        dominant_lexical_domains: [],
        rhetorical_mode: rhetorical
      }
    end
  end

  @doc """
  Computes cosine similarity between two document profiles' mean vectors.
  """
  @spec similarity(t(), t()) :: float()
  def similarity(%__MODULE__{mean_vector: v1}, %__MODULE__{mean_vector: v2})
      when is_list(v1) and is_list(v2) and v1 != [] and v2 != [] do
    cosine_similarity(v1, v2)
  end

  def similarity(_, _), do: 0.0

  @doc """
  Computes cosine distance between a document profile's mean vector and a
  reference centroid (e.g., a neutral-framing centroid).
  """
  @spec deviation_from(t(), list(float())) :: float()
  def deviation_from(%__MODULE__{mean_vector: v}, centroid)
      when is_list(v) and is_list(centroid) and v != [] and centroid != [] do
    1.0 - cosine_similarity(v, centroid)
  end

  def deviation_from(_, _), do: 1.0

  @doc """
  Returns the dimension count of the mean vector (matches chunk-level
  `ChunkFeatures.vector_dimension/0` when populated).
  """
  @spec vector_dimension(t()) :: non_neg_integer()
  def vector_dimension(%__MODULE__{mean_vector: v}) when is_list(v), do: length(v)
  def vector_dimension(_), do: 0

  # -- Weighted statistics -----------------------------------------------

  defp build_weights(valid_indices, nil) do
    n = length(valid_indices)
    List.duplicate(1.0 / max(n, 1), n)
  end

  defp build_weights(valid_indices, token_counts) when is_list(token_counts) do
    raw =
      Enum.map(valid_indices, fn {_v, i} ->
        count = Enum.at(token_counts, i, 1)
        max(count, 1) * 1.0
      end)

    total = Enum.sum(raw)

    if total == 0.0 do
      n = length(valid_indices)
      List.duplicate(1.0 / max(n, 1), n)
    else
      Enum.map(raw, &(&1 / total))
    end
  end

  defp weighted_mean(vectors, weights) do
    dim = length(hd(vectors))
    zero = List.duplicate(0.0, dim)

    vectors
    |> Enum.zip(weights)
    |> Enum.reduce(zero, fn {vec, w}, acc ->
      Enum.zip_with(acc, vec, fn a, v -> a + v * w end)
    end)
  end

  defp weighted_variance(vectors, mean, weights) do
    dim = length(mean)
    zero = List.duplicate(0.0, dim)

    vectors
    |> Enum.zip(weights)
    |> Enum.reduce(zero, fn {vec, w}, acc ->
      Enum.zip_with(acc, Enum.zip(vec, mean), fn a, {v, m} ->
        diff = v - m
        a + w * diff * diff
      end)
    end)
  end

  defp weighted_skew(vectors, mean, variance, weights) do
    dim = length(mean)
    zero = List.duplicate(0.0, dim)

    std_devs = Enum.map(variance, fn v -> :math.sqrt(max(v, 1.0e-12)) end)

    vectors
    |> Enum.zip(weights)
    |> Enum.reduce(zero, fn {vec, w}, acc ->
      vec
      |> Enum.zip(mean)
      |> Enum.zip(std_devs)
      |> Enum.zip(acc)
      |> Enum.map(fn {{{v, m}, sd}, a} ->
        z = (v - m) / sd
        a + w * z * z * z
      end)
    end)
  end

  # -- Entity slicing ----------------------------------------------------

  defp build_entity_slices(_valid_indices, nil, _weights), do: %{}

  defp build_entity_slices(valid_indices, entity_lists, weights) when is_list(entity_lists) do
    entity_chunks =
      valid_indices
      |> Enum.zip(weights)
      |> Enum.flat_map(fn {{vec, original_idx}, w} ->
        entities = Enum.at(entity_lists, original_idx, [])

        entities
        |> Enum.flat_map(&extract_entity_tokens/1)
        |> Enum.uniq()
        |> Enum.map(fn token -> {token, vec, w} end)
      end)
      |> Enum.group_by(fn {token, _vec, _w} -> token end, fn {_token, vec, w} -> {vec, w} end)

    entity_chunks
    |> Enum.sort_by(fn {_token, entries} -> -length(entries) end)
    |> Enum.take(@max_entity_slices)
    |> Map.new(fn {token, entries} ->
      vecs = Enum.map(entries, fn {v, _w} -> v end)
      ws = Enum.map(entries, fn {_v, w} -> w end)
      total_w = Enum.sum(ws)
      normalized_ws = Enum.map(ws, &(&1 / max(total_w, 1.0e-12)))
      {token, weighted_mean(vecs, normalized_ws)}
    end)
  end

  defp extract_entity_tokens(entity) when is_map(entity) do
    value =
      Map.get(entity, :value) ||
        Map.get(entity, "value") ||
        Map.get(entity, :name) ||
        Map.get(entity, "name")

    if is_binary(value) and value != "" do
      [String.downcase(value)]
    else
      []
    end
  end

  defp extract_entity_tokens(_), do: []

  # -- Rhetorical mode ---------------------------------------------------

  defp compute_rhetorical_mode(profiles) do
    n = max(length(profiles), 1)

    counts =
      Enum.reduce(profiles, %{assertion: 0, question: 0, imperative: 0, expressive: 0}, fn p, acc ->
        case p do
          %ChunkProfile{modality: :interrogative} ->
            %{acc | question: acc.question + 1}

          %ChunkProfile{modality: :imperative} ->
            %{acc | imperative: acc.imperative + 1}

          %ChunkProfile{modality: :exclamatory} ->
            %{acc | expressive: acc.expressive + 1}

          %ChunkProfile{speech_act_category: :expressive} ->
            %{acc | expressive: acc.expressive + 1}

          _ ->
            %{acc | assertion: acc.assertion + 1}
        end
      end)

    %{
      assertion_share: counts.assertion / n,
      question_share: counts.question / n,
      imperative_share: counts.imperative / n,
      expressive_share: counts.expressive / n
    }
  end

  # -- Math helpers ------------------------------------------------------

  defp cosine_similarity(a, b) when length(a) == length(b) do
    {dot, mag_a_sq, mag_b_sq} =
      Enum.zip_reduce(a, b, {0.0, 0.0, 0.0}, fn x, y, {d, ma, mb} ->
        {d + x * y, ma + x * x, mb + y * y}
      end)

    mag_a = :math.sqrt(mag_a_sq)
    mag_b = :math.sqrt(mag_b_sq)

    if mag_a == 0.0 or mag_b == 0.0, do: 0.0, else: dot / (mag_a * mag_b)
  end

  defp cosine_similarity(_, _), do: 0.0
end
