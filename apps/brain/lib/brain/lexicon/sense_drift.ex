defmodule Brain.Lexicon.SenseDrift do
  @moduledoc """
  Detects semantic drift between a word's canonical (lexicon-derived)
  feature vector and its observed (conversation-derived) feature vector.

  Inspired by the SenSE toolkit's approach to lexical semantic change:
  represent words as feature vectors and measure drift via cosine distance.
  High drift + neighbor change indicates a sense shift.
  """

  alias Brain.Lexicon.UserDefined

  @drift_threshold 0.3
  @high_drift_threshold 0.6

  @doc """
  Measures sense drift for a word between its canonical and observed vectors.

  Returns a map with:
  - `:drift` - cosine distance (0.0 = identical, 1.0 = orthogonal)
  - `:level` - `:none`, `:low`, `:moderate`, `:high`
  - `:should_clarify` - whether the drift warrants a clarification interrupt
  """
  def measure(word, canonical_vector, observed_vector)
      when is_binary(word) and is_list(canonical_vector) and is_list(observed_vector) do
    drift = cosine_distance(canonical_vector, observed_vector)

    level =
      cond do
        drift < 0.1 -> :none
        drift < @drift_threshold -> :low
        drift < @high_drift_threshold -> :moderate
        true -> :high
      end

    %{
      drift: drift,
      level: level,
      should_clarify: level == :high
    }
  end

  def measure(_word, _canonical, _observed), do: %{drift: 0.0, level: :none, should_clarify: false}

  @doc """
  Checks if an OOV word has accumulated enough observations for Tier 2 refinement.
  """
  def tier2_eligible?(word, min_observations \\ 3) when is_binary(word) do
    case UserDefined.get(String.downcase(word)) do
      nil ->
        false

      %{senses: senses} ->
        Enum.any?(senses, fn sense ->
          (sense[:frequency] || 0) >= min_observations
        end)
    end
  end

  @doc """
  Detects whether a word's usage suggests a new sense has emerged.

  Compares the new context centroid against all existing senses.
  If no sense is similar enough, returns `{:new_sense, drift_score}`.
  If a sense matches, returns `{:existing_sense, sense_index, similarity}`.
  """
  def detect_sense_shift(word, context_centroid, opts \\ [])
      when is_binary(word) and is_list(context_centroid) do
    threshold = Keyword.get(opts, :similarity_threshold, 0.7)

    case UserDefined.get(String.downcase(word)) do
      nil ->
        {:unknown_word, 0.0}

      %{senses: senses} ->
        matches =
          senses
          |> Enum.with_index()
          |> Enum.map(fn {sense, idx} ->
            case sense[:centroid] do
              c when is_list(c) ->
                sim = cosine_similarity(c, context_centroid)
                {idx, sim}

              _ ->
                {idx, 0.0}
            end
          end)
          |> Enum.sort_by(fn {_idx, sim} -> -sim end)

        case matches do
          [] ->
            {:new_sense, 1.0}

          [{best_idx, best_sim} | _] ->
            if best_sim >= threshold do
              {:existing_sense, best_idx, best_sim}
            else
              {:new_sense, 1.0 - best_sim}
            end
        end
    end
  end

  @doc """
  Returns the confidence reduction factor based on drift level.

  High drift means we're less confident about the word's features
  in the chunk vector, so downstream confidence should be reduced.
  """
  def confidence_factor(%{level: :none}), do: 1.0
  def confidence_factor(%{level: :low}), do: 0.95
  def confidence_factor(%{level: :moderate}), do: 0.8
  def confidence_factor(%{level: :high}), do: 0.6
  def confidence_factor(_), do: 1.0

  # -- Private ----------------------------------------------------------------

  defp cosine_distance(a, b) do
    1.0 - cosine_similarity(a, b)
  end

  defp cosine_similarity(a, b) when is_list(a) and is_list(b) do
    FourthWall.Math.cosine_similarity(a, b)
  rescue
    _ ->
      dot = Enum.zip(a, b) |> Enum.reduce(0.0, fn {x, y}, acc -> acc + x * y end)
      mag_a = :math.sqrt(Enum.reduce(a, 0.0, fn x, acc -> acc + x * x end))
      mag_b = :math.sqrt(Enum.reduce(b, 0.0, fn x, acc -> acc + x * x end))

      if mag_a == 0.0 or mag_b == 0.0, do: 0.0, else: dot / (mag_a * mag_b)
  end
end
