defmodule Brain.Lattice.FragmentVectorizer do
  @moduledoc """
  Computes dense feature vectors for lattice phrase fragments via the analysis pipeline.

  Used at inventory build time (`mix gen_lattice_data`) and when feeding runtime
  fragments from `DecompressorCollector`.
  """

  alias Brain.Analysis.{FeatureExtractor, Pipeline}
  alias Brain.Analysis.FeatureExtractor.ChunkFeatures

  require Logger

  @default_timeout_ms 60_000
  @default_min_success_ratio 0.95

  @doc """
  Vectorizes a single fragment text.

  Returns `{:ok, feature_vector}` or `{:error, reason}`.
  """
  def vectorize_fragment_text(text) when is_binary(text) do
    trimmed = String.trim(text)

    if trimmed == "" do
      {:error, :empty_text}
    else
      try do
        analysis = Pipeline.analyze_chunk(trimmed)
        {feature_vector, _word_feats} = FeatureExtractor.extract(analysis)

        if is_list(feature_vector) and feature_vector != [] and
             length(feature_vector) == ChunkFeatures.vector_dimension() do
          {:ok, feature_vector}
        else
          {:error, {:invalid_vector, length(feature_vector || [])}}
        end
      rescue
        e -> {:error, {:exception, Exception.message(e)}}
      catch
        kind, reason -> {:error, {kind, reason}}
      end
    end
  end

  def vectorize_fragment_text(_), do: {:error, :not_binary}

  @doc """
  Vectorizes a list of fragment maps (`%{"text" => ...}`).

  Options:
    - `:min_success_ratio` - fail if below this fraction succeed (default 0.95)
    - `:timeout_ms` - per-fragment async timeout
    - `:verbose` - log progress

  Returns `{:ok, fragments}` or `{:error, reason}`.
  """
  def vectorize_fragments(fragments, opts \\ []) when is_list(fragments) do
    min_ratio = Keyword.get(opts, :min_success_ratio, @default_min_success_ratio)
    timeout_ms = Keyword.get(opts, :timeout_ms, @default_timeout_ms)
    verbose? = Keyword.get(opts, :verbose, false)

    total = length(fragments)

    if total == 0 do
      {:ok, []}
    else
      if verbose? do
        Logger.info("FragmentVectorizer: vectorizing #{total} fragments...")
      end

      started_at = System.monotonic_time(:millisecond)

      results =
        fragments
        |> Task.async_stream(
          fn frag ->
            text = Map.get(frag, "text", "")

            case vectorize_fragment_text(text) do
              {:ok, fv} -> {:ok, fv}
              {:error, reason} -> {:error, reason}
            end
          end,
          max_concurrency: System.schedulers_online(),
          timeout: timeout_ms,
          on_timeout: :kill_task,
          ordered: true
        )
        |> Enum.to_list()

      {vectorized, failures} =
        Enum.zip(fragments, results)
        |> Enum.map_reduce(0, fn {frag, result}, fail_count ->
          case result do
            {:ok, {:ok, fv}} ->
              {Map.put(frag, "prototype_vector", fv), fail_count}

            _ ->
              {Map.put(frag, "prototype_vector", []), fail_count + 1}
          end
        end)
      success_count = Enum.count(vectorized, fn f -> Map.get(f, "prototype_vector", []) != [] end)
      elapsed = System.monotonic_time(:millisecond) - started_at

      if verbose? do
        Logger.info(
          "FragmentVectorizer: #{success_count}/#{total} succeeded in #{elapsed}ms, #{failures} failed"
        )
      end

      ratio = success_count / total

      if ratio < min_ratio do
        {:error,
         "Fragment vectorization below threshold: #{success_count}/#{total} (#{Float.round(ratio * 100, 1)}%). " <>
           "Run `mix train` (micro-classifiers and dependencies) and retry without --skip-vectorize."}
      else
        {:ok, vectorized}
      end
    end
  end

  @doc "Returns the expected feature vector dimension."
  def vector_dimension, do: ChunkFeatures.vector_dimension()
end
