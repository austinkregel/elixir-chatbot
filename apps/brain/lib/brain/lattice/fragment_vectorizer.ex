defmodule Brain.Lattice.FragmentVectorizer do
  @moduledoc """
  Computes dense feature vectors for lattice phrase fragments via the analysis pipeline.

  Used at inventory build time (`mix gen_lattice_data`) and when feeding runtime
  fragments from `DecompressorCollector`.

  A fragment's vector is the one the pipeline classified the fragment's intent
  from (`analysis.feature_vector`), the vector a turn's input is scored with
  (`Brain.Response.LatticeScorer`), so a fragment and an input are compared in
  the same space. Extracting a vector again from the finished analysis gave a
  different one: by then pass 2 had refined the speech act and filled the slots.

  A fragment that cannot be vectorized raises, naming it: a fragment left
  without a vector scores nothing on its features, which silently skews the
  lattice toward tone alone.
  """

  alias Brain.Analysis.Pipeline
  alias Brain.Analysis.FeatureExtractor.ChunkFeatures

  require Logger

  @default_timeout_ms 60_000

  @doc """
  The feature vector of a fragment's text, analyzed in the world `world_id`:
  `{:ok, feature_vector}`. A text that is blank or not a string raises, and so
  does an analysis that kept no vector of the feature extractor's dimension.
  """
  def vectorize_fragment_text(text, world_id) when is_binary(text) do
    world_id!(world_id)
    trimmed = String.trim(text)

    if trimmed == "" do
      raise ArgumentError, "FragmentVectorizer: a fragment's text is blank, so there is nothing to vectorize: #{inspect(text)}"
    end

    # A fragment is vectorized, not said: its analysis writes nothing to the world.
    analysis = Pipeline.analyze_chunk(trimmed, world_id: world_id, side_effects: false)
    dimension = ChunkFeatures.vector_dimension()

    case analysis.feature_vector do
      [_ | _] = vector when length(vector) == dimension ->
        {:ok, vector}

      other ->
        raise "FragmentVectorizer: the pipeline kept no classified feature vector of dimension #{dimension} " <>
                "for the fragment #{inspect(trimmed)}: #{inspect(other, limit: 5)}"
    end
  end

  def vectorize_fragment_text(text, _world_id) do
    raise ArgumentError, "FragmentVectorizer: a fragment's text must be a string, got: #{inspect(text)}"
  end

  # Checked before the analysis runs, so a missing world raises before any
  # fragment is analyzed.
  defp world_id!(world_id) when is_binary(world_id) and world_id != "", do: world_id

  defp world_id!(other) do
    raise ArgumentError,
          "FragmentVectorizer needs a world id, the world fragments are analyzed in, got: #{inspect(other)}"
  end

  @doc """
  Vectorizes a list of fragment maps (`%{"text" => ...}`), returning each with
  its `"prototype_vector"`, in order.

  Options:
    - `:world_id` - required: the world fragments are analyzed in. Absent raises.
    - `:timeout_ms` - per-fragment async timeout
    - `:verbose` - log progress

  Every fragment is vectorized or the call raises: a fragment whose analysis
  raised, or that was killed at the timeout, is named in the error.
  """
  def vectorize_fragments(fragments, opts) when is_list(fragments) do
    world_id = world_id!(Keyword.get(opts, :world_id))
    timeout_ms = Keyword.get(opts, :timeout_ms, @default_timeout_ms)
    verbose? = Keyword.get(opts, :verbose, false)

    total = length(fragments)

    if verbose? do
      Logger.info("FragmentVectorizer: vectorizing #{total} fragments...")
    end

    started_at = System.monotonic_time(:millisecond)

    vectorized =
      fragments
      |> Task.async_stream(
        fn frag -> vectorize_fragment_text(Map.fetch!(frag, "text"), world_id) end,
        max_concurrency: System.schedulers_online(),
        timeout: timeout_ms,
        on_timeout: :kill_task,
        ordered: true
      )
      |> Enum.zip(fragments)
      |> Enum.map(fn
        {{:ok, {:ok, fv}}, frag} ->
          Map.put(frag, "prototype_vector", fv)

        {{:exit, reason}, frag} ->
          raise "FragmentVectorizer: the fragment #{inspect(Map.get(frag, "text"))} (#{inspect(Map.get(frag, "id"))}) " <>
                  "was not vectorized: #{inspect(reason, limit: 10)}"
      end)

    if verbose? do
      Logger.info("FragmentVectorizer: #{total} fragments in #{System.monotonic_time(:millisecond) - started_at}ms")
    end

    vectorized
  end

  @doc "Returns the expected feature vector dimension."
  def vector_dimension, do: ChunkFeatures.vector_dimension()
end
