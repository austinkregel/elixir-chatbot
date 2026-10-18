defmodule Atlas.Axes do
  @moduledoc """
  Context module for axis measurement runs.

  A run is one pass of a corpus through `Brain.Analysis.ChunkProfile`, stored
  with the provenance that says what it was measured under. See
  `Atlas.Schemas.AxisRun` for the tagged/untagged model.

  ## Writes are all-or-nothing

  `record_run/2` validates every observation before writing anything, and wraps
  the run and its observations in one transaction. A run whose observations are
  half-written is worse than no run: it looks like a complete measurement and
  quietly under-reports whichever axes fell in the missing half.

  This follows `Atlas.Lexicon.upsert_facts/1`, which takes the same position for
  the same reason.
  """

  import Ecto.Query

  alias Atlas.Repo
  alias Atlas.Schemas.{AxisObservation, AxisRun}

  # Postgres binds at most 65,535 parameters per statement. An observation row
  # has eleven columns, and the `feature_vector` array counts as one bind
  # regardless of its length, so a chunk of 1,000 binds 11,000.
  @chunk_size 1_000

  @doc """
  Records a run and its observations in one transaction.

  `observations` are maps accepted by `AxisObservation.changeset/2`, without
  `:run_id` — it is filled in from the run this call creates.

  Returns `{:ok, %{run: run, observations: count}}`. Returns
  `{:error, {:run, changeset}}` when the run itself is invalid, or
  `{:error, {:observation, index, changeset}}` naming the first bad observation.
  Nothing is written in either case.
  """
  @spec record_run(map(), [map()]) ::
          {:ok, %{run: AxisRun.t(), observations: non_neg_integer()}}
          | {:error, {:run, Ecto.Changeset.t()}}
          | {:error, {:observation, non_neg_integer(), Ecto.Changeset.t()}}
  def record_run(run_attrs, observations) when is_map(run_attrs) and is_list(observations) do
    run_changeset = AxisRun.changeset(%AxisRun{}, run_attrs)

    if run_changeset.valid? do
      # Validated outside the transaction so an invalid batch costs no database
      # work, and with a placeholder run_id so `validate_required(:run_id)`
      # checks the same shape that will be inserted.
      case validate_observations(observations) do
        {:ok, _} ->
          # Repo.insert/1 rather than insert!/1: a duplicate tag is a constraint
          # violation, and insert!/1 would raise Ecto.InvalidChangesetError
          # instead of returning the changeset that says which field collided.
          Repo.transaction(fn ->
            case Repo.insert(run_changeset) do
              {:ok, run} ->
                count = insert_observations!(run.id, observations)
                %{run: run, observations: count}

              {:error, changeset} ->
                Repo.rollback({:run, changeset})
            end
          end)

        {:error, _} = error ->
          error
      end
    else
      {:error, {:run, run_changeset}}
    end
  end

  @doc """
  Tags an existing run, promoting it to a durable reference point.

  The note is required — see `Atlas.Schemas.AxisRun`.
  """
  @spec tag_run(Ecto.UUID.t(), String.t(), String.t()) ::
          {:ok, AxisRun.t()} | {:error, Ecto.Changeset.t()} | {:error, :not_found}
  def tag_run(run_id, tag, note) when is_binary(tag) and is_binary(note) do
    case Repo.get(AxisRun, run_id) do
      nil ->
        {:error, :not_found}

      run ->
        run
        |> AxisRun.changeset(%{tag: tag, tag_note: note})
        |> Repo.update()
    end
  end

  @doc "The run with this tag, or `nil`."
  @spec get_by_tag(String.t()) :: AxisRun.t() | nil
  def get_by_tag(tag) when is_binary(tag), do: Repo.get_by(AxisRun, tag: tag)

  @doc "Runs, newest first."
  @spec list_runs(keyword()) :: [AxisRun.t()]
  def list_runs(opts \\ []) do
    query = from(r in AxisRun, order_by: [desc: r.inserted_at])
    query = if opts[:tagged_only], do: AxisRun.tagged(query), else: query
    query = if opts[:limit], do: limit(query, ^opts[:limit]), else: query

    Repo.all(query)
  end

  @doc """
  The per-axis computed/defaulted census for a run, as
  `%{axis => %{computed: n, defaulted: n}}`.

  An axis absent from a run's observations does not appear here — it is not
  reported as zero, because "not measured" and "measured as never computed" are
  different claims.
  """
  @spec census(Ecto.UUID.t()) :: %{String.t() => %{computed: non_neg_integer(), defaulted: non_neg_integer()}}
  def census(run_id) do
    AxisObservation.census(AxisObservation, run_id)
    |> Repo.all()
    |> Enum.reduce(%{}, fn {axis, status, count}, acc ->
      # Matched rather than String.to_existing_atom/1: that would turn a stored
      # value into an atom lookup, and atlas does not depend on brain, so the
      # atoms ChunkProfile uses are not guaranteed to exist in this VM.
      key =
        case status do
          "computed" -> :computed
          "defaulted" -> :defaulted
          other -> raise ArgumentError, "unknown axis observation status #{inspect(other)}"
        end

      Map.update(acc, axis, Map.put(%{computed: 0, defaulted: 0}, key, count), &Map.put(&1, key, count))
    end)
  end

  @doc """
  Deletes untagged runs older than `cutoff`, and their observations with them.

  Returns the number of runs deleted. Observations go via the
  `on_delete: :delete_all` foreign key rather than a second query, so a run and
  its rows cannot be separated.
  """
  @spec prune_untagged(DateTime.t()) :: non_neg_integer()
  def prune_untagged(%DateTime{} = cutoff) do
    {count, _} = AxisRun.prunable(AxisRun, cutoff) |> Repo.delete_all()
    count
  end

  # -- internals --------------------------------------------------------------

  defp validate_observations(observations) do
    placeholder = Ecto.UUID.generate()

    observations
    |> Enum.with_index()
    |> Enum.reduce_while({:ok, 0}, fn {attrs, index}, {:ok, n} ->
      changeset =
        AxisObservation.changeset(%AxisObservation{}, Map.put(attrs, :run_id, placeholder))

      if changeset.valid? do
        {:cont, {:ok, n + 1}}
      else
        {:halt, {:error, {:observation, index, changeset}}}
      end
    end)
  end

  defp insert_observations!(run_id, observations) do
    now = DateTime.utc_now()

    observations
    |> Enum.map(fn attrs ->
      attrs
      |> Map.take([:utterance_id, :axis, :value, :status, :reason, :feature_vector])
      |> Map.merge(%{
        id: Ecto.UUID.generate(),
        run_id: run_id,
        inserted_at: now,
        updated_at: now
      })
    end)
    |> Enum.chunk_every(@chunk_size)
    |> Enum.reduce(0, fn chunk, total ->
      {count, _} = Repo.insert_all(AxisObservation, chunk)
      total + count
    end)
  end
end
