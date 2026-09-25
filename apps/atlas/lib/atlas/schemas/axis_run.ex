defmodule Atlas.Schemas.AxisRun do
  @moduledoc """
  One measurement run over the 18 `Brain.Analysis.ChunkProfile` axes.

  A run records what a snapshot was taken *under*, so that two snapshots can be
  compared and a difference attributed. Task 072 is the case this exists for:
  six classifiers were trained on feature vectors whose memory dimensions later
  inverted, the vector length never changed, and no record of either run existed
  to diff.

  ## Tagged and untagged runs

  Runs accumulate freely and cheaply; an untagged run is disposable. **Tagging**
  promotes one to a durable reference point, with a note saying what it
  represents. Comparison is always run-vs-run or run-vs-tag — no run is
  privileged by default, and a "golden set" is not authored up front but emerges
  from tagged snapshots once an axis looks right.

  That is why `tag` is nullable with a partial unique index rather than a
  required field: the storage layer does not decide which run is authoritative.

  ## Comparability

  `schema_fingerprint` is lifted out of `provenance` into its own column because
  it is the precondition for comparison, not merely a property of the run. Two
  runs whose fingerprints differ hold vectors whose dimension *names* differ,
  and comparing them silently compares different things. This is not
  hypothetical: the fingerprint moved from `fb3ce7f1e4739cb3` to
  `bc289842ba5ccb3d` between 2026-09-12 and 2026-09-25 with the extractor code
  unchanged, because feature group 23 reads its names from the AGE graph at call
  time.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  @primary_key {:id, :binary_id, autogenerate: true}

  @type t :: %__MODULE__{}

  schema "atlas_axis_runs" do
    field :provenance, :map, default: %{}
    field :schema_fingerprint, :string
    field :corpus_sha256, :string
    field :utterance_count, :integer
    field :tag, :string
    field :tag_note, :string

    has_many :observations, Atlas.Schemas.AxisObservation, foreign_key: :run_id

    timestamps(type: :utc_datetime_usec)
  end

  @required_fields ~w(provenance schema_fingerprint corpus_sha256 utterance_count)a
  @optional_fields ~w(tag tag_note)a

  @doc false
  def changeset(run, attrs) do
    run
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> validate_number(:utterance_count, greater_than: 0)
    |> validate_format(:schema_fingerprint, ~r/^[0-9a-f]{16}$/)
    |> validate_tag_note()
    |> unique_constraint(:tag, name: :atlas_axis_runs_tag_index)
  end

  # A tag without a note is a reference point nobody can interpret later. The
  # whole value of tagging is that it says what the run represents, so the note
  # is required exactly when the tag is present.
  defp validate_tag_note(changeset) do
    case {get_field(changeset, :tag), get_field(changeset, :tag_note)} do
      {nil, _} ->
        changeset

      {_tag, note} when is_binary(note) ->
        if String.trim(note) == "" do
          add_error(changeset, :tag_note, "is required when a run is tagged")
        else
          changeset
        end

      {_tag, _} ->
        add_error(changeset, :tag_note, "is required when a run is tagged")
    end
  end

  @doc "Query only tagged runs, newest first."
  def tagged(query \\ __MODULE__) do
    from(r in query, where: not is_nil(r.tag), order_by: [desc: r.inserted_at])
  end

  @doc """
  Query runs that are comparable to `fingerprint`.

  Anything outside this set was produced under a different vector schema.
  """
  def comparable_to(query \\ __MODULE__, fingerprint) when is_binary(fingerprint) do
    from(r in query, where: r.schema_fingerprint == ^fingerprint)
  end

  @doc "Query untagged runs older than `cutoff` — the prunable set."
  def prunable(query \\ __MODULE__, %DateTime{} = cutoff) do
    from(r in query, where: is_nil(r.tag) and r.inserted_at < ^cutoff)
  end
end
