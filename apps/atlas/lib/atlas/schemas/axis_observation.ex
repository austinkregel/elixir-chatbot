defmodule Atlas.Schemas.AxisObservation do
  @moduledoc """
  One axis's value for one utterance within an `Atlas.Schemas.AxisRun`.

  18 axes over ~5,274 utterances is ~95k rows for a full run, so these are
  written with `Repo.insert_all/3` in chunks rather than one at a time — see
  `Atlas.Axes.record_run/2`.

  ## Why `value` is a string for every axis

  Three of the 18 axes are continuous (`polarity`, `slot_completeness`,
  `novelty_score`) and fifteen are categorical. Storing them in typed columns
  would restate a fact `Brain.Analysis.ChunkProfile.axis_manifest/0` already
  declares, and a second declaration can disagree with the first. The manifest
  owns the kind; this table records what was observed.

  ## Why `status` is not derivable

  `status` is `"computed"` or `"defaulted"`, from the axis's provenance entry. It
  cannot be inferred from `value`, because a defaulted axis holds exactly its
  declared default — so `polarity: 0.0` means "affirmative" when computed and
  "never determined" when defaulted, and those are different observations that
  look identical. Recording it is what turns "is this axis dead?" from an
  investigation into a query.

  `reason` carries why an axis defaulted (`no_negation_particle_matched`,
  `target_not_self`, `absent`, ...), which is the difference between an axis
  being wrong and an axis being unreachable.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  alias Atlas.Schemas.AxisRun

  @primary_key {:id, :binary_id, autogenerate: true}

  @type t :: %__MODULE__{}

  @statuses ~w(computed defaulted)

  schema "atlas_axis_observations" do
    belongs_to :run, AxisRun, type: :binary_id
    field :utterance_id, :string
    field :axis, :string
    field :value, :string
    field :status, :string
    field :reason, :string
    field :feature_vector, {:array, :float}

    timestamps(type: :utc_datetime_usec)
  end

  @required_fields ~w(run_id utterance_id axis value status)a
  @optional_fields ~w(reason feature_vector)a

  @doc false
  def changeset(observation, attrs) do
    observation
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> validate_inclusion(:status, @statuses)
    |> validate_reason_accompanies_default()
    |> foreign_key_constraint(:run_id)
  end

  @doc "The two permitted `status` values."
  def statuses, do: @statuses

  # A defaulted axis with no reason is the gap this table exists to close: it
  # records that something did not happen without recording why, which is the
  # state task 081 had to reconstruct by hand.
  defp validate_reason_accompanies_default(changeset) do
    with "defaulted" <- get_field(changeset, :status),
         nil <- get_field(changeset, :reason) do
      add_error(changeset, :reason, "is required when status is defaulted")
    else
      _ -> changeset
    end
  end

  @doc "Query the observations of one run."
  def for_run(query \\ __MODULE__, run_id) do
    from(o in query, where: o.run_id == ^run_id)
  end

  @doc "Query one axis across a run."
  def for_axis(query \\ __MODULE__, run_id, axis) do
    from(o in query, where: o.run_id == ^run_id and o.axis == ^axis)
  end

  @doc """
  Per-axis `{axis, status, count}` for a run — the computed/defaulted census
  that `.claude/corpus/provenance_report.exs` computes in memory over 16
  sentences.
  """
  def census(query \\ __MODULE__, run_id) do
    from(o in query,
      where: o.run_id == ^run_id,
      group_by: [o.axis, o.status],
      select: {o.axis, o.status, count(o.id)}
    )
  end
end
