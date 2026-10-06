defmodule Atlas.Schemas.VerificationCase do
  @moduledoc """
  One human-curated verification scenario: an input to a subsystem, what a
  person decided the right answer is, and what the subsystem last produced.

  Task 039 exists because a feature nobody can exercise in isolation is a
  feature nobody can check. Austin's constraint, recorded there: *"every feature
  gets its own page so a human can exercise it in isolation and verify it
  works"*, and the harness persists expected-versus-actual rather than offering
  transient inspection, so a scenario checked once stays checked.

  ## These are not a gold standard

  A gold standard is a bulk corpus for training and evaluation — tasks 013-015.
  These are a handful of scenarios a person reasoned about, most with no
  gold-standard analogue. Conflating the two would put hand-picked cases into
  training data, which is the contamination task 079 spent a day removing.

  ## `expected` is partial on purpose

  A person asserts the parts of the output they have an opinion about, not the
  whole term. A full-term expectation breaks on every unrelated change and so
  gets deleted rather than maintained. The cost is that an unasserted field can
  regress unnoticed, so the partiality is **made visible** — `ChatWeb.Harness.Diff`
  reports how many keys were checked against how many the subsystem returned,
  rather than letting a green case imply the whole output was verified.

  ## An edited expectation un-passes the case, explicitly

  `Atlas.Verification.save_case/1` resets `status` to `"pending"` when `input`,
  `expected` or `tolerance` changes, so a stored verdict always describes the
  values currently in the row.

  An earlier version of this module also derived staleness from the timestamps —
  `updated_at` later than `last_run_at` meaning the expectation had moved since
  the run. That was removed: Ecto stamps `updated_at` at write time, microseconds
  *after* the `DateTime.utc_now/0` that produced `last_run_at`, so every
  freshly-run case read as stale. Widening the comparison to a tolerance would
  have hidden the race rather than fixed it, and the real objection is that it
  was a second, implicit source for a fact the status field already states.

  `last_run_at` remains, for the honest question it answers on its own: when did
  this case last run.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  alias Atlas.Verification.Comparison
  alias Atlas.Verification.Subsystems

  @primary_key {:id, :binary_id, autogenerate: true}

  @type t :: %__MODULE__{}

  # "pending" is the state of a case that has never run. It is the default
  # because a case that has not run has not passed, and defaulting to "pass"
  # would make /verify's counts wrong on the day it ships.
  @statuses ~w(pending pass fail error)

  schema "atlas_verification_cases" do
    field :subsystem, :string
    field :name, :string
    field :input, :map, default: %{}
    field :expected, :map, default: %{}
    field :last_actual, :map
    field :tolerance, :float
    field :status, :string, default: "pending"
    field :last_run_at, :utc_datetime_usec
    field :world_id, :string

    timestamps(type: :utc_datetime_usec)
  end

  @required_fields ~w(subsystem name world_id)a
  @optional_fields ~w(input expected last_actual tolerance status last_run_at)a

  @doc "Every status a case can hold."
  @spec statuses() :: [String.t()]
  def statuses, do: @statuses

  @doc false
  def changeset(verification_case, attrs) do
    verification_case
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> validate_inclusion(:status, @statuses)
    |> validate_subsystem()
    |> validate_trimmed(:name)
    |> validate_trimmed(:world_id)
    |> validate_number(:tolerance, greater_than_or_equal_to: 0.0)
    |> validate_expectation_asserts_something()
    |> unique_constraint([:subsystem, :world_id, :name],
      name: :atlas_verification_cases_identity_index
    )
  end

  # The boundary enforcement task 077 criterion 4 asks for: a subsystem outside
  # the declared vocabulary cannot enter storage. Without this a typo creates a
  # case that no page will ever run and that /verify will never list, so it sits
  # in the store reading "pending" forever.
  defp validate_subsystem(changeset) do
    validate_change(changeset, :subsystem, fn :subsystem, id ->
      if Subsystems.known?(id) do
        []
      else
        [
          subsystem:
            "is not a declared subsystem; declared: #{Enum.join(Subsystems.ids(), ", ")}"
        ]
      end
    end)
  end

  # A case whose expectation asserts nothing cannot fail, so it would sit in
  # /verify's counts as a permanent pass that verifies no behaviour. Rejecting
  # it here is why `Comparison.compare/3` can treat an empty expectation as a
  # caller error rather than a state to report.
  # Checked with get_field/2 rather than validate_change/3, which only runs for
  # a field the cast actually changed. `expected` defaults to %{}, so casting
  # %{} is not a change and validate_change/3 skipped it entirely — the
  # validation existed and never fired. Caught by
  # Atlas.VerificationTest "an expectation that asserts nothing is refused".
  defp validate_expectation_asserts_something(changeset) do
    if Comparison.leaf_count(get_field(changeset, :expected) || %{}) > 0 do
      changeset
    else
      add_error(
        changeset,
        :expected,
        "must assert at least one value; a case that asserts nothing cannot fail"
      )
    end
  end

  defp validate_trimmed(changeset, field) do
    validate_change(changeset, field, fn ^field, value ->
      if is_binary(value) and String.trim(value) != "" do
        []
      else
        [{field, "must be non-empty and not only whitespace"}]
      end
    end)
  end

  @doc "Query the cases for one subsystem."
  def for_subsystem(query \\ __MODULE__, subsystem) when is_binary(subsystem) do
    from(c in query, where: c.subsystem == ^subsystem)
  end

  @doc "Query the cases recorded in one world."
  def in_world(query \\ __MODULE__, world_id) when is_binary(world_id) do
    from(c in query, where: c.world_id == ^world_id)
  end

  @doc "Query the cases holding one status."
  def with_status(query \\ __MODULE__, status) when is_binary(status) do
    from(c in query, where: c.status == ^status)
  end

  @doc "Order by name, so a page lists cases the same way twice."
  def by_name(query \\ __MODULE__) do
    from(c in query, order_by: [asc: c.name])
  end
end
