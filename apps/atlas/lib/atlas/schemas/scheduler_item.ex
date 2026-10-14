defmodule Atlas.Schemas.SchedulerItem do
  @moduledoc """
  An alarm, reminder, timer or calendar item set through the local scheduler,
  stored so it survives a restart. It belongs to the world it was set in.

  Which fields an item carries depends on its kind:

  - `"alarm"`: `time`, `date`, `status`
  - `"reminder"`: `text`, `time`, `date`, `status`
  - `"timer"`: `duration`, `status`
  - `"calendar"`: `text` (the event), `time`, `date`

  `date` is the day the item is for, as it was given. An item stored before
  dates were stored has none (`nil`), as does every timer.

  Beside the text it was given as, each item keeps what its date and time,
  or a timer's duration, name, resolved against the turn it was set in
  (`Brain.Temporal.Resolver`):

  - `resolution`: `"resolved"`; `"nothing_said"` when no date, time or
    duration was given; or `"recorded_before_resolution"` for an item stored
    before resolutions were, whose turn was never recorded. Only a resolved
    item has the fields below; for the others they are `nil`.
  - `turn_time`: the moment of the turn it was set in.
  - `time_zone`: the name of the zone its days and clock times were read in.
  - `utc_offset`: the zone's offset from UTC at `starts_at`, in seconds east
    of UTC.
  - `starts_at`, `ends_at`: when the time it names begins and ends (`ends_at`
    exclusive; equal to `starts_at` for a point in time; `nil` for a time with
    no end, "later"). A timer starts when it was set and ends when it runs
    out.
  - `clock`: the time of day said, `nil` when none was.

  Each item belongs to the web user who set it (`web_user_id`), who is told
  when it fires, and records whether it has:

  - `fire_state`: `"pending"`, it fires at `due_at`; `"fired"`;
    `"fired_late"`, it was due while the scheduler was not running and fired
    when it started; `"no_due_time"`, nothing it was set with names an
    instant to fire at; or `"owner_not_recorded"`, for an item stored before
    owners were, which has no owner and never fires.
  - `due_at`: when it fires; `nil` for an item with no due time or no owner.
  - `fired_at`: when it fired; `nil` until it has.

  A pending item has its owner and due time and has not fired; a fired one
  has all three; an item with no due time has its owner and neither of the
  others; one with no owner recorded has none of the three.

  `inserted_at` is when the item was set.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}

  @kinds ~w(alarm reminder timer calendar)

  @resolutions ~w(resolved nothing_said recorded_before_resolution)

  @fire_states ~w(pending fired fired_late no_due_time owner_not_recorded)

  # The fields only a resolved item has.
  @resolved_fields [:turn_time, :time_zone, :utc_offset, :starts_at]

  # The firing fields each fire state carries; it carries none of the others.
  @fire_fields %{
    "pending" => [:web_user_id, :due_at],
    "fired" => [:web_user_id, :due_at, :fired_at],
    "fired_late" => [:web_user_id, :due_at, :fired_at],
    "no_due_time" => [:web_user_id],
    "owner_not_recorded" => []
  }

  @type t :: %__MODULE__{}

  schema "atlas_scheduler_items" do
    field :world_id, :string
    field :kind, :string
    field :time, :string
    field :date, :string
    field :text, :string
    field :duration, :string
    field :status, :string
    field :resolution, :string
    field :turn_time, :utc_datetime_usec
    field :time_zone, :string
    field :utc_offset, :integer
    field :starts_at, :utc_datetime_usec
    field :ends_at, :utc_datetime_usec
    field :clock, :time
    field :web_user_id, :binary_id
    field :due_at, :utc_datetime_usec
    field :fire_state, :string
    field :fired_at, :utc_datetime_usec

    timestamps(type: :utc_datetime_usec)
  end

  @doc "The kinds of item the scheduler stores."
  def kinds, do: @kinds

  @doc "How an item's time can have been resolved."
  def resolutions, do: @resolutions

  @doc "The states of an item's firing."
  def fire_states, do: @fire_states

  # An empty string is kept as given: an alarm set with no time reads back with
  # the empty time it was set with. A blank world or kind is still refused.
  # Every item names its resolution; a resolved one names its turn, zone,
  # offset and start, and any other carries none of the resolved fields. Every
  # item names its fire state, and carries exactly that state's firing fields.
  @doc false
  def changeset(item, attrs) do
    item
    |> cast(
      attrs,
      [:world_id, :kind, :time, :date, :text, :duration, :status, :resolution] ++
        [:turn_time, :time_zone, :utc_offset, :starts_at, :ends_at, :clock] ++
        [:web_user_id, :due_at, :fire_state, :fired_at],
      empty_values: []
    )
    |> validate_required([:world_id, :kind, :resolution, :fire_state])
    |> validate_change(:world_id, &not_blank/2)
    |> validate_inclusion(:kind, @kinds)
    |> validate_inclusion(:resolution, @resolutions)
    |> validate_inclusion(:fire_state, @fire_states)
    |> validate_resolved_fields()
    |> validate_fire_fields()
    |> foreign_key_constraint(:web_user_id)
  end

  defp validate_resolved_fields(changeset) do
    case get_field(changeset, :resolution) do
      "resolved" ->
        validate_required(changeset, @resolved_fields)

      _other ->
        Enum.reduce(@resolved_fields ++ [:ends_at, :clock], changeset, fn field, changeset ->
          if get_field(changeset, field) == nil,
            do: changeset,
            else: add_error(changeset, field, "is only kept for a resolved item")
        end)
    end
  end

  defp validate_fire_fields(changeset) do
    case Map.fetch(@fire_fields, get_field(changeset, :fire_state)) do
      {:ok, carried} ->
        absent = [:web_user_id, :due_at, :fired_at] -- carried

        Enum.reduce(absent, validate_required(changeset, carried), fn field, changeset ->
          if get_field(changeset, field) == nil,
            do: changeset,
            else: add_error(changeset, field, "is not kept for an item #{get_field(changeset, :fire_state)}")
        end)

      # An unknown or missing fire state is refused above.
      :error ->
        changeset
    end
  end

  defp not_blank(field, value) do
    if String.trim(value) == "", do: [{field, "must not be blank"}], else: []
  end
end
