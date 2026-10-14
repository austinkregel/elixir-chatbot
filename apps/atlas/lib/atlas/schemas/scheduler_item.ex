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
  - `time_zone`: the zone its days and clock times were read in.
  - `starts_at`, `ends_at`: when the time it names begins and ends (`ends_at`
    exclusive; equal to `starts_at` for a point in time; `nil` for a time with
    no end, "later"). A timer starts when it was set and ends when it runs
    out.
  - `clock`: the time of day said, `nil` when none was.

  `inserted_at` is when the item was set.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}

  @kinds ~w(alarm reminder timer calendar)

  @resolutions ~w(resolved nothing_said recorded_before_resolution)

  # The fields only a resolved item has.
  @resolved_fields [:turn_time, :time_zone, :starts_at]

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
    field :starts_at, :utc_datetime_usec
    field :ends_at, :utc_datetime_usec
    field :clock, :time

    timestamps(type: :utc_datetime_usec)
  end

  @doc "The kinds of item the scheduler stores."
  def kinds, do: @kinds

  @doc "How an item's time can have been resolved."
  def resolutions, do: @resolutions

  # An empty string is kept as given: an alarm set with no time reads back with
  # the empty time it was set with. A blank world or kind is still refused.
  # Every item names its resolution; a resolved one names its turn, zone and
  # start, and any other carries none of the resolved fields.
  @doc false
  def changeset(item, attrs) do
    item
    |> cast(
      attrs,
      [:world_id, :kind, :time, :date, :text, :duration, :status, :resolution] ++
        [:turn_time, :time_zone, :starts_at, :ends_at, :clock],
      empty_values: []
    )
    |> validate_required([:world_id, :kind, :resolution])
    |> validate_change(:world_id, &not_blank/2)
    |> validate_inclusion(:kind, @kinds)
    |> validate_inclusion(:resolution, @resolutions)
    |> validate_resolved_fields()
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

  defp not_blank(field, value) do
    if String.trim(value) == "", do: [{field, "must not be blank"}], else: []
  end
end
