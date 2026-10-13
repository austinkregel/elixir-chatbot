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

  `inserted_at` is when the item was set.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}

  @kinds ~w(alarm reminder timer calendar)

  @type t :: %__MODULE__{}

  schema "atlas_scheduler_items" do
    field :world_id, :string
    field :kind, :string
    field :time, :string
    field :date, :string
    field :text, :string
    field :duration, :string
    field :status, :string

    timestamps(type: :utc_datetime_usec)
  end

  @doc "The kinds of item the scheduler stores."
  def kinds, do: @kinds

  # An empty string is kept as given: an alarm set with no time reads back with
  # the empty time it was set with. A blank world or kind is still refused.
  @doc false
  def changeset(item, attrs) do
    item
    |> cast(attrs, [:world_id, :kind, :time, :date, :text, :duration, :status], empty_values: [])
    |> validate_required([:world_id, :kind])
    |> validate_change(:world_id, &not_blank/2)
    |> validate_inclusion(:kind, @kinds)
  end

  defp not_blank(field, value) do
    if String.trim(value) == "", do: [{field, "must not be blank"}], else: []
  end
end
