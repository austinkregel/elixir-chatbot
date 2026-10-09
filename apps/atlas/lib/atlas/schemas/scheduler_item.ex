defmodule Atlas.Schemas.SchedulerItem do
  @moduledoc """
  An alarm, reminder, timer or calendar item set through the local scheduler,
  stored so it survives a restart. It belongs to the world it was set in.

  Which fields an item carries depends on its kind:

  - `"alarm"`: `time`, `status`
  - `"reminder"`: `text`, `time`, `status`
  - `"timer"`: `duration`, `status`
  - `"calendar"`: `text` (the event), `time`

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
    |> cast(attrs, [:world_id, :kind, :time, :text, :duration, :status], empty_values: [])
    |> validate_required([:world_id, :kind])
    |> validate_change(:world_id, &not_blank/2)
    |> validate_inclusion(:kind, @kinds)
  end

  defp not_blank(field, value) do
    if String.trim(value) == "", do: [{field, "must not be blank"}], else: []
  end
end
