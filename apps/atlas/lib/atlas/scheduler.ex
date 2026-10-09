defmodule Atlas.Scheduler do
  @moduledoc """
  Context module for the local scheduler's alarms, reminders, timers and
  calendar items (`Atlas.Schemas.SchedulerItem`). Every item belongs to a
  world, and every write names it.
  """

  import Ecto.Query

  alias Atlas.Repo
  alias Atlas.Schemas.SchedulerItem

  @doc "Every stored item of every world, oldest first."
  @spec list_items() :: [SchedulerItem.t()]
  def list_items do
    SchedulerItem
    |> order_by([i], asc: i.inserted_at)
    |> Repo.all()
  end

  @doc """
  Stores an item. `attrs` names its `:world_id` and `:kind` and the fields its
  kind carries. Raises on a missing or blank world, or an unknown kind.
  """
  @spec insert_item!(map()) :: SchedulerItem.t()
  def insert_item!(attrs) when is_map(attrs) do
    %SchedulerItem{}
    |> SchedulerItem.changeset(attrs)
    |> Repo.insert!()
  end

  @doc "Deletes the world's items of one kind. Returns how many were deleted."
  @spec delete_items!(String.t(), String.t()) :: non_neg_integer()
  def delete_items!(world_id, kind) when is_binary(world_id) and is_binary(kind) do
    {count, _} =
      SchedulerItem
      |> where([i], i.world_id == ^world_id and i.kind == ^kind)
      |> Repo.delete_all()

    count
  end
end
