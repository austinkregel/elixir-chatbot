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
  Stores an item. `attrs` names its `:world_id`, `:kind`, `:resolution` and
  `:fire_state` and the fields its kind carries, with the resolved fields
  when it is resolved and the firing fields its fire state carries
  (`Atlas.Schemas.SchedulerItem`). Raises on a missing or blank world, an
  unknown kind, resolution or fire state, a resolved item without its turn,
  zone, offset or start, resolved fields on an item that is not resolved, a
  fire state without the firing fields it carries or with others, or an
  owner who is no web user.
  """
  @spec insert_item!(map()) :: SchedulerItem.t()
  def insert_item!(attrs) when is_map(attrs) do
    %SchedulerItem{}
    |> SchedulerItem.changeset(attrs)
    |> Repo.insert!()
  end

  @doc """
  Changes the stored item `id`, with the same checks as `insert_item!/1`, and
  returns it as stored. Raises when no item has that id.
  """
  @spec update_item!(Ecto.UUID.t(), map()) :: SchedulerItem.t()
  def update_item!(id, attrs) when is_binary(id) and is_map(attrs) do
    SchedulerItem
    |> Repo.get!(id)
    |> SchedulerItem.changeset(attrs)
    |> Repo.update!()
  end

  @doc "When the soonest pending item is due, or `nil` when no item is pending."
  @spec next_due_at() :: DateTime.t() | nil
  def next_due_at do
    SchedulerItem
    |> where([i], i.fire_state == "pending")
    |> select([i], min(i.due_at))
    |> Repo.one()
  end

  @doc "Every pending item due at or before `instant`, soonest first."
  @spec due_items(DateTime.t()) :: [SchedulerItem.t()]
  def due_items(%DateTime{} = instant) do
    SchedulerItem
    |> where([i], i.fire_state == "pending" and i.due_at <= ^instant)
    |> order_by([i], asc: i.due_at, asc: i.inserted_at)
    |> Repo.all()
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

  @doc """
  Deletes the world's items of one kind whose ids are given, and returns how
  many were deleted. Every id must name an item of that world and kind: when
  one does not, nothing is deleted and this raises, naming the ids.
  """
  @spec delete_items_by_id!(String.t(), String.t(), [Ecto.UUID.t()]) :: pos_integer()
  def delete_items_by_id!(world_id, kind, [_ | _] = ids) when is_binary(world_id) and is_binary(kind) do
    ids = Enum.uniq(ids)

    {:ok, count} =
      Repo.transaction(fn ->
        {count, _} =
          SchedulerItem
          |> where([i], i.world_id == ^world_id and i.kind == ^kind and i.id in ^ids)
          |> Repo.delete_all()

        if count != length(ids) do
          raise ArgumentError,
                "Atlas.Scheduler.delete_items_by_id!: only #{count} of the ids #{inspect(ids)} name a " <>
                  "#{inspect(kind)} item of world #{inspect(world_id)}, so none was deleted"
        end

        count
      end)

    count
  end
end
