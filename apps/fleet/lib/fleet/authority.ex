defmodule Fleet.Authority do
  @moduledoc """
  Authorities are the real teeth of a grant: named terms that gate behaviour an
  ensign actually performs. An order's grant is the agent's action scope; an
  agent may only do what its held authorities permit.

  Vocabulary (extensible as capabilities grow):

    * `:cognition`        — may run `Brain.evaluate/3` (the primary gate).
    * `{:world, world_id}`— may operate in that world (cognition binds the soul
      via the world roster).
    * `:issue_orders`     — a CO may hand ORDERs to its reports.
    * `:relieve`          — may relieve a subordinate of duty.

  Grants are held as a `MapSet` in `state.context_tags.grants`.
  """

  alias Fleet.Order

  @doc "Normalise a list/MapSet of authorities into a MapSet."
  def to_set(%MapSet{} = set), do: set
  def to_set(list) when is_list(list), do: MapSet.new(list)
  def to_set(nil), do: MapSet.new()

  @doc "Does the grant set include this authority?"
  def holds?(%MapSet{} = grants, authority), do: MapSet.member?(grants, authority)

  @doc "The authorities an order requires to execute, derived from the order itself."
  def required_for(%Order{world_id: world_id}), do: [:cognition, {:world, world_id}]

  @doc "Merge conferred authorities (from an ORDER's grant or a GRANT signal) into a grant set."
  def confer(%MapSet{} = grants, authorities) when is_list(authorities),
    do: Enum.reduce(authorities, grants, fn a, acc -> MapSet.put(acc, a) end)

  def confer(%MapSet{} = grants, authority), do: MapSet.put(grants, authority)

  @doc """
  A commanding officer may only GRANT an authority it itself holds — you cannot
  delegate authority you lack. The real basis for a GRANT/DENY decision.
  """
  def grantable?(%MapSet{} = co_grants, authority), do: MapSet.member?(co_grants, authority)

  @doc "Authorities in `required` that are NOT present in `grants` (the block reason list)."
  def missing(%MapSet{} = grants, required) when is_list(required),
    do: Enum.reject(required, &holds?(grants, &1))

  @doc "The authorities an order confers (its grant scope)."
  def conferred_by(%Order{grant: grant}) when is_map(grant), do: Map.get(grant, :authorities, [])
  def conferred_by(_), do: []
end
