defmodule World.Roster do
  @moduledoc """
  The crew residing in a World.

  A World respects the Souls that reside in it: this resolves which resident
  Soul is *acting* so the generation chain can render that Soul's constitution
  as the system prompt.

  Souls are portable — they are owned by `Brain.Soul` (their own files), not by
  the World. A World only holds references (ids) in its `:residents`. The same
  Soul id may reside in several Worlds, which is what lets the same Soul be run
  under different world framings and compared.
  """

  alias World.Manager

  @doc """
  Soul ids residing in the world. Thin slice: a world's own residents only;
  inheriting residents along the `:base_world` chain (as entities/memory already
  do) is a later refinement.
  """
  @spec residents(String.t()) :: [String.t()]
  def residents(world_id) when is_binary(world_id) do
    case Manager.get(world_id) do
      {:ok, world} -> Map.get(world, :residents, [])
      _ -> []
    end
  end

  @doc """
  The acting resident's Soul (`%Brain.Soul{}`), or `nil` when the world has no
  residents or the soul is not on file. Thin slice: the first resident acts;
  turn-taking / explicit actor selection is a later refinement.
  """
  @spec acting_soul(String.t()) :: Brain.Soul.t() | nil
  def acting_soul(world_id) when is_binary(world_id) do
    with [soul_id | _] <- residents(world_id),
         {:ok, soul} <- Brain.Soul.get(soul_id) do
      soul
    else
      _ -> nil
    end
  end
end
