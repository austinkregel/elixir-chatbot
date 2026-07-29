defmodule Fleet.MindWorld do
  @moduledoc """
  An agent's private mind-world: the `world_id` that scopes its own memory,
  beliefs, and JTMS. Keyed by `soul_id` so it is stable across restarts (the
  same soul always rehydrates the same mind), never collides with another soul,
  and stays out of the normal training-world namespace via the `"mind:"` prefix.

  This is what makes cognition per-agent: an officer runs `Brain.evaluate` in its
  mind-world, so everything it remembers/believes is isolated to it. Nothing
  crosses between minds except what is communicated (see `Fleet.Officer`).
  """

  @prefix "mind:"

  @doc "The mind-world id for a soul."
  def id(soul_id) when is_binary(soul_id), do: @prefix <> soul_id

  @doc "Is this world a mind-world?"
  def mind_world?(world_id) when is_binary(world_id), do: String.starts_with?(world_id, @prefix)
  def mind_world?(_), do: false

  @doc "The soul_id owning a mind-world id (inverse of id/1)."
  def soul_id(@prefix <> soul_id), do: soul_id
  def soul_id(_), do: nil
end
