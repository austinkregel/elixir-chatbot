defmodule Fleet.Ship do
  @moduledoc """
  The identity of THIS ship — one running instance of the umbrella.

  A "ship" is an *instance*, not the umbrella itself: one ship runs now, but a
  fleet-of-ships (including specialized research/science ships) may be commissioned
  later, each a differently-configured instance. `id/0` is the authoritative id of
  the ship this node *is* — the ship whose systems this node can actually answer
  for, and the default ship every locally-commissioned agent is posted to
  (`Fleet.Ensign`), stamped onto every audit/black-box record (`Fleet.Audit`), and
  compared against a reader's commission by `Fleet.Clearance`.

      config :fleet, ship_id: "USS-DREAMCOM"   # env-overridable via SHIP_ID
  """

  @default "USS-DREAMCOM"

  @doc "This ship's id — from `config :fleet, :ship_id` (defaults to `#{@default}`)."
  @spec id() :: String.t()
  def id, do: Application.get_env(:fleet, :ship_id, @default)
end
