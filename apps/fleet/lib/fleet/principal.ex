defmodule Fleet.Principal do
  @moduledoc """
  A normalized **reader** — agent or human — that `Fleet.Clearance` reasons over.

  Built from RUNTIME state (an ensign's `context_tags` + `duty`, or an authenticated
  human session), NEVER from anything the model wrote — the same trust invariant as
  `Fleet.Comms.attribute/1` (the Registry vouches for who a message is from). Two
  kinds:

    * `:ensign` — a commissioned agent, with its billet, live duty, ship commission,
      and chain position.
    * `:admiral` — the human Admiralty: top of seniority, fleet-wide (`ship_id: nil`),
      always on duty. Bypasses the ship and chain gates but is still routed through
      `can_read?/4` so every Admiralty read is auditable.
  """

  @enforce_keys [:kind]
  defstruct kind: nil, id: nil, rank: :ensign, duty: :active, ship_id: nil, co: nil, reports: []

  @type t :: %__MODULE__{
          kind: :ensign | :admiral,
          id: String.t() | nil,
          rank: atom(),
          duty: :active | :relieved,
          ship_id: String.t() | nil,
          co: String.t() | nil,
          reports: [String.t()]
        }

  @doc "The human Admiralty principal — top clearance, fleet-wide, on duty."
  @spec admiral() :: t()
  def admiral, do: %__MODULE__{kind: :admiral, id: "admiral", rank: :admiral, duty: :active, ship_id: nil}

  @doc """
  Build an agent principal from an ensign's runtime facts. `attrs` is a map/keyword
  with `:agent_id, :rank, :duty, :ship_id, :co, :reports` — sourced from process
  state at dispatch time, never the payload.
  """
  @spec agent(map() | keyword()) :: t()
  def agent(attrs) do
    a = Map.new(attrs)

    %__MODULE__{
      kind: :ensign,
      id: a[:agent_id] || a[:id],
      rank: a[:rank] || :ensign,
      duty: a[:duty] || :active,
      ship_id: a[:ship_id],
      co: a[:co],
      reports: a[:reports] || []
    }
  end
end
