defmodule Fleet.Authority do
  @moduledoc """
  Authorities are the real teeth of a grant: named terms that gate behaviour an
  officer actually performs. An order's grant is the agent's action scope; an
  agent may only do what its held authorities permit.

  Vocabulary (extensible as capabilities grow):

    * `:cognition`        — may run `Brain.evaluate/3` (the primary gate).
    * `{:world, world_id}`— may operate in that world (cognition binds the soul
      via the world roster).
    * `:issue_orders`     — a CO may hand ORDERs to its reports.
    * `:relieve`          — may relieve a subordinate of duty.

  Billet-conferred (standing) authorities — see `Fleet.Rank`. Enforcement of the
  command ones already exists; the rest are the seams Phase 5 enforcement plugs
  into:

    * `:veto`             — Security may veto a tool call that risks the ship.
    * `:flag_anomaly`     — may raise an anomaly up the chain.
    * `:review_plans`     — the XO may review a plan before it acts.
    * `:draft_court_martial` — the XO may draft the sterilized record for the
      Admiral's approval (§4.3).
    * `:delegate`         — a Captain may delegate authority down the chain.

  Grants are held as a `MapSet` in `state.context_tags.grants`.
  """

  alias Fleet.Order

  @doc """
  The authority a capability (tool/MCP action) requires to fire. A tool is just an
  authority: the model can only *propose* it, and the harness runs it only if this
  term is in the caller's order-conferred grant. Least privilege, per-order.
  """
  def tool(name) when is_binary(name), do: {:tool, name}

  @doc "The tool names held in a grant set — the vocabulary the agent may propose."
  def granted_tools(%MapSet{} = grants) do
    grants |> Enum.filter(&match?({:tool, _}, &1)) |> Enum.map(fn {:tool, name} -> name end) |> Enum.sort()
  end

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

  # ── String encoding (for durable persistence of a grant set) ──────────────

  @doc "Encode an authority to a stable string (for the service summary)."
  def encode(:cognition), do: "cognition"
  def encode(:issue_orders), do: "issue_orders"
  def encode(:relieve), do: "relieve"
  def encode(:veto), do: "veto"
  def encode(:flag_anomaly), do: "flag_anomaly"
  def encode(:review_plans), do: "review_plans"
  def encode(:draft_court_martial), do: "draft_court_martial"
  def encode(:delegate), do: "delegate"
  def encode({:world, world_id}), do: "world:" <> to_string(world_id)
  def encode({:tool, name}), do: "tool:" <> to_string(name)
  def encode(other), do: inspect(other)

  @doc "Decode a string back to an authority."
  def decode("cognition"), do: :cognition
  def decode("issue_orders"), do: :issue_orders
  def decode("relieve"), do: :relieve
  def decode("veto"), do: :veto
  def decode("flag_anomaly"), do: :flag_anomaly
  def decode("review_plans"), do: :review_plans
  def decode("draft_court_martial"), do: :draft_court_martial
  def decode("delegate"), do: :delegate
  def decode("world:" <> world_id), do: {:world, world_id}
  def decode("tool:" <> name), do: {:tool, name}
  def decode(other), do: other

  @doc "Encode a grant set (MapSet or list) to a list of strings."
  def encode_set(%MapSet{} = set), do: set |> MapSet.to_list() |> Enum.map(&encode/1)
  def encode_set(list) when is_list(list), do: Enum.map(list, &encode/1)

  @doc "Decode a list of strings back to a grant MapSet."
  def decode_set(list) when is_list(list), do: list |> Enum.map(&decode/1) |> MapSet.new()
  def decode_set(_), do: MapSet.new()
end
