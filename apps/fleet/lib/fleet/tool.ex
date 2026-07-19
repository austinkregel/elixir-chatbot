defmodule Fleet.Tool do
  @moduledoc """
  A registered capability the *harness* can execute on an agent's behalf — and the
  code-owned registry of them.

  A tool is **not** something the model can invoke. It is an authority-gated action
  the harness dispatches only when an approved `Fleet.Proposal` names it. Three
  invariants make this structural, not behavioural:

    * **The registry is code-owned.** A human registers tools here; the model can
      never add one, and a name not in the registry is **unrunnable** (default-deny).
    * **Effect tier lives in the data.** Each tool declares `:read | :mutate |
      :irreversible`, so the gate reasons about blast radius from the spec, not from
      anything the model says.
    * **Firing requires an authority.** `required_authority` must be in the caller's
      order-conferred grant (`Fleet.Authority.tool/1`) — checked by the harness
      against the agent's real grants, never against the proposal payload.

  This first slice registers exactly one tool — `beliefs.read` (effect `:read`) —
  to prove the propose-not-dispatch loop end-to-end with zero blast radius.
  """

  alias Brain.Epistemic.BeliefStore

  @enforce_keys [:name, :effect, :required_authority, :handler]
  defstruct [:name, :effect, :required_authority, :handler, :description, :info_class]

  @type effect :: :read | :mutate | :irreversible
  @type ctx :: %{optional(any) => any}
  @type t :: %__MODULE__{
          name: String.t(),
          effect: effect(),
          required_authority: term(),
          handler: (map(), ctx() -> {:ok, term()} | {:error, term()}),
          description: String.t() | nil,
          info_class: atom() | nil
        }

  @doc """
  The code-owned registry (name => spec). Human-registered; never model-registered.

  A `:read`-effect tool that declares an `:info_class` is gated by `Fleet.Clearance`
  in the Dispatcher (in addition to the action-grant): `beliefs.read` reads the
  caller's own mind (`:agent_mind`, self-read); `systems.read` reads ship health
  (`:system_status`, ship-commissioned).
  """
  @spec registry() :: %{optional(String.t()) => t()}
  def registry do
    %{
      "beliefs.read" => %__MODULE__{
        name: "beliefs.read",
        effect: :read,
        info_class: :agent_mind,
        required_authority: Fleet.Authority.tool("beliefs.read"),
        description: "Read the calling agent's own beliefs (its mind-world belief store). Read-only.",
        handler: &beliefs_read/2
      },
      "systems.read" => %__MODULE__{
        name: "systems.read",
        effect: :read,
        info_class: :system_status,
        required_authority: Fleet.Authority.tool("systems.read"),
        description:
          "Read the ship's systems health — the layered inventory of services, " <>
            "processes, and subsystems. Read-only; ground truth independent of any LLM.",
        handler: &systems_read/2
      }
    }
  end

  @doc "Look up a tool by name. `:error` (default-deny) for anything unregistered."
  @spec lookup(term()) :: {:ok, t()} | :error
  def lookup(name) when is_binary(name), do: Map.fetch(registry(), name)
  def lookup(_), do: :error

  # ── Handlers: (args, ctx) -> {:ok, data} | {:error, reason} ────────────────
  # A handler NEVER decides whether it may run — that is settled by the gate before
  # it is ever called. It only produces data, which the dispatcher frames as DATA.

  defp beliefs_read(_args, %{world_id: world_id}) when is_binary(world_id) do
    cond do
      is_nil(Process.whereis(BeliefStore)) ->
        {:error, :belief_store_unavailable}

      true ->
        case BeliefStore.query_beliefs(world_id: world_id) do
          {:ok, beliefs} ->
            {:ok,
             Enum.map(beliefs, fn b ->
               b |> Map.from_struct() |> Map.take([:subject, :predicate, :object, :confidence, :world_id])
             end)}

          {:error, reason} ->
            {:error, reason}
        end
    end
  end

  defp beliefs_read(_args, _ctx), do: {:error, :no_world}

  # systems.read — the ship's health, as ground truth. Returns a compact summary
  # (counts + rolled-up health + any non-nominal live systems) so the framed <data>
  # stays legible; the full inventory is the human board's job.
  defp systems_read(_args, _ctx) do
    snap = Fleet.Systems.snapshot()

    not_nominal =
      (snap.services ++ snap.processes)
      |> Enum.reject(&(&1.status == :up))
      |> Enum.map(&%{system: &1.name, layer: &1.layer, deck: &1.deck, status: &1.status})

    {:ok,
     %{
       ship_id: snap.ship_id,
       health: Map.take(snap.health, [:health_score, :health_status, :services_up, :services_total, :genservers_running, :genservers_total]),
       counts: snap.counts,
       not_nominal: not_nominal
     }}
  end
end
