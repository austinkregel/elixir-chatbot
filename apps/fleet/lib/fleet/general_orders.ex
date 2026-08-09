defmodule Fleet.GeneralOrders do
  @moduledoc """
  The fleet constitution — the top of the normative stack, shared by every crew
  member.

      General Orders   (fleet-wide, this module)
        └─ soul         (this officer's identity, values, hardcoded bounds)
            └─ order    (this task: objective, constraints, grants)

  Lower layers cannot override higher ones. A soul is written to *extend* the
  General Orders, not to restate or contradict them, which is also why the text
  is byte-identical for every officer: it is a shared, cacheable prompt prefix,
  and an article that varies per agent is not a constitution.

  ## Doctrine and enforcement are two different things

  Each article is stated to the agent here *and*, where it can be, enforced in
  code — because a rule that lives only in a prompt is a rule the system does
  not actually have. What backs each article today:

  | # | Article | Enforcement |
  |---|---|---|
  | 1 | Truth | none yet — evidence verification is the trust-ledger work |
  | 2 | Chain of command | `Fleet.Comms.attribute/1` (the Registry vouches, not the payload) and `Fleet.DataFrame` (tool results are escaped and framed as data) |
  | 3 | Scope | `Fleet.Dispatcher.decide/2` against order-conferred grants; `Fleet.Clearance` for reads |
  | 4 | Dissent is duty | `Fleet.Appraisal` and the DISSENT signal path |
  | 5 | Anomalies surface | `Fleet.DataFrame.anomaly?/1` → `:provenance_anomaly` audit |
  | 6 | Memory honesty | provenance-stamped ingest (`Fleet.Officer`), belief confidence in `Brain.Epistemic` |

  Article 1 is the honest gap: nothing yet checks that a report's claims are
  supported by what the agent actually did.

  The text is read from `priv/general_orders.md` at compile time and is a
  `@external_resource`, so editing the constitution recompiles this module. It
  is a leaf — no runtime dependencies — so it can be rendered from anywhere
  without dragging the fleet into a cycle.
  """

  @orders_path Path.join(:code.priv_dir(:fleet) |> to_string(), "general_orders.md")
  @external_resource @orders_path

  @text (case File.read(@orders_path) do
           {:ok, contents} -> String.trim(contents)
           {:error, reason} -> raise "cannot read general_orders.md: #{inspect(reason)}"
         end)

  @doc """
  The constitution as it is rendered into an officer's standing orders.

  Identical for every officer, deliberately.
  """
  @spec text() :: String.t()
  def text, do: @text

  @doc """
  The normative stack as text: the General Orders, then this officer's own
  constitution beneath them.
  """
  @spec normative_stack(Brain.Soul.t() | nil) :: String.t()
  def normative_stack(soul) do
    case soul && Brain.Soul.system_prompt(soul) do
      nil -> @text
      "" -> @text
      constitution -> @text <> "\n\n---\n\n" <> constitution
    end
  end

  @doc """
  Return the soul an officer should actually think under: its own constitution
  with the General Orders seated above it.

  Fleet applies the stack on its side rather than Brain rendering it, because
  `brain` knows nothing about fleets and must not — `fleet` depends on `brain`,
  never the reverse. This is the same seam `Fleet.Officer.soul_with_tool_context/2`
  uses to add the tool appendix.

  An officer with no soul of its own still serves under the constitution, so a
  minimal soul carrying just the General Orders is returned rather than `nil`.
  """
  @spec apply_to(Brain.Soul.t() | nil) :: Brain.Soul.t()
  def apply_to(nil) do
    %Brain.Soul{id: "fleet-general-orders", name: "Fleet", constitution: @text}
  end

  def apply_to(%Brain.Soul{} = soul) do
    %{soul | constitution: normative_stack(soul)}
  end
end
