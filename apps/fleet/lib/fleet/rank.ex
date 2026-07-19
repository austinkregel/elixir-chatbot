defmodule Fleet.Rank do
  @moduledoc """
  A **billet** — the crew posting Command commissions an agent into, and the
  source of its *standing* authorities: what the agent may do by virtue of who
  the Admiral commissioned it as, independent of any single order.

  This is the concrete implementation of FLEET §3.5 — "an Ensign, a Security
  officer, a First Officer, a Captain differ only in *data* — rank, role, and the
  special **authorities** the rank confers." Authority is **conferred at
  commission by Command and derived here** — never declared by the soul (a soul is
  identity only, §3.3) and never self-asserted in a message (§4.5). The billet is
  held in `context_tags.rank`, persisted in the durable service summary, and its
  authorities are re-derived into the standing grant on rehydrate.

  Workers (`:ensign`, `:lieutenant`) hold **no** standing command authorities —
  cognition is conferred per-order, exactly as before. Command billets add the
  authorities their office carries on top of that.

  The new authorities named here (`:veto`, `:draft_court_martial`, …) are defined
  so Phase 5 enforcement plugs straight in; `:issue_orders` / `:relieve` are
  already enforced today, so an XO or Captain commissioned here can command its
  reports immediately.
  """

  # Ordered by seniority so a UI selector reads top-down. Each billet carries a
  # display label, a one-line description, and the standing authorities it confers.
  @billets [
    ensign: %{
      label: "Ensign",
      description: "Junior officer. Executes orders; cognition is conferred per order.",
      authorities: []
    },
    lieutenant: %{
      label: "Lieutenant",
      description: "Senior worker (analysis, operations). Executes orders; no command authority.",
      authorities: []
    },
    security: %{
      label: "Security Officer",
      description:
        "Hair-trigger watch. May veto a tool call that risks the ship and flag an anomaly up the chain.",
      authorities: [:veto, :flag_anomaly]
    },
    executive_officer: %{
      label: "First Officer (XO)",
      description:
        "Second in command. Commands the crew, reviews plans, and drafts a court martial for the Admiral's approval.",
      authorities: [:issue_orders, :relieve, :review_plans, :draft_court_martial]
    },
    captain: %{
      label: "Captain",
      description: "Commands and delegates down the chain.",
      authorities: [:issue_orders, :relieve, :delegate]
    }
  ]

  @default :ensign

  @doc "The default billet for a freshly commissioned agent."
  def default, do: @default

  @doc "All billets as an ordered `{key, meta}` keyword list (for a UI selector)."
  def all, do: @billets

  # Seniority order (the @billets order), most-junior first. `:admiral` sits above
  # every billet; an unknown billet sorts below every billet.
  @order @billets |> Enum.map(fn {k, _} -> k end)

  @doc """
  Seniority index of a billet — higher is more senior. Used by `Fleet.Clearance` for
  the billet-floor rule. `:admiral` is above all billets; an unknown billet is below.
  """
  @spec seniority(atom()) :: integer()
  def seniority(:admiral), do: length(@order)
  def seniority(billet) when is_atom(billet), do: Enum.find_index(@order, &(&1 == billet)) || -1
  def seniority(_), do: -1

  @doc "Is `billet` at least as senior as `floor`?"
  @spec at_least?(atom(), atom()) :: boolean()
  def at_least?(billet, floor), do: seniority(billet) >= seniority(floor)

  @doc "Is `rank` a known billet key?"
  def known?(rank) when is_atom(rank), do: Keyword.has_key?(@billets, rank)
  def known?(_), do: false

  @doc """
  The standing authorities a billet confers, as a plain list. Unknown billets
  confer nothing (`[]`) — an agent never gains authority from an unrecognised
  rank. Accepts an atom key or its string form (e.g. from a form submission).
  """
  def standing_authorities(rank) when is_atom(rank) do
    case Keyword.get(@billets, rank) do
      %{authorities: a} -> a
      _ -> []
    end
  end

  def standing_authorities(rank) when is_binary(rank), do: standing_authorities(to_key(rank))
  def standing_authorities(_), do: []

  @doc "Human-readable label for a billet (falls back to the capitalised key)."
  def label(rank) when is_atom(rank) do
    case Keyword.get(@billets, rank) do
      %{label: l} -> l
      _ -> rank |> to_string() |> String.capitalize()
    end
  end

  def label(rank) when is_binary(rank), do: label(to_key(rank))
  def label(_), do: "Unknown"

  @doc "One-line description of a billet (empty string when unknown)."
  def describe(rank) when is_atom(rank) do
    case Keyword.get(@billets, rank) do
      %{description: d} -> d
      _ -> ""
    end
  end

  def describe(rank) when is_binary(rank), do: describe(to_key(rank))
  def describe(_), do: ""

  @doc """
  Parse a billet key from a string (a form value). Returns the matching known
  key, or the default billet when the string names no known billet — the UI can
  never mint an authority-bearing rank from arbitrary input.
  """
  def to_key(key) when is_atom(key), do: key

  def to_key(str) when is_binary(str) do
    Enum.find_value(@billets, @default, fn {k, _} -> if to_string(k) == str, do: k end)
  end
end
