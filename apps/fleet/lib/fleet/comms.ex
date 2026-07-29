defmodule Fleet.Comms do
  @moduledoc """
  The command channel's send-path and the sender-authentication primitives.

  This module is the ONLY place a command message is put on the wire. It stamps
  the real sender (`self()`) into the message at the call site, so the receiver
  can attribute it via the Registry — "the Registry vouches for who a message is
  from; an agent trusts an order because the runtime attributes it, not because
  the payload asserts a rank" (FLEET.md §4.5). `from_pid`/`sender_pid` are never
  accepted as arguments — only ever `self()` — which is what makes attribution
  trustworthy.
  """

  alias Fleet.{Order, Signal}

  # ── Send-path (stamps self()) ─────────────────────────────────────────────

  @doc "Deliver an ORDER to an officer, stamping the real issuer pid."
  def order(to_agent_id, %Order{} = order) do
    GenServer.cast(via(to_agent_id), {:order, %{order | reply_to: self()}, self()})
  end

  @doc "Deliver a SIGNAL to an officer, stamping the real sender pid."
  def signal(to_agent_id, %Signal{} = sig) do
    GenServer.cast(via(to_agent_id), {:signal, %{sig | from_pid: self()}})
  end

  @doc """
  Open a conversational HAIL to an officer — a question, not an order. Stamps the
  real sender pid (for attribution) which is also where the reply is delivered.
  Unlike an ORDER, a HAIL confers no authority and starts no assignment.
  """
  def hail(to_agent_id, question) when is_binary(question) do
    GenServer.cast(via(to_agent_id), {:hail, question, self()})
  end

  @doc """
  Deliver a SIGNAL directly to a pid (used when the recipient is not a
  registered officer — e.g. the Admiral/issuer process). Still stamps `self()`.
  """
  def signal_pid(pid, %Signal{} = sig) when is_pid(pid) do
    send(pid, {:signal, %{sig | from_pid: self()}})
  end

  # ── Sender authentication (Registry vouches) ──────────────────────────────

  @doc """
  Attribute a sender pid to a principal. A registered officer pid resolves to
  `{:officer, agent_id}`; anything else (the Admiral / iex / a facade caller) is
  `:admiral`.
  """
  def attribute(sender_pid) when is_pid(sender_pid) do
    case Registry.keys(Fleet.Registry, sender_pid) do
      [{:officer, agent_id} | _] -> {:officer, agent_id}
      _ -> :admiral
    end
  end

  # ── Bounded topology ──────────────────────────────────────────────────────

  @doc "Is the attributed sender this officer's commanding officer?"
  def from_co?(sender, %{context_tags: %{co: co}}) when not is_nil(co),
    do: sender == {:officer, co}

  def from_co?(_sender, _state), do: false

  @doc "Is the sender the Admiral, and this officer at the top of its chain (co == nil)?"
  def from_admiral_root?(:admiral, %{context_tags: %{co: nil}}), do: true
  def from_admiral_root?(_sender, _state), do: false

  @doc "May this sender issue ORDERs to this officer (its CO, or the Admiral if co == nil)?"
  def authorized_issuer?(sender, state),
    do: from_co?(sender, state) or from_admiral_root?(sender, state)

  @doc "Is the attributed sender one of this officer's direct reports?"
  def from_report?({:officer, id}, %{context_tags: %{reports: reports}}), do: id in reports
  def from_report?(_sender, _state), do: false

  @doc "Render an attributed principal as a string for the audit log."
  def principal_string(:admiral), do: "admiral"
  def principal_string({:officer, id}), do: id
  def principal_string(other), do: inspect(other)

  defp via(agent_id), do: {:via, Registry, {Fleet.Registry, {:officer, agent_id}}}
end
