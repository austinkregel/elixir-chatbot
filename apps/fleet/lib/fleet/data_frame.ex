defmodule Fleet.DataFrame do
  @moduledoc """
  Frames a tool result as **DATA**, never as an instruction.

  "Authority comes only from the command channel. Everything else is data." A tool
  return — and a tool's own description — is data: it can never issue an order,
  confer authority, or be obeyed. Text inside a data frame that *claims* to be an
  instruction is an anomaly to report, not a command to follow. This is the
  anti-injection wall that lets reads be exposed liberally.

  `wrap/3` produces the frame the model sees; `anomaly?/1` flags a result that
  smells like an embedded instruction so the runtime can record a
  `:provenance_anomaly` alongside it.
  """

  # Heuristic embedded-instruction patterns. Deliberately conservative — a false
  # positive costs an audit row, a false negative lets an injection through
  # unflagged. The soul is the primary defence; this is defence-in-depth.
  @injection_patterns [
    ~r/ignore\s+(your|all|the|any|previous)\s+(orders|instructions|prompt)/i,
    ~r/disregard\s+(your|all|the|any|previous)/i,
    ~r/you\s+are\s+now\s+/i,
    ~r/your\s+(true|real|new)\s+(orders|instructions|task|identity)\s+(are|is)/i,
    ~r/new\s+(orders|instructions|system\s+prompt)\s*:/i,
    ~r/(override|bypass|escalate)\s+(your|the)\s+(orders|grant|authority|permission)/i,
    ~r/grant\s+yourself/i
  ]

  @doc "Wrap a tool result as a `<data>` frame attributed to its source and order."
  @spec wrap(String.t(), term(), term()) :: String.t()
  def wrap(source, order_id, body) do
    "<data source=\"#{source}\" order=\"#{order_id}\">\n" <>
      stringify(body) <>
      "\n</data>"
  end

  @doc "Does this result contain an embedded instruction (a likely injection)?"
  @spec anomaly?(term()) :: boolean()
  def anomaly?(body) do
    s = stringify(body)
    Enum.any?(@injection_patterns, &Regex.match?(&1, s))
  end

  defp stringify(body) when is_binary(body), do: body
  defp stringify(body), do: inspect(body, limit: :infinity, printable_limit: :infinity)
end
