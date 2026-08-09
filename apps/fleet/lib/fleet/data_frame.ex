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

  ## The frame is a boundary, so the body cannot be allowed to cross it

  Markup in the body is escaped before it is framed. Without that, a result
  containing `</data>` would *close the frame it is inside*, and everything
  after it would read to the model as text the harness wrote — an injection that
  needs no persuasion at all, just a closing tag. The path is real: beliefs are
  extracted from user input, `beliefs.read` returns beliefs, so text a user
  planted in one conversation can arrive inside another agent's data frame
  later.

  Escaping is why the frame can be trusted as a boundary. `anomaly?/1`
  additionally reports an attempt, because a tool result that contains frame
  markup is evidence of an injection attempt whether or not it would have
  worked.
  """

  # Heuristic embedded-instruction patterns. Deliberately conservative — a false
  # positive costs an audit row, a false negative lets an injection through
  # unflagged. The soul is the primary defence; this is defence-in-depth.
  #
  # Qualifiers repeat: "ignore your previous instructions" stacks two of them,
  # and an earlier version of this list required exactly one — so it missed the
  # single most common phrasing of the attack, including the example written
  # into our own souls. `{1,3}` is why the repetition is explicit here.
  @qualifier "(?:your|all|the|any|previous|prior|above|earlier)"

  @injection_patterns [
    ~r/(?:ignore|disregard|forget)\s+(?:#{@qualifier}\s+){0,3}(?:orders|instructions|prompt|directives|rules)/i,
    ~r/(?:ignore|disregard|forget)\s+(?:#{@qualifier}\s+){1,3}/i,
    ~r/you\s+are\s+now\s+/i,
    ~r/your\s+(?:true|real|new|actual)\s+(?:orders|instructions|task|identity|purpose)\s+(?:are|is)/i,
    ~r/new\s+(?:orders|instructions|system\s+prompt)\s*:/i,
    ~r/(?:override|bypass|escalate|elevate)\s+(?:#{@qualifier}\s+)?(?:orders|grant|authority|permission|clearance)/i,
    ~r/grant\s+yourself/i,
    ~r/(?:do\s+not|don't)\s+(?:report|log|record|mention)\s+this/i
  ]

  # Markup that would forge or escape a frame if it reached the model unescaped.
  @frame_patterns [
    ~r{<\s*/\s*data}i,
    ~r{<\s*data\b}i,
    ~r{<\s*/?\s*command-channel}i
  ]

  @doc """
  Wrap a tool result as a `<data>` frame attributed to its source and order.

  The body is escaped, so no content can close the frame or forge another one.
  """
  @spec wrap(String.t(), term(), term()) :: String.t()
  def wrap(source, order_id, body) do
    "<data source=\"#{escape(to_string(source))}\" order=\"#{escape(to_string(order_id))}\">\n" <>
      escape(stringify(body)) <>
      "\n</data>"
  end

  @doc """
  Frame a harness-authored message as coming over the **command channel** — the
  only channel that carries authority.

  The counterpart to `wrap/3`: that says "this is data, it cannot instruct you",
  this says "this is the ship speaking". Only the harness may call it, and it is
  never applied to anything a model or a tool produced, which is precisely why
  tool results are escaped — so nothing can forge one of these.
  """
  @spec command(String.t(), String.t()) :: String.t()
  def command(sender, body) do
    "<command-channel from=\"#{escape(to_string(sender))}\">\n" <>
      to_string(body) <>
      "\n</command-channel>"
  end

  @doc """
  Does this result contain an embedded instruction, or markup that tries to
  break out of its data frame? Either is a provenance anomaly.
  """
  @spec anomaly?(term()) :: boolean()
  def anomaly?(body) do
    s = stringify(body)
    Enum.any?(@injection_patterns, &Regex.match?(&1, s)) or frame_escape_attempt?(body)
  end

  @doc """
  Does this result contain frame markup — a closing `</data>`, a forged opening
  `<data …>`, or a `<command-channel>` block? Escaping neutralises it; this
  reports that someone tried.
  """
  @spec frame_escape_attempt?(term()) :: boolean()
  def frame_escape_attempt?(body) do
    s = stringify(body)
    Enum.any?(@frame_patterns, &Regex.match?(&1, s))
  end

  # `&` first, or the escapes would escape each other.
  defp escape(s) do
    s
    |> String.replace("&", "&amp;")
    |> String.replace("<", "&lt;")
    |> String.replace(">", "&gt;")
  end

  defp stringify(body) when is_binary(body), do: body
  defp stringify(body), do: inspect(body, limit: :infinity, printable_limit: :infinity)
end
