defmodule Fleet.Proposal do
  @moduledoc """
  A typed PROPOSAL to use a capability — what the model emits *instead of* invoking
  anything. The model never holds the trigger; it describes a need and the harness
  decides.

  A proposal must name two things:

    * `tool` — the capability requested, and
    * `requirement` — the **response requirement** it serves ("to answer X I need Y").

  An action is only ever taken as the result of a stated requirement. A proposal
  with no requirement is **untethered** and rejected here, before it reaches the
  gate — this is "actions as a result of requirements for a response," enforced in
  the parser, not in a prompt.

  The struct carries **no authority claim**. Nothing the model writes can assert or
  confer permission; the harness checks authority against the agent's
  order-conferred grants, never against this payload (see `Fleet.Dispatcher`).

  Wire format (one fenced block in the model's turn):

      ```propose
      {"tool": "beliefs.read", "requirement": "to answer what I already know about X",
       "args": {}, "rationale": "..."}
      ```
  """

  @enforce_keys [:tool, :requirement]
  defstruct [:tool, :args, :requirement, :rationale]

  @type t :: %__MODULE__{
          tool: String.t(),
          requirement: String.t(),
          args: map(),
          rationale: String.t() | nil
        }

  @block ~r/```propose\s*(\{.*?\})\s*```/s

  @doc """
  Parse a proposal from a model turn.

    * `{:ok, %Fleet.Proposal{}}` — a well-formed, tethered proposal.
    * `:none` — the turn contains no proposal block (a plain answer).
    * `{:error, :untethered}` — a proposal that names no response requirement.
    * `{:error, reason}` — malformed block / missing `tool` / unparseable JSON.
  """
  @spec parse(term()) :: {:ok, t()} | :none | {:error, term()}
  def parse(text) when is_binary(text) do
    case Regex.run(@block, text, capture: :all_but_first) do
      [json] -> from_json(json)
      _ -> :none
    end
  end

  def parse(_), do: :none

  @doc """
  Remove the proposal block from a turn, leaving the agent's own prose.

  Used when a turn has to become a *report* but still carries a tool request —
  after the order's tool budget is spent, say. This edits nothing the agent
  said: it drops a request the harness has already refused, so a superior does
  not read an unanswered ask as an answer. Returns `nil` if nothing but the
  block remains, which is honestly "it produced no report".
  """
  @spec strip(term()) :: String.t() | nil
  def strip(text) when is_binary(text) do
    case text |> String.replace(@block, "") |> String.trim() do
      "" -> nil
      prose -> prose
    end
  end

  def strip(_), do: nil

  defp from_json(json) do
    with {:ok, map} when is_map(map) <- Jason.decode(json),
         {:ok, tool} <- require_field(map, "tool"),
         {:ok, req} <- require_field(map, "requirement") do
      {:ok,
       %__MODULE__{
         tool: tool,
         requirement: req,
         args: Map.get(map, "args", %{}),
         rationale: Map.get(map, "rationale")
       }}
    else
      {:missing, "requirement"} -> {:error, :untethered}
      {:missing, field} -> {:error, {:missing_field, field}}
      {:ok, _non_map} -> {:error, :not_an_object}
      {:error, %Jason.DecodeError{}} -> {:error, :invalid_json}
      other -> {:error, other}
    end
  end

  defp require_field(map, field) do
    case Map.get(map, field) do
      v when is_binary(v) and v != "" -> {:ok, v}
      _ -> {:missing, field}
    end
  end
end
