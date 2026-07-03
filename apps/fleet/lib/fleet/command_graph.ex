defmodule Fleet.CommandGraph do
  @moduledoc """
  Persists the chain of command as a graph: `Agent -[COMMANDS]-> Agent`
  (a commanding officer to a direct report), in the Apache AGE `command_graph`.

  Writes are **error-surfacing** — `Brain.AtlasIntegration.sync/1` returns
  `{:error, reason}` when Atlas is unavailable or the write fails, and callers
  propagate it. There is no graceful degradation; a failed persistence is a real
  failure, not a silent no-op.
  """

  alias Brain.AtlasIntegration

  @graph "command_graph"

  @doc """
  Establish (idempotently) the `co -[COMMANDS]-> report` edge. Returns
  `{:ok, edge} | {:error, reason}`.
  """
  def establish_command(co_id, report_id) when is_binary(co_id) and is_binary(report_id) do
    AtlasIntegration.sync(fn ->
      {:ok, co} = AtlasIntegration.ensure_node(@graph, "Agent", %{name: co_id, agent_id: co_id})
      {:ok, report} = AtlasIntegration.ensure_node(@graph, "Agent", %{name: report_id, agent_id: report_id})

      {:ok, edge} =
        AtlasIntegration.find_or_create_edge(
          @graph,
          co.id,
          report.id,
          Atlas.Graph.EdgeLabels.commands(),
          %{established_at: DateTime.utc_now() |> DateTime.to_iso8601()}
        )

      edge
    end)
  end

  @doc "The graph name (for tests/queries)."
  def graph, do: @graph
end
