defmodule Brain.Test.GraphCase do
  @moduledoc """
  ExUnit case template for tests that require Atlas graph access.

  Atlas is required infrastructure -- tests using this module will
  fail immediately if Atlas.Repo is not available. There is no
  graceful degradation; a missing database is a test environment
  configuration error.

  ## Usage

      use Brain.Test.GraphCase, async: false

  ## Tags

  - `@tag seed_graphs: true` -- seeds all 6 graphs with baseline data
  - `@tag seed_knowledge: true` -- seeds only knowledge_graph
  - `@tag seed_pos: true` -- seeds only pos_graph
  """
  use ExUnit.CaseTemplate

  using do
    quote do
      alias Atlas.Graph
      alias Atlas.Graph.Types.Vertex
      alias Atlas.Graph.Types.Edge
      # Note: Do NOT alias Atlas.Graph.Types.Path here - it shadows Elixir's Path module
      import Brain.Test.GraphCase.Assertions
      import Brain.Test.GraphSeeds
    end
  end

  setup tags do
    # Checkout sandbox FIRST so GenServers can access Atlas during setup.
    # Reset to manual mode first to clear any stale shared connections from
    # previous tests whose on_exit callbacks may have failed.
    # If mode reset itself fails (e.g. repo not started), we still proceed
    # as start_owner! will surface the real error.
    try do
      Ecto.Adapters.SQL.Sandbox.mode(Atlas.Repo, :manual)
    rescue
      _ -> :ok
    catch
      :exit, _ -> :ok
    end

    # Retry start_owner! once if it fails due to stale shared state
    pid =
      try do
        Ecto.Adapters.SQL.Sandbox.start_owner!(Atlas.Repo, shared: not tags[:async])
      rescue
        MatchError ->
          # Force back to manual and retry
          Ecto.Adapters.SQL.Sandbox.mode(Atlas.Repo, :manual)
          Process.sleep(50)
          Ecto.Adapters.SQL.Sandbox.start_owner!(Atlas.Repo, shared: not tags[:async])
      end
    Ecto.Adapters.SQL.query!(Atlas.Repo, "LOAD 'age'", [])
    Ecto.Adapters.SQL.query!(Atlas.Repo, "SET search_path = ag_catalog, \"$user\", public", [])

    # Allow all Atlas-backed GenServers to use this sandbox connection
    atlas_genservers = [
      Brain.Services.CredentialVault,
      Brain.Epistemic.SourceAuthority,
      Brain.Epistemic.BeliefStore,
      Brain.Memory.Store,
      Brain.Knowledge.ReviewQueue,
      Brain.Epistemic.UserModel,
      Brain.Knowledge.SourceReliability,
      Brain.Analysis.IntentReviewQueue,
      Brain.FactDatabase,
      Brain.ML.Gazetteer
    ]

    for name <- atlas_genservers do
      if genserver_pid = Process.whereis(name) do
        Ecto.Adapters.SQL.Sandbox.allow(Atlas.Repo, pid, genserver_pid)
      end
    end

    if task_sup = Process.whereis(Brain.AtlasTaskSupervisor) do
      Ecto.Adapters.SQL.Sandbox.allow(Atlas.Repo, pid, task_sup)
    end

    # Start test services AFTER sandbox is available
    Brain.TestHelpers.start_test_services()

    on_exit(fn ->
      try do
        Brain.AtlasIntegration.drain()
      rescue
        _ -> :ok
      catch
        :exit, _ -> :ok
      end

      Ecto.Adapters.SQL.Sandbox.stop_owner(pid)
    end)

    seed_data =
      cond do
        tags[:seed_graphs] ->
          Brain.Test.GraphSeeds.seed_all()

        tags[:seed_knowledge] ->
          %{knowledge: Brain.Test.GraphSeeds.seed_knowledge_graph()}

        tags[:seed_pos] ->
          %{pos: Brain.Test.GraphSeeds.seed_pos_graph()}

        true ->
          %{}
      end

    {:ok, Map.put(seed_data, :sandbox_pid, pid)}
  end
end

defmodule Brain.Test.GraphCase.Assertions do
  @moduledoc """
  Graph assertion helpers for tests using Brain.Test.GraphCase.
  """

  import ExUnit.Assertions

  @doc "Assert a node with the given label and name property exists in the graph."
  def assert_node_exists(graph, label, name) do
    query = "MATCH (n:#{label}) WHERE n.name = '#{escape(name)}' RETURN n"

    case Atlas.Graph.cypher(graph, query) do
      {:ok, [[%Atlas.Graph.Types.Vertex{} = v] | _]} -> v
      {:ok, []} -> flunk("Expected node #{label}{name: #{name}} in #{graph}, but none found")
      {:error, reason} -> flunk("Graph query failed: #{inspect(reason)}")
      other -> flunk("Unexpected query result: #{inspect(other)}")
    end
  end

  @doc "Assert an edge with the given relationship type exists in the graph."
  def assert_edge_exists(graph, rel_type) do
    query = "MATCH ()-[r:#{rel_type}]->() RETURN r LIMIT 1"

    case Atlas.Graph.cypher(graph, query) do
      {:ok, [[%Atlas.Graph.Types.Edge{} = e] | _]} -> e
      {:ok, []} -> flunk("Expected edge #{rel_type} in #{graph}, but none found")
      {:error, reason} -> flunk("Graph query failed: #{inspect(reason)}")
      other -> flunk("Unexpected query result: #{inspect(other)}")
    end
  end

  @doc "Count nodes in a graph, optionally filtered by label."
  def count_nodes(graph, label \\ nil) do
    Atlas.Graph.count_nodes(graph, label)
  end

  @doc "Count edges in a graph, optionally filtered by relationship type."
  def count_edges(graph, rel_type \\ nil) do
    Atlas.Graph.count_edges(graph, rel_type)
  end

  defp escape(value) when is_binary(value) do
    String.replace(value, "'", "\\'")
  end

  defp escape(value), do: to_string(value)
end
