defmodule Atlas.GraphWorldTest do
  @moduledoc """
  Each world has its own graph of each per-world kind; the expression-level
  graphs are shared. Every graph a test creates, renames or writes into is
  inside the test's transaction and is gone with it.
  """

  use Atlas.DataCase, async: false

  alias Atlas.Graph
  alias Atlas.Graph.Types.Vertex

  defp test_world(name), do: "test_#{name}_#{System.unique_integer([:positive])}"

  defp names(graph, label) do
    {:ok, rows} = Graph.cypher(graph, "MATCH (n:#{label}) RETURN n")
    rows |> List.flatten() |> Enum.map(fn %Vertex{properties: props} -> props["name"] end) |> Enum.sort()
  end

  # Write into a graph by its name without going through Atlas.Graph, which
  # refuses an unscoped per-world kind.
  defp raw_create(graph, label, name) do
    Repo.query!(
      "SELECT * FROM ag_catalog.cypher('#{graph}', $$ CREATE (n:#{label} {name: '#{name}'}) RETURN n $$) AS (n ag_catalog.agtype)",
      []
    )
  end

  describe "world_graph/2" do
    test "names a world's graph after its kind and the world id, unchanged" do
      assert Graph.world_graph("knowledge_graph", "default") == "knowledge_graph__default"
      assert Graph.world_graph("epistemic_graph", "rho24-hl1Dg") == "epistemic_graph__rho24-hl1Dg"
      assert Graph.world_graph("semantic_graph", "-x_Y") == "semantic_graph__-x_Y"

      assert Graph.world_graphs("default") == [
               "knowledge_graph__default",
               "user_graph__default",
               "semantic_graph__default",
               "conversation_graph__default",
               "epistemic_graph__default"
             ]
    end

    test "raises without a world" do
      assert_raise ArgumentError, ~r/needs the world/, fn -> Graph.world_graph("knowledge_graph", nil) end
      assert_raise ArgumentError, ~r/needs a world id/, fn -> Graph.world_graph("knowledge_graph", "") end
      assert_raise ArgumentError, ~r/needs a world id/, fn -> Graph.world_graph("knowledge_graph", :default) end
    end

    test "raises on a world id that cannot be part of an AGE graph name" do
      for bad <- ["has space", "dot.ted", "quote'", "ends-", "ünï", "semi;colon"] do
        assert_raise ArgumentError, ~r/cannot name a graph/, fn -> Graph.world_graph("user_graph", bad) end
      end

      longest = String.duplicate("a", 63 - byte_size("conversation_graph__"))
      assert Graph.world_graph("conversation_graph", longest) |> byte_size() == 63

      assert_raise ArgumentError, ~r/too long/, fn ->
        Graph.world_graph("conversation_graph", longest <> "a")
      end
    end

    test "raises for a shared graph and for an unknown kind" do
      assert_raise ArgumentError, ~r/shared by every world/, fn -> Graph.world_graph("pos_graph", "default") end
      assert_raise ArgumentError, ~r/not a graph kind/, fn -> Graph.world_graph("memory_graph", "default") end
    end
  end

  describe "per-world graphs" do
    test "a per-world kind's bare name is refused" do
      for kind <- Graph.world_graph_kinds() do
        assert_raise ArgumentError, ~r/is per world/, fn -> Graph.cypher(kind, "MATCH (n) RETURN n") end
        assert_raise ArgumentError, ~r/is per world/, fn -> Graph.add_node(kind, "Thing", %{name: "x"}) end
      end
    end

    test "ensure_world_graphs!/1 creates every graph of the world, and is idempotent" do
      world = test_world("ensure")
      refute Enum.any?(Graph.world_graphs(world), &Graph.graph_exists?/1)

      assert Graph.ensure_world_graphs!(world) == Graph.world_graphs(world)
      assert Enum.all?(Graph.world_graphs(world), &Graph.graph_exists?/1)

      assert Graph.ensure_world_graphs!(world) == Graph.world_graphs(world)
    end

    test "data written in one world's graph is invisible in another's" do
      world_a = test_world("graph_a")
      world_b = test_world("graph_b")
      Graph.ensure_world_graphs!(world_a)
      Graph.ensure_world_graphs!(world_b)

      for kind <- Graph.world_graph_kinds() do
        {:ok, _} = Graph.add_node(Graph.world_graph(kind, world_a), "Thing", %{name: "only_in_a"})

        assert names(Graph.world_graph(kind, world_a), "Thing") == ["only_in_a"]
        assert names(Graph.world_graph(kind, world_b), "Thing") == []
        assert {:ok, 0} = Graph.count_nodes(Graph.world_graph(kind, world_b))
      end
    end

    test "world ids that differ only in case are different worlds with different graphs" do
      upper = "test_Case_#{System.unique_integer([:positive])}"
      lower = String.downcase(upper)
      Graph.ensure_world_graphs!(upper)
      Graph.ensure_world_graphs!(lower)

      {:ok, _} = Graph.add_node(Graph.world_graph("knowledge_graph", upper), "Thing", %{name: "upper"})

      assert names(Graph.world_graph("knowledge_graph", upper), "Thing") == ["upper"]
      assert names(Graph.world_graph("knowledge_graph", lower), "Thing") == []
    end

    test "the shared pos_graph is the same graph for every world" do
      assert Graph.shared_graphs() == ["pos_graph"]
      assert Graph.graph_exists?("pos_graph")

      name = "shared_tag_#{System.unique_integer([:positive])}"
      {:ok, _} = Graph.add_node("pos_graph", "POSTag", %{name: name})

      assert name in names("pos_graph", "POSTag")
    end
  end

  describe "drop_world_graphs!/1" do
    test "drops every graph of the world with its data, and leaves another world's" do
      world = test_world("drop")
      other = test_world("drop_other")
      Graph.ensure_world_graphs!(world)
      Graph.ensure_world_graphs!(other)

      {:ok, _} = Graph.add_node(Graph.world_graph("knowledge_graph", world), "Thing", %{name: "dropped"})
      {:ok, _} = Graph.add_node(Graph.world_graph("knowledge_graph", other), "Thing", %{name: "kept"})

      assert Graph.drop_world_graphs!(world) == Graph.world_graphs(world)

      refute Enum.any?(Graph.world_graphs(world), &Graph.graph_exists?/1)
      assert Enum.all?(Graph.world_graphs(other), &Graph.graph_exists?/1)
      assert names(Graph.world_graph("knowledge_graph", other), "Thing") == ["kept"]
    end

    test "drops only the graphs the world has, and is idempotent" do
      world = test_world("drop_partial")
      Repo.query!("SELECT ag_catalog.create_graph($1)", [Graph.world_graph("user_graph", world)])

      assert Graph.drop_world_graphs!(world) == [Graph.world_graph("user_graph", world)]
      assert Graph.drop_world_graphs!(world) == []
    end

    test "raises without a world" do
      assert_raise ArgumentError, ~r/needs the world/, fn -> Graph.drop_world_graphs!(nil) end
    end
  end

  describe "refuse_unscoped_graphs!/0" do
    test "is :ok while no unscoped graph exists" do
      assert Graph.refuse_unscoped_graphs!() == :ok
    end

    test "raises naming the unscoped graphs and the task that moves them" do
      Repo.query!("SELECT ag_catalog.create_graph('epistemic_graph')", [])

      assert_raise ArgumentError, ~r/unscoped graphs \["epistemic_graph"\].*mix atlas.move_graphs_to_world/, fn ->
        Graph.refuse_unscoped_graphs!()
      end
    end
  end

  describe "move_unscoped_graphs_to_world!/1" do
    test "renames an unscoped graph to the world's graph of its kind, keeping its data" do
      world = test_world("move")
      Repo.query!("SELECT ag_catalog.create_graph('semantic_graph')", [])
      raw_create("semantic_graph", "SemanticFact", "kept_fact")

      result = Graph.move_unscoped_graphs_to_world!(world)

      assert {"semantic_graph", :moved} in result

      for kind <- Graph.world_graph_kinds() -- ["semantic_graph"] do
        assert {kind, :no_unscoped_graph} in result
      end

      refute Graph.graph_exists?("semantic_graph")
      assert names(Graph.world_graph("semantic_graph", world), "SemanticFact") == ["kept_fact"]
    end

    test "while an unscoped graph exists, a world's graphs are not created beside it" do
      Repo.query!("SELECT ag_catalog.create_graph('user_graph')", [])

      assert_raise ArgumentError, ~r/unscoped graphs \["user_graph"\]/, fn ->
        Graph.ensure_world_graphs!(test_world("blocked"))
      end
    end

    test "refuses, renaming nothing, when the world already has a graph of the kind" do
      world = test_world("conflict")
      Graph.ensure_world_graphs!(world)
      {:ok, _} = Graph.add_node(Graph.world_graph("user_graph", world), "User", %{name: "existing"})

      Repo.query!("SELECT ag_catalog.create_graph('user_graph')", [])
      Repo.query!("SELECT ag_catalog.create_graph('knowledge_graph')", [])
      raw_create("user_graph", "User", "unscoped")

      message =
        "cannot move unscoped graphs into world #{inspect(world)}: the world already has " <>
          "knowledge_graph__#{world} (and knowledge_graph still exists), " <>
          "user_graph__#{world} (and user_graph still exists)"

      assert_raise ArgumentError, message, fn -> Graph.move_unscoped_graphs_to_world!(world) end

      assert Graph.graph_exists?("user_graph")
      assert Graph.graph_exists?("knowledge_graph")
      assert names(Graph.world_graph("user_graph", world), "User") == ["existing"]
    end
  end
end
