defmodule Atlas.GraphBrainIntegrationTest do
  @moduledoc """
  Integration test that verifies Atlas graph + relational data
  works end-to-end with known seeded data.

  Seeds a known knowledge graph and relational tables, then makes
  positive assertions about:

  1. Graph CRUD (nodes/edges created, queryable, correct types)
  2. Graph adjacency extraction (GNN-ready output format)
  3. Graph neighborhood traversal (multi-hop queries)
  4. Relational schema CRUD (beliefs, facts stored and queryable)
  5. Triples export (KG-BERT ready)
  6. Telemetry collector records graph operations
  """

  use Atlas.DataCase, async: false

  alias Atlas.Graph
  alias Atlas.Graph.Types.{Vertex, Edge}
  alias Atlas.Schemas.{Belief, LearnedFact}

  @knowledge_graph "knowledge_graph"
  @user_graph "user_graph"

  describe "knowledge graph seeding and querying" do
    test "creates a domain of cities, countries, and relations and extracts adjacency data" do
      # --- Seed the knowledge graph with a known topology ---
      # Nodes: Paris, France, Europe, Berlin, Germany
      {:ok, paris} = Graph.add_node(@knowledge_graph, "City", %{name: "Paris", population: 2161000})
      {:ok, france} = Graph.add_node(@knowledge_graph, "Country", %{name: "France", continent: "Europe"})
      {:ok, europe} = Graph.add_node(@knowledge_graph, "Continent", %{name: "Europe"})
      {:ok, berlin} = Graph.add_node(@knowledge_graph, "City", %{name: "Berlin", population: 3748148})
      {:ok, germany} = Graph.add_node(@knowledge_graph, "Country", %{name: "Germany", continent: "Europe"})

      # Verify nodes are Vertex structs with properties
      assert %Vertex{label: "City", properties: %{"name" => "Paris"}} = paris
      assert %Vertex{label: "Country", properties: %{"name" => "France"}} = france
      assert %Vertex{label: "Continent", properties: %{"name" => "Europe"}} = europe

      # Edges: Paris -[:capital_of]-> France, Berlin -[:capital_of]-> Germany,
      #        France -[:is_in]-> Europe, Germany -[:is_in]-> Europe
      {:ok, cap_edge} = Graph.add_edge(@knowledge_graph, paris.id, france.id, "capital_of")
      {:ok, _} = Graph.add_edge(@knowledge_graph, berlin.id, germany.id, "capital_of")
      {:ok, _} = Graph.add_edge(@knowledge_graph, france.id, europe.id, "is_in")
      {:ok, _} = Graph.add_edge(@knowledge_graph, germany.id, europe.id, "is_in")

      # Verify edge struct
      assert %Edge{label: "capital_of", start_id: _, end_id: _} = cap_edge

      # --- Positive assertions on graph state ---
      assert {:ok, 5} = Graph.count_nodes(@knowledge_graph)
      assert {:ok, 4} = Graph.count_edges(@knowledge_graph)
      assert {:ok, 2} = Graph.count_nodes(@knowledge_graph, "City")
      assert {:ok, 2} = Graph.count_nodes(@knowledge_graph, "Country")
      assert {:ok, 1} = Graph.count_nodes(@knowledge_graph, "Continent")
      assert {:ok, 2} = Graph.count_edges(@knowledge_graph, "capital_of")
      assert {:ok, 2} = Graph.count_edges(@knowledge_graph, "is_in")

      # --- Adjacency extraction for GNN ---
      {:ok, adjacency} = Graph.to_adjacency(@knowledge_graph)

      assert is_list(adjacency.node_ids)
      assert length(adjacency.node_ids) == 5
      assert is_list(adjacency.edges)
      assert length(adjacency.edges) == 4
      assert is_map(adjacency.features)
      assert map_size(adjacency.features) == 5

      # Each node's features should be a map with a "name" property
      Enum.each(adjacency.features, fn {_id, props} ->
        assert is_map(props)
        assert Map.has_key?(props, "name")
      end)

      # The edge pairs should reference valid node IDs
      all_ids = MapSet.new(adjacency.node_ids)

      Enum.each(adjacency.edges, fn {from, to} ->
        assert MapSet.member?(all_ids, from), "Edge source #{from} not in node_ids"
        assert MapSet.member?(all_ids, to), "Edge target #{to} not in node_ids"
      end)

      # --- Triples export for KG-BERT ---
      {:ok, triples} = Graph.to_triples(@knowledge_graph)
      assert length(triples) == 4

      Enum.each(triples, fn [subj_props, _rel_type, obj_props] ->
        assert is_map(subj_props)
        assert is_map(obj_props)
      end)
    end

    test "neighborhood query returns connected nodes within hops" do
      {:ok, elixir} = Graph.add_node(@knowledge_graph, "Language", %{name: "Elixir"})
      {:ok, beam} = Graph.add_node(@knowledge_graph, "Runtime", %{name: "BEAM"})
      {:ok, erlang} = Graph.add_node(@knowledge_graph, "Language", %{name: "Erlang"})

      {:ok, _} = Graph.add_edge(@knowledge_graph, elixir.id, beam.id, "runs_on")
      {:ok, _} = Graph.add_edge(@knowledge_graph, erlang.id, beam.id, "runs_on")

      # Elixir's 1-hop neighborhood should include BEAM
      {:ok, neighbors} = Graph.neighborhood(@knowledge_graph, elixir.id, 1)
      assert length(neighbors) >= 1

      neighbor_names =
        neighbors
        |> List.flatten()
        |> Enum.filter(&match?(%Vertex{}, &1))
        |> Enum.map(& &1.properties["name"])

      assert "BEAM" in neighbor_names

      # Elixir's 2-hop neighborhood should include BEAM and Erlang
      {:ok, neighbors_2hop} = Graph.neighborhood(@knowledge_graph, elixir.id, 2)

      neighbor_names_2hop =
        neighbors_2hop
        |> List.flatten()
        |> Enum.filter(&match?(%Vertex{}, &1))
        |> Enum.map(& &1.properties["name"])

      assert "BEAM" in neighbor_names_2hop
      assert "Erlang" in neighbor_names_2hop
    end
  end

  describe "user graph for conversational memory" do
    test "models user preferences as graph relationships" do
      {:ok, user} = Graph.add_node(@user_graph, "User", %{name: "Alice", id: "user_alice"})
      {:ok, jazz} = Graph.add_node(@user_graph, "Topic", %{name: "Jazz"})
      {:ok, auto} = Graph.add_node(@user_graph, "Industry", %{name: "Automotive"})
      {:ok, hydro} = Graph.add_node(@user_graph, "Project", %{name: "Hydroponics System"})

      {:ok, likes_edge} = Graph.add_edge(@user_graph, user.id, jazz.id, "likes")
      {:ok, _} = Graph.add_edge(@user_graph, user.id, auto.id, "works_in")
      {:ok, _} = Graph.add_edge(@user_graph, user.id, hydro.id, "built")

      # Verify edge structure
      assert %Edge{label: "likes"} = likes_edge

      # Count user's relationships
      assert {:ok, 3} = Graph.count_edges(@user_graph)

      # User's neighborhood should include all their connected nodes
      {:ok, user_neighbors} = Graph.neighborhood(@user_graph, user.id, 1)

      neighbor_names =
        user_neighbors
        |> List.flatten()
        |> Enum.filter(&match?(%Vertex{}, &1))
        |> Enum.map(& &1.properties["name"])

      assert "Jazz" in neighbor_names
      assert "Automotive" in neighbor_names
      assert "Hydroponics System" in neighbor_names
    end
  end

  describe "relational tables with known data" do
    test "stores and queries beliefs about geographic entities" do
      assert {:ok, _} =
               %Belief{}
               |> Belief.changeset(%{
                 subject: "Paris",
                 predicate: "is_capital_of",
                 object: "France",
                 confidence: 0.95,
                 source: "knowledge_base"
               })
               |> Repo.insert()

      assert {:ok, _} =
               %Belief{}
               |> Belief.changeset(%{
                 subject: "Paris",
                 predicate: "has_population",
                 object: "2.1 million",
                 confidence: 0.9,
                 source: "knowledge_base"
               })
               |> Repo.insert()

      assert {:ok, _} =
               %LearnedFact{}
               |> LearnedFact.changeset(%{
                 id: "fact_paris_capital",
                 entity: "Paris",
                 entity_type: "city",
                 fact: "Paris is the capital city of France",
                 confidence: 0.95
               })
               |> Repo.insert()

      # Verify beliefs are queryable
      paris_beliefs = Belief |> Belief.for_subject("Paris") |> Belief.active() |> Repo.all()
      assert length(paris_beliefs) == 2

      capital_belief =
        Enum.find(paris_beliefs, &(&1.predicate == "is_capital_of"))

      assert capital_belief.object == "France"
      assert capital_belief.confidence == 0.95

      # Verify learned facts are queryable
      paris_facts = LearnedFact |> LearnedFact.for_entity("Paris") |> Repo.all()
      assert length(paris_facts) == 1
      assert hd(paris_facts).fact == "Paris is the capital city of France"

      # Verify confidence filtering works
      high_conf = Belief |> Belief.above_confidence(0.92) |> Repo.all()
      assert Enum.any?(high_conf, &(&1.predicate == "is_capital_of"))
      refute Enum.any?(high_conf, &(&1.predicate == "has_population"))
    end
  end

  describe "telemetry collector records graph operations" do
    test "graph CRUD operations are tracked by the collector" do
      Atlas.Telemetry.attach_handlers()
      Atlas.Stats.Collector.reset()
      Process.sleep(10)

      # Perform some graph operations
      {:ok, _} = Graph.add_node(@knowledge_graph, "TestNode", %{name: "Telemetry Test"})
      {:ok, node_a} = Graph.add_node(@knowledge_graph, "TestNode", %{name: "Node A"})
      {:ok, node_b} = Graph.add_node(@knowledge_graph, "TestNode", %{name: "Node B"})
      {:ok, _} = Graph.add_edge(@knowledge_graph, node_a.id, node_b.id, "test_rel")

      # Give the async casts a moment to process
      Process.sleep(50)

      # Check that the collector recorded operations
      metrics = Atlas.Stats.Collector.get_metrics()

      assert metrics.total_graph_queries > 0,
             "Expected graph queries to be recorded, got #{metrics.total_graph_queries}"

      # Node additions should be tracked
      total_nodes_added =
        metrics.nodes_added
        |> Map.values()
        |> Enum.sum()

      assert total_nodes_added >= 3,
             "Expected at least 3 node additions, got #{total_nodes_added}"

      # Edge additions should be tracked
      total_edges_added =
        metrics.edges_added
        |> Map.values()
        |> Enum.sum()

      assert total_edges_added >= 1,
             "Expected at least 1 edge addition, got #{total_edges_added}"
    end
  end
end
