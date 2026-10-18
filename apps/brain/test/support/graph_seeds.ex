defmodule Brain.Test.GraphSeeds do
  @moduledoc """
  Seed data for graph tests. Creates a known baseline state
  that tests can reason about. All data is created within
  the Ecto sandbox and rolled back after each test.

  Graphs other than the shared `pos_graph` belong to a world, so their seed
  functions take the world whose graphs they fill (the world's graphs must
  exist; `Brain.Test.GraphCase` creates them).

  Each seed function returns a map of created nodes for use
  in test assertions.
  """
  alias Atlas.Graph
  alias Atlas.Graph.EdgeLabels

  @doc "Seed the world's five graphs and pos_graph, and return a combined map of created nodes."
  def seed_all(world_id) do
    %{
      knowledge: seed_knowledge_graph(world_id),
      user: seed_user_graph(world_id),
      semantic: seed_semantic_graph(world_id),
      conversation: seed_conversation_graph(world_id),
      epistemic: seed_epistemic_graph(world_id),
      pos: seed_pos_graph()
    }
  end

  @doc "Seed the world's knowledge_graph with geographic and domain entities."
  def seed_knowledge_graph(world_id) do
    g = Graph.world_graph("knowledge_graph", world_id)

    {:ok, paris} = Graph.add_node(g, "Location", %{name: "Paris", type: "city"})
    {:ok, france} = Graph.add_node(g, "Location", %{name: "France", type: "country"})
    {:ok, _} = Graph.add_edge(g, paris.id, france.id, "capital_of")
    {:ok, _} = Graph.add_edge(g, paris.id, france.id, "located_in")

    {:ok, london} = Graph.add_node(g, "Location", %{name: "London", type: "city"})
    {:ok, uk} = Graph.add_node(g, "Location", %{name: "United Kingdom", type: "country"})
    {:ok, _} = Graph.add_edge(g, london.id, uk.id, "capital_of")
    {:ok, _} = Graph.add_edge(g, london.id, uk.id, "located_in")

    {:ok, elixir} = Graph.add_node(g, "Language", %{name: "Elixir"})
    {:ok, beam} = Graph.add_node(g, "Runtime", %{name: "BEAM"})
    {:ok, _} = Graph.add_edge(g, elixir.id, beam.id, "runs_on")

    %{paris: paris, france: france, london: london, uk: uk, elixir: elixir, beam: beam}
  end

  @doc "Seed the world's user_graph with test user preferences."
  def seed_user_graph(world_id) do
    g = Graph.world_graph("user_graph", world_id)

    {:ok, user} = Graph.add_node(g, "User", %{name: "TestUser", id: "test_user_1"})
    {:ok, jazz} = Graph.add_node(g, "Topic", %{name: "jazz"})
    {:ok, weather} = Graph.add_node(g, "Topic", %{name: "weather"})
    {:ok, _} = Graph.add_edge(g, user.id, jazz.id, EdgeLabels.likes(), %{confidence: 0.9})
    {:ok, _} = Graph.add_edge(g, user.id, weather.id, EdgeLabels.asked_about(), %{count: 3})

    %{user: user, jazz: jazz, weather: weather}
  end

  @doc "Seed the world's semantic_graph with a semantic fact and episode evidence."
  def seed_semantic_graph(world_id) do
    g = Graph.world_graph("semantic_graph", world_id)

    {:ok, fact} = Graph.add_node(g, "SemanticFact", %{
      name: "weather_pattern",
      representation: "Action: query | Examples: weather in London, what is weather",
      evidence_count: 3
    })
    {:ok, ep1} = Graph.add_node(g, "Episode", %{name: "ep_1", action: "weather.query"})
    {:ok, ep2} = Graph.add_node(g, "Episode", %{name: "ep_2", action: "weather.query"})
    {:ok, _} = Graph.add_edge(g, ep1.id, fact.id, EdgeLabels.evidence_for())
    {:ok, _} = Graph.add_edge(g, ep2.id, fact.id, EdgeLabels.evidence_for())

    %{fact: fact, ep1: ep1, ep2: ep2}
  end

  @doc "Seed the world's conversation_graph with a sample conversation thread."
  def seed_conversation_graph(world_id) do
    g = Graph.world_graph("conversation_graph", world_id)

    {:ok, conv} = Graph.add_node(g, "Conversation", %{
      name: "test_conv_1", world_id: world_id, created_at: System.system_time(:millisecond)
    })
    {:ok, msg1} = Graph.add_node(g, "Message", %{
      name: "msg_1", role: "user", content: "What is the weather in London?"
    })
    {:ok, msg2} = Graph.add_node(g, "Message", %{
      name: "msg_2", role: "assistant", content: "The weather in London is currently clear."
    })
    {:ok, topic_weather} = Graph.add_node(g, "Topic", %{name: "weather.query"})
    {:ok, topic_greeting} = Graph.add_node(g, "Topic", %{name: "smalltalk.greetings.hello"})
    {:ok, _} = Graph.add_edge(g, conv.id, msg1.id, EdgeLabels.contains())
    {:ok, _} = Graph.add_edge(g, conv.id, msg2.id, EdgeLabels.contains())
    {:ok, _} = Graph.add_edge(g, msg1.id, msg2.id, EdgeLabels.follows())
    {:ok, _} = Graph.add_edge(g, msg1.id, topic_weather.id, EdgeLabels.has_topic())
    {:ok, _} = Graph.add_edge(g, topic_greeting.id, topic_weather.id, EdgeLabels.topic_transition(), %{count: 2})

    %{conversation: conv, msg1: msg1, msg2: msg2, topic_weather: topic_weather, topic_greeting: topic_greeting}
  end

  @doc "Seed the world's epistemic_graph with a JTMS dependency network."
  def seed_epistemic_graph(world_id) do
    g = Graph.world_graph("epistemic_graph", world_id)

    {:ok, premise} = Graph.add_node(g, "JTMSNode", %{
      name: "paris_capital", datum: "Paris is the capital of France",
      node_type: "premise", label: "in"
    })
    {:ok, assumption} = Graph.add_node(g, "JTMSNode", %{
      name: "user_likes_jazz", datum: "User likes jazz",
      node_type: "assumption", label: "in"
    })
    {:ok, derived} = Graph.add_node(g, "JTMSNode", %{
      name: "recommend_jazz", datum: "Recommend jazz content",
      node_type: "derived", label: "in"
    })
    {:ok, justification} = Graph.add_node(g, "Justification", %{
      name: "j1", informant: "preference_rule"
    })
    {:ok, _} = Graph.add_edge(g, justification.id, derived.id, EdgeLabels.supports())
    {:ok, _} = Graph.add_edge(g, justification.id, assumption.id, EdgeLabels.requires_in())

    %{premise: premise, assumption: assumption, derived: derived, justification: justification}
  end

  @doc "Seed the shared pos_graph with POS tags, transitions, and sample tokens."
  def seed_pos_graph do
    tags = ~w(NOUN PROPN VERB AUX ADJ ADV PRON DET ADP CONJ PART NUM INTJ PUNCT SYM X)

    tag_nodes =
      Enum.map(tags, fn tag ->
        {:ok, node} = Graph.add_node("pos_graph", "POSTag", %{name: tag})
        {tag, node}
      end)
      |> Map.new()

    transitions = [
      {"DET", "NOUN", 0.65}, {"DET", "ADJ", 0.25},
      {"ADJ", "NOUN", 0.70}, {"NOUN", "VERB", 0.40},
      {"PRON", "VERB", 0.55}, {"VERB", "ADP", 0.30},
      {"VERB", "NOUN", 0.25}, {"ADP", "NOUN", 0.50},
      {"ADP", "PROPN", 0.30}, {"PROPN", "VERB", 0.35}
    ]

    for {from, to, freq} <- transitions do
      from_node = tag_nodes[from]
      to_node = tag_nodes[to]
      Graph.add_edge("pos_graph", from_node.id, to_node.id, EdgeLabels.followed_by(), %{frequency: freq})
    end

    {:ok, the_token} = Graph.add_node("pos_graph", "Token", %{name: "the"})
    {:ok, _} = Graph.add_edge("pos_graph", the_token.id, tag_nodes["DET"].id, EdgeLabels.has_tag(), %{count: 100})

    {:ok, run_token} = Graph.add_node("pos_graph", "Token", %{name: "run"})
    {:ok, _} = Graph.add_edge("pos_graph", run_token.id, tag_nodes["VERB"].id, EdgeLabels.has_tag(), %{count: 80})
    {:ok, _} = Graph.add_edge("pos_graph", run_token.id, tag_nodes["NOUN"].id, EdgeLabels.has_tag(), %{count: 20})

    %{tag_nodes: tag_nodes, tokens: %{the: the_token, run: run_token}}
  end
end
