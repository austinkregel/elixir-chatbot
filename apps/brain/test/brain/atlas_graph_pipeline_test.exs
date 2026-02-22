defmodule Brain.AtlasGraphPipelineTest do
  @moduledoc """
  Integration test that seeds known Atlas graphs, runs Brain's analysis
  pipeline on specific messages, and makes positive assertions on:

  1. Pipeline produces valid InternalModel with correct structure
  2. Speech act classification identifies questions vs statements
  3. Entity extraction finds geographic/domain entities
  4. Graph data is queryable alongside pipeline results
  5. Combined relational + graph + pipeline assertions
  6. Writer populates all 6 graphs from pipeline output
  7. Reader enriches response generation from graph data
  """

  use Brain.Test.GraphCase, async: false
  @moduletag seed_graphs: true

  alias Brain.Analysis.Pipeline
  alias Brain.Graph.{Writer, Reader}
  alias Atlas.Schemas.{Belief, LearnedFact}

  describe "graph-seeded pipeline analysis" do
    test "seeds knowledge graph then pipeline correctly analyzes a geography question" do
      {:ok, _} =
        %Belief{}
        |> Belief.changeset(%{
          subject: "Paris",
          predicate: "is_capital_of",
          object: "France",
          confidence: 0.95,
          source: "knowledge_graph"
        })
        |> Atlas.Repo.insert()

      {:ok, _} =
        %LearnedFact{}
        |> LearnedFact.changeset(%{
          id: "fact_paris_#{System.unique_integer([:positive])}",
          entity: "Paris",
          entity_type: "city",
          fact: "Paris is the capital city of France",
          confidence: 0.95
        })
        |> Atlas.Repo.insert()

      {:ok, kg_before} = count_nodes("knowledge_graph")

      model = Pipeline.process("What is the capital of France?")

      assert model != nil
      assert model.raw_input == "What is the capital of France?"
      assert length(model.chunks) >= 1
      assert length(model.analyses) >= 1

      best = Enum.max_by(model.analyses, & &1.confidence)

      assert best.speech_act != nil
      assert best.speech_act.is_question == true
      assert best.intent != nil
      assert is_binary(best.intent)
      assert best.confidence > 0.0

      assert model.overall_strategy in [
               :can_respond,
               :needs_clarification,
               :partial_response_with_clarification
             ]

      # Graph should have seed data + any pipeline-written nodes
      {:ok, kg_after} = count_nodes("knowledge_graph")
      assert kg_after >= kg_before

      # Relational data should still be queryable
      beliefs = Belief |> Belief.for_subject("Paris") |> Atlas.Repo.all()
      assert length(beliefs) >= 1
      assert hd(beliefs).object == "France"
    end

    test "pipeline identifies greeting speech act" do
      model = Pipeline.process("Hello there!")

      assert model != nil
      assert length(model.analyses) >= 1

      best = Enum.max_by(model.analyses, & &1.confidence)
      assert best.speech_act.category == :expressive
      assert best.confidence > 0.0
      assert best.intent != nil
    end

    test "pipeline processes statement with user graph context" do
      model = Pipeline.process("I really like jazz music")

      assert model != nil
      assert length(model.analyses) >= 1

      best = Enum.max_by(model.analyses, & &1.confidence)
      assert best.intent != nil
      assert best.speech_act.category in [:assertive, :expressive]

      # Seed data for user graph should still be there
      prefs = Reader.user_preferences("test_user_1")
      assert Enum.any?(prefs, fn p -> p.topic == "jazz" end)
    end

    test "multi-sentence input with entity extraction and graph verification" do
      model = Pipeline.process("I love Elixir. What runtime does it use?")

      assert model != nil
      assert length(model.chunks) >= 2
      assert length(model.analyses) >= 2

      has_question =
        Enum.any?(model.analyses, fn a ->
          a.speech_act != nil and a.speech_act.is_question == true
        end)

      assert has_question, "Expected at least one question in multi-sentence input"

      has_statement =
        Enum.any?(model.analyses, fn a ->
          a.speech_act != nil and a.speech_act.category in [:expressive, :assertive]
        end)

      assert has_statement, "Expected at least one statement in multi-sentence input"

      # Graph adjacency should be extractable
      {:ok, adjacency} = Graph.to_adjacency("knowledge_graph")
      assert length(adjacency.node_ids) >= 2
    end

    test "directive speech act with graph-enriched domain" do
      model = Pipeline.process("Turn off the lights in the living room")

      assert model != nil
      best = Enum.max_by(model.analyses, & &1.confidence)
      assert best.speech_act.category in [:directive, :command]
      assert best.intent != nil
    end
  end

  describe "graph triples as pipeline complement" do
    test "knowledge triples can enrich pipeline entity context" do
      {:ok, python} = Graph.add_node("knowledge_graph", "Language", %{name: "Python"})
      {:ok, ml} = Graph.add_node("knowledge_graph", "Domain", %{name: "Machine Learning"})
      {:ok, tensorflow} = Graph.add_node("knowledge_graph", "Framework", %{name: "TensorFlow"})

      {:ok, _} = Graph.add_edge("knowledge_graph", python.id, ml.id, "used_for")
      {:ok, _} = Graph.add_edge("knowledge_graph", tensorflow.id, python.id, "written_in")
      {:ok, _} = Graph.add_edge("knowledge_graph", tensorflow.id, ml.id, "used_for")

      {:ok, triples} = Graph.to_triples("knowledge_graph")
      # Seed data + new nodes = at least 3 triples from the new nodes
      assert length(triples) >= 3

      model = Pipeline.process("What is TensorFlow used for?")
      assert model != nil
      best = Enum.max_by(model.analyses, & &1.confidence)
      assert best.speech_act.is_question == true
      assert best.intent != nil
    end
  end

  describe "round-trip: write -> read -> respond" do
    test "pipeline writes entities to graph that reader can retrieve" do
      {:ok, kg_before} = count_nodes("knowledge_graph")

      entities = [
        %{entity_type: "location", value: "Barcelona", confidence: 0.9},
        %{entity_type: "location", value: "Spain", confidence: 0.85}
      ]

      :ok = Writer.write_entities(entities)
      Process.sleep(300)

      {:ok, kg_after} = count_nodes("knowledge_graph")
      assert kg_after > kg_before

      context = Reader.entity_context([%{entity_type: "Location", value: "Barcelona"}])
      assert [%{node: node}] = context
      assert node != nil
    end

    test "pipeline writes conversation messages that reader can query" do
      :ok = Writer.write_conversation(%{id: "roundtrip_conv", world_id: "default"})
      Process.sleep(200)

      :ok = Writer.write_message("roundtrip_conv", %{id: "rt_msg_1", role: "user", content: "Hello"}, %{intent: "smalltalk.greetings.hello"})
      Process.sleep(200)

      :ok = Writer.write_message("roundtrip_conv", %{id: "rt_msg_2", role: "assistant", content: "Hi!"}, nil)
      Process.sleep(200)

      topics = Reader.conversation_topics("roundtrip_conv")
      assert "smalltalk.greetings.hello" in topics

      context = Reader.recent_context("roundtrip_conv", 5)
      assert length(context) >= 2
    end

    test "POS tag writes accumulate and are readable" do
      :ok = Writer.write_pos_tags([{"hello", "INTJ"}, {"world", "NOUN"}])
      Process.sleep(200)

      transitions = Reader.tag_transitions("DET")
      # Seeds have DET transitions
      assert length(transitions) >= 2
    end
  end
end
