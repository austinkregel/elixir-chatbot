defmodule ChatBot.Memory.TypesTest do
  use ExUnit.Case, async: true

  alias ChatBot.Memory.Types.{Episode, SemanticFact, Procedure}

  describe "Episode" do
    test "new/5 creates an episode with auto-generated id and timestamp" do
      episode = Episode.new("hello", "greeting", "hi there", ["greeting"], [0.1, 0.2])

      assert episode.state == "hello"
      assert episode.action == "greeting"
      assert episode.outcome == "hi there"
      assert episode.tags == ["greeting"]
      assert episode.embedding == [0.1, 0.2]
      assert is_binary(episode.id)
      assert String.length(episode.id) == 32
      assert is_integer(episode.timestamp)
      assert episode.semantic_id == nil
    end

    test "new/5 generates unique ids" do
      ep1 = Episode.new("a", "b", "c", [], [])
      ep2 = Episode.new("a", "b", "c", [], [])

      assert ep1.id != ep2.id
    end

    test "default values for optional fields" do
      episode = %Episode{}

      assert episode.tags == []
      assert episode.embedding == []
      assert episode.semantic_id == nil
    end
  end

  describe "SemanticFact" do
    test "new/4 creates a semantic fact with auto-generated id and timestamp" do
      fact = SemanticFact.new("greetings pattern", [0.1, 0.2], ["ep1", "ep2"], ["greeting"])

      assert fact.representation == "greetings pattern"
      assert fact.embedding == [0.1, 0.2]
      assert fact.evidence_ids == ["ep1", "ep2"]
      assert fact.tags == ["greeting"]
      assert is_binary(fact.id)
      assert String.length(fact.id) == 32
      assert is_integer(fact.timestamp)
    end

    test "new/4 generates unique ids" do
      f1 = SemanticFact.new("a", [], [], [])
      f2 = SemanticFact.new("a", [], [], [])

      assert f1.id != f2.id
    end
  end

  describe "Procedure" do
    test "new/3 creates a procedure with auto-generated id and timestamp" do
      proc = Procedure.new("user greeting", "respond hello", ["greeting"])

      assert proc.state == "user greeting"
      assert proc.action == "respond hello"
      assert proc.tags == ["greeting"]
      assert is_binary(proc.id)
      assert String.length(proc.id) == 32
      assert is_integer(proc.timestamp)
    end
  end
end
