defmodule Brain.Analysis.SemanticRoleIntegrationTest do
  use ExUnit.Case, async: false
  @moduletag :integration

  alias Brain.Analysis.SemanticRoleLabeler

  describe "semantic role integration" do
    test "SRL frames can be converted to graph-writable triples" do
      tokens = ["The", "president", "visited", "Berlin", "yesterday"]
      bio_tags = ["B-ARG0", "I-ARG0", "B-V", "B-ARG1", "B-ARGM-TMP"]

      entities = [
        %{text: "The president", type: :person, start_pos: 0},
        %{text: "Berlin", type: :location, start_pos: 3}
      ]

      frames = SemanticRoleLabeler.label(tokens, bio_tags, entities: entities)
      triples = SemanticRoleLabeler.to_triples(frames)

      assert length(frames) == 1
      assert length(triples) >= 2

      assert {"The president", "visited", "Berlin"} in triples
      assert {"visited", "OCCURRED_AT", "yesterday"} in triples
    end

    test "multiple frames produce independent triple sets" do
      tokens = ["John", "gave", "Mary", "the", "book", "in", "the", "library"]
      bio_tags = ["B-ARG0", "B-V", "B-ARG2", "B-ARG1", "I-ARG1", "B-ARGM-LOC", "I-ARGM-LOC", "I-ARGM-LOC"]

      frames = SemanticRoleLabeler.label(tokens, bio_tags)
      triples = SemanticRoleLabeler.to_triples(frames)

      assert length(frames) == 1
      frame = hd(frames)

      assert frame.predicate == "gave"

      arg0 = Enum.find(frame.arguments, &(&1.role == :arg0))
      assert arg0.text == "John"

      arg1 = Enum.find(frame.arguments, &(&1.role == :arg1))
      assert arg1.text == "the book"

      arg2 = Enum.find(frame.arguments, &(&1.role == :arg2))
      assert arg2.text == "Mary"

      loc = Enum.find(frame.arguments, &(&1.role == :argm_loc))
      assert loc.text == "in the library"

      assert {"John", "gave", "the book"} in triples
      assert {"gave", "LOCATED_AT", "in the library"} in triples
    end

    test "SRL output is consistent with entity extraction" do
      tokens = ["Einstein", "discovered", "relativity", "in", "Switzerland"]
      bio_tags = ["B-ARG0", "B-V", "B-ARG1", "B-ARGM-LOC", "I-ARGM-LOC"]

      entities = [
        %{text: "Einstein", type: :person, start_pos: 0},
        %{text: "Switzerland", type: :location, start_pos: 4}
      ]

      frames = SemanticRoleLabeler.label(tokens, bio_tags, entities: entities)
      frame = hd(frames)

      arg0 = Enum.find(frame.arguments, &(&1.role == :arg0))
      assert arg0.entity != nil
      assert arg0.entity.type == :person

      loc = Enum.find(frame.arguments, &(&1.role == :argm_loc))
      assert loc.text == "in Switzerland"
    end
  end
end
