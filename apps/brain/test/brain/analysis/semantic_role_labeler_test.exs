defmodule Brain.Analysis.SemanticRoleLabelerTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.SemanticRoleLabeler

  describe "label/3" do
    test "extracts predicate-argument frame from BIO tags" do
      tokens = ["John", "eagerly", "visited", "New", "York"]
      bio_tags = ["B-ARG0", "O", "B-V", "B-ARG1", "I-ARG1"]

      frames = SemanticRoleLabeler.label(tokens, bio_tags)

      assert length(frames) == 1
      frame = hd(frames)

      assert frame.predicate == "visited"
      assert frame.predicate_index == 2

      arg0 = Enum.find(frame.arguments, &(&1.role == :arg0))
      assert arg0.text == "John"
      assert arg0.span == {0, 0}

      arg1 = Enum.find(frame.arguments, &(&1.role == :arg1))
      assert arg1.text == "New York"
      assert arg1.span == {3, 4}
    end

    test "handles multiple predicates in one sentence" do
      tokens = ["John", "visited", "Berlin", "and", "saw", "the", "wall"]
      bio_tags = ["B-ARG0", "B-V", "B-ARG1", "O", "B-V", "B-ARG1", "I-ARG1"]

      frames = SemanticRoleLabeler.label(tokens, bio_tags)

      assert length(frames) == 2

      predicates = Enum.map(frames, & &1.predicate)
      assert "visited" in predicates
      assert "saw" in predicates
    end

    test "handles modifier arguments (LOC, TMP, MNR)" do
      tokens = ["She", "ran", "quickly", "in", "the", "park", "yesterday"]
      bio_tags = ["B-ARG0", "B-V", "B-ARGM-MNR", "B-ARGM-LOC", "I-ARGM-LOC", "I-ARGM-LOC", "B-ARGM-TMP"]

      frames = SemanticRoleLabeler.label(tokens, bio_tags)

      assert length(frames) == 1
      frame = hd(frames)

      assert frame.predicate == "ran"

      roles = Enum.map(frame.arguments, & &1.role)
      assert :arg0 in roles
      assert :argm_mnr in roles
      assert :argm_loc in roles
      assert :argm_tmp in roles

      loc = Enum.find(frame.arguments, &(&1.role == :argm_loc))
      assert loc.text == "in the park"
    end

    test "links arguments to matching entities" do
      tokens = ["John", "visited", "Berlin"]
      bio_tags = ["B-ARG0", "B-V", "B-ARG1"]
      entities = [
        %{text: "John", type: :person, start_pos: 0},
        %{text: "Berlin", type: :location, start_pos: 2}
      ]

      frames = SemanticRoleLabeler.label(tokens, bio_tags, entities: entities)
      frame = hd(frames)

      arg0 = Enum.find(frame.arguments, &(&1.role == :arg0))
      assert arg0.entity != nil
      assert arg0.entity.type == :person

      arg1 = Enum.find(frame.arguments, &(&1.role == :arg1))
      assert arg1.entity != nil
      assert arg1.entity.type == :location
    end

    test "handles all-O tags (no predicate)" do
      tokens = ["hello", "world"]
      bio_tags = ["O", "O"]

      frames = SemanticRoleLabeler.label(tokens, bio_tags)
      assert frames == []
    end

    test "handles token struct input" do
      tokens = [
        %{text: "John"},
        %{text: "visited"},
        %{text: "Berlin"}
      ]
      bio_tags = ["B-ARG0", "B-V", "B-ARG1"]

      frames = SemanticRoleLabeler.label(tokens, bio_tags)
      assert length(frames) == 1
      assert hd(frames).predicate == "visited"
    end
  end

  describe "extract_spans/1" do
    test "extracts correct spans from BIO sequence" do
      tags = ["B-ARG0", "I-ARG0", "O", "B-V", "B-ARG1"]
      spans = SemanticRoleLabeler.extract_spans(tags)

      assert length(spans) == 3
      assert {:arg0, 0, 1} in spans
      assert {:verb, 3, 3} in spans
      assert {:arg1, 4, 4} in spans
    end

    test "handles consecutive B tags as separate spans" do
      tags = ["B-ARG0", "B-ARG1"]
      spans = SemanticRoleLabeler.extract_spans(tags)

      assert length(spans) == 2
    end

    test "handles empty tag list" do
      spans = SemanticRoleLabeler.extract_spans([])
      assert spans == []
    end
  end

  describe "to_triples/1" do
    test "converts frames to knowledge graph triples" do
      frames = [
        %{
          predicate: "visited",
          predicate_index: 1,
          arguments: [
            %{role: :arg0, text: "John", span: {0, 0}, entity: nil},
            %{role: :arg1, text: "Berlin", span: {2, 2}, entity: nil},
            %{role: :argm_tmp, text: "yesterday", span: {3, 3}, entity: nil}
          ]
        }
      ]

      triples = SemanticRoleLabeler.to_triples(frames)

      assert {"John", "visited", "Berlin"} in triples
      assert {"visited", "OCCURRED_AT", "yesterday"} in triples
    end

    test "handles missing arg0 or arg1" do
      frames = [
        %{
          predicate: "rained",
          predicate_index: 0,
          arguments: [
            %{role: :argm_loc, text: "Seattle", span: {1, 1}, entity: nil}
          ]
        }
      ]

      triples = SemanticRoleLabeler.to_triples(frames)
      assert {"rained", "LOCATED_AT", "Seattle"} in triples
    end
  end
end
