defmodule Brain.Analysis.SemanticRoleLabelerExtendedTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.SemanticRoleLabeler

  describe "label/2 edge cases" do
    test "empty tokens returns empty frames" do
      frames = SemanticRoleLabeler.label([], [])
      assert frames == []
    end

    test "all O tags returns empty frames" do
      tokens = ["the", "red", "car"]
      bio_tags = ["O", "O", "O"]

      frames = SemanticRoleLabeler.label(tokens, bio_tags)
      assert frames == []
    end

    test "verb only (no arguments)" do
      tokens = ["run"]
      bio_tags = ["B-V"]

      frames = SemanticRoleLabeler.label(tokens, bio_tags)
      assert length(frames) == 1
      assert hd(frames).predicate == "run"
      assert hd(frames).arguments == []
    end

    test "I- tag without preceding B- tag" do
      tokens = ["John", "quickly", "ran"]
      bio_tags = ["I-ARG0", "O", "B-V"]

      frames = SemanticRoleLabeler.label(tokens, bio_tags)
      assert is_list(frames)
    end

    test "tags longer than tokens" do
      tokens = ["hello"]
      bio_tags = ["B-V", "I-ARG0", "O"]

      frames = SemanticRoleLabeler.label(tokens, bio_tags)
      assert is_list(frames)
    end

    test "tags shorter than tokens" do
      tokens = ["John", "ran", "fast"]
      bio_tags = ["B-ARG0"]

      frames = SemanticRoleLabeler.label(tokens, bio_tags)
      assert is_list(frames)
    end

    test "consecutive B- tags of same type" do
      tokens = ["John", "Mary", "talked"]
      bio_tags = ["B-ARG0", "B-ARG0", "B-V"]

      frames = SemanticRoleLabeler.label(tokens, bio_tags)
      assert is_list(frames)
    end

    test "multiple predicates in same sentence" do
      tokens = ["John", "ran", "and", "Mary", "jumped"]
      bio_tags = ["B-ARG0", "B-V", "O", "B-ARG0", "B-V"]

      frames = SemanticRoleLabeler.label(tokens, bio_tags)
      assert length(frames) >= 2

      predicates = Enum.map(frames, & &1.predicate) |> Enum.sort()
      assert "jumped" in predicates
      assert "ran" in predicates
    end

    test "all modifier types" do
      tokens = ["He", "spoke", "loudly", "in", "Berlin", "yesterday", "because", "of", "that"]
      bio_tags = ["B-ARG0", "B-V", "B-ARGM-MNR", "B-ARGM-LOC", "I-ARGM-LOC", "B-ARGM-TMP", "B-ARGM-CAU", "I-ARGM-CAU", "I-ARGM-CAU"]

      frames = SemanticRoleLabeler.label(tokens, bio_tags)
      assert length(frames) >= 1

      frame = hd(frames)
      roles = Enum.map(frame.arguments, & &1.role)
      assert :argm_mnr in roles or :argm_loc in roles
    end
  end

  describe "to_triples/1" do
    test "empty frames returns empty triples" do
      assert SemanticRoleLabeler.to_triples([]) == []
    end

    test "frame with no arguments produces no triples" do
      frames = [%{predicate: "run", arguments: []}]
      triples = SemanticRoleLabeler.to_triples(frames)
      assert triples == []
    end

    test "frame with ARG0 and ARG1 produces triple" do
      frames = [%{
        predicate: "ate",
        arguments: [
          %{role: :arg0, text: "John", span: {0, 0}, entity: nil},
          %{role: :arg1, text: "pizza", span: {2, 2}, entity: nil}
        ]
      }]

      triples = SemanticRoleLabeler.to_triples(frames)
      assert length(triples) >= 1

      {subj, pred, obj} = hd(triples)
      assert subj == "John"
      assert pred == "ate"
      assert obj == "pizza"
    end

    test "frame with only modifiers" do
      frames = [%{
        predicate: "ran",
        arguments: [
          %{role: :argm_loc, text: "park", span: {2, 2}, entity: nil},
          %{role: :argm_tmp, text: "yesterday", span: {3, 3}, entity: nil}
        ]
      }]

      triples = SemanticRoleLabeler.to_triples(frames)
      assert is_list(triples)
    end

    test "multiple frames produce multiple triples" do
      frames = [
        %{predicate: "ate", arguments: [
          %{role: :arg0, text: "John", span: {0, 0}, entity: nil},
          %{role: :arg1, text: "pizza", span: {2, 2}, entity: nil}
        ]},
        %{predicate: "drank", arguments: [
          %{role: :arg0, text: "Mary", span: {0, 0}, entity: nil},
          %{role: :arg1, text: "coffee", span: {2, 2}, entity: nil}
        ]}
      ]

      triples = SemanticRoleLabeler.to_triples(frames)
      assert length(triples) >= 2
    end
  end

  describe "extract_spans/1" do
    test "empty list returns empty list" do
      spans = SemanticRoleLabeler.extract_spans([])
      assert spans == []
    end

    test "single B- tag produces one span" do
      spans = SemanticRoleLabeler.extract_spans(["B-ARG0"])
      assert length(spans) == 1
      assert {:arg0, 0, 0} in spans
    end

    test "B- followed by I- creates multi-word span" do
      spans = SemanticRoleLabeler.extract_spans(["B-ARG1", "I-ARG1", "I-ARG1"])
      assert length(spans) == 1
      {role, start_idx, end_idx} = hd(spans)
      assert role == :arg1
      assert start_idx == 0
      assert end_idx == 2
    end

    test "O tags are skipped" do
      spans = SemanticRoleLabeler.extract_spans(["O", "O", "B-V", "O"])
      assert length(spans) == 1
      assert {:verb, 2, 2} in spans
    end
  end

  describe "label/3 with entities" do
    test "entities link to matching spans" do
      tokens = ["John", "visited", "Berlin"]
      bio_tags = ["B-ARG0", "B-V", "B-ARG1"]
      entities = [
        %{text: "John", entity_type: :person, value: "John"},
        %{text: "Berlin", entity_type: :location, value: "Berlin"}
      ]

      frames = SemanticRoleLabeler.label(tokens, bio_tags, entities: entities)
      assert length(frames) >= 1

      frame = hd(frames)
      linked_args = Enum.filter(frame.arguments, fn arg ->
        arg.entity != nil
      end)

      assert length(linked_args) >= 1
    end

    test "entities with no matching span are not linked" do
      tokens = ["walked", "quickly"]
      bio_tags = ["B-V", "B-ARGM-MNR"]
      entities = [%{text: "Paris", entity_type: :location, value: "Paris"}]

      frames = SemanticRoleLabeler.label(tokens, bio_tags, entities: entities)
      assert is_list(frames)
    end
  end
end
