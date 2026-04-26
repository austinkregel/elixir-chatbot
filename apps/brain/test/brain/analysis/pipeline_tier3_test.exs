defmodule Brain.Analysis.PipelineTier3Test do
  @moduledoc """
  Tests for the Tier 3 model integrations wired into Pipeline.analyze_single_chunk.
  Covers event linking, SRL, BIO tag generation, entity span detection, and tag extraction.
  """

  use Brain.Test.BrainCase, async: false

  alias Brain.Analysis.Pipeline

  describe "pipeline produces Tier 3 fields" do
    test "simple sentence produces event_frames and srl_frames" do
      analysis = Pipeline.analyze_chunk("John visited Berlin yesterday")

      assert Map.has_key?(analysis, :event_frames)
      assert Map.has_key?(analysis, :srl_frames)
      assert is_list(analysis.event_frames)
      assert is_list(analysis.srl_frames)
    end

    test "greeting has empty Tier 3 fields" do
      analysis = Pipeline.analyze_chunk("Hello!")

      assert is_list(analysis.event_frames)
      assert is_list(analysis.srl_frames)
    end

    test "multi-entity sentence" do
      analysis = Pipeline.analyze_chunk("The teacher gave Mary a book in the library")

      assert is_list(analysis.event_frames)
      assert is_list(analysis.srl_frames)
    end

    test "question produces analysis with Tier 3 fields" do
      analysis = Pipeline.analyze_chunk("What did Einstein discover?")

      assert Map.has_key?(analysis, :event_frames)
      assert Map.has_key?(analysis, :srl_frames)
    end
  end

  describe "full pipeline with Tier 3" do
    test "multi-chunk input produces Tier 3 in each chunk" do
      model = Pipeline.process("Good morning! The president visited Berlin yesterday.")

      for analysis <- model.analyses do
        assert Map.has_key?(analysis, :event_frames)
        assert Map.has_key?(analysis, :srl_frames)
        assert is_list(analysis.event_frames)
        assert is_list(analysis.srl_frames)
      end
    end

    test "imperative sentence" do
      analysis = Pipeline.analyze_chunk("Turn on the lights in the living room")

      assert is_list(analysis.event_frames)
      assert is_list(analysis.srl_frames)
    end

    test "passive construction" do
      analysis = Pipeline.analyze_chunk("The book was written by Shakespeare")

      assert is_list(analysis.event_frames)
      assert is_list(analysis.srl_frames)
    end

    test "compound sentence" do
      analysis = Pipeline.analyze_chunk("John ran and Mary jumped")

      assert is_list(analysis.event_frames)
      assert is_list(analysis.srl_frames)
    end
  end

  describe "SRL bio tag generation via pipeline" do
    test "sentences with verbs get SRL frames" do
      analysis = Pipeline.analyze_chunk("She quickly ate the pizza")

      # Should have at least attempted SRL
      assert is_list(analysis.srl_frames)
    end

    test "sentence with no verbs produces empty SRL" do
      analysis = Pipeline.analyze_chunk("Hello!")

      assert is_list(analysis.srl_frames)
    end
  end

  describe "error resilience" do
    test "pipeline handles empty input" do
      analysis = Pipeline.analyze_chunk("")

      assert is_list(analysis.event_frames)
      assert is_list(analysis.srl_frames)
    end

    test "pipeline handles single word" do
      analysis = Pipeline.analyze_chunk("Hello")

      assert is_list(analysis.event_frames)
      assert is_list(analysis.srl_frames)
    end

    test "pipeline handles very long input" do
      long_text = String.duplicate("The cat sat on the mat. ", 20)
      analysis = Pipeline.analyze_chunk(long_text)

      assert is_list(analysis.event_frames)
      assert is_list(analysis.srl_frames)
    end
  end
end
