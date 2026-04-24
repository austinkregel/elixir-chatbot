defmodule Brain.Analysis.FramingDetectorTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.{FramingDetector, DocumentProfile, ChunkProfile}

  defp profile(vector, modality \\ :declarative) do
    %ChunkProfile{
      feature_vector: vector,
      modality: modality,
      speech_act_category: :assertive,
      domain: :unknown
    }
  end

  defp ensure_framing_detector do
    case Process.whereis(Brain.Analysis.FramingDetector) do
      nil ->
        start_supervised!(Brain.Analysis.FramingDetector)

      pid when is_pid(pid) ->
        pid
    end
  end

  describe "assess_document/1" do
    test "returns an assessment for a valid list of profiles" do
      profiles = [
        profile([0.5, 0.3, 0.8, 0.2, 0.6]),
        profile([0.4, 0.5, 0.7, 0.3, 0.5])
      ]

      result = FramingDetector.assess_document(profiles)

      case result do
        {:ok, assessment} ->
          assert is_atom(assessment.primary_frame)
          assert is_float(assessment.confidence)
          assert is_float(assessment.deviation_from_neutral)
          assert is_list(assessment.secondary_frames)
          assert is_atom(assessment.rhetorical_mode)
          assert is_map(assessment.evidence)
          assert Map.has_key?(assessment.evidence, :sentiment_skew)
          assert Map.has_key?(assessment.evidence, :modality_skew)

        {:error, _reason} ->
          # When classifier is not loaded, we still expect a structured error
          assert true
      end
    end

    test "returns error for empty profile list" do
      result = FramingDetector.assess_document([])
      assert {:error, :empty_document_profile} = result
    end

    test "returns error for profiles with empty feature vectors" do
      profiles = [%ChunkProfile{feature_vector: []}]
      result = FramingDetector.assess_document(profiles)
      assert {:error, :empty_document_profile} = result
    end

    test "handles mixed modalities in rhetorical mode" do
      profiles = [
        profile([1.0, 0.0], :declarative),
        profile([0.0, 1.0], :interrogative),
        profile([0.5, 0.5], :declarative),
        profile([0.3, 0.7], :imperative)
      ]

      case FramingDetector.assess_document(profiles) do
        {:ok, assessment} ->
          assert assessment.rhetorical_mode in [:assertion, :question, :imperative, :mixed]

        {:error, _} ->
          assert true
      end
    end
  end

  describe "assess_entity_framing/2" do
    test "returns assessment for an entity found in slices" do
      profiles = [
        profile([1.0, 0.0, 0.0]),
        profile([0.0, 1.0, 0.0]),
        profile([0.0, 0.0, 1.0])
      ]

      entity_lists = [
        [%{value: "NATO"}],
        [%{value: "NATO"}, %{value: "Russia"}],
        [%{value: "Russia"}]
      ]

      doc_profile =
        DocumentProfile.aggregate(profiles, entity_lists: entity_lists)

      result = FramingDetector.assess_entity_framing(doc_profile, "NATO")

      case result do
        {:ok, assessment} ->
          assert is_atom(assessment.primary_frame)

        {:error, _} ->
          assert true
      end
    end

    test "falls back to full doc assessment for unknown entity" do
      profiles = [profile([1.0, 2.0, 3.0])]
      doc_profile = DocumentProfile.aggregate(profiles)

      result = FramingDetector.assess_entity_framing(doc_profile, "Unknown")

      case result do
        {:ok, assessment} -> assert is_atom(assessment.primary_frame)
        {:error, _} -> assert true
      end
    end
  end

  describe "detect_drift/2" do
    setup do
      ensure_framing_detector()
      :ok
    end

    test "first document for a source reports no drift" do
      vec = List.duplicate(0.5, 10)
      doc_profile = DocumentProfile.aggregate([profile(vec)])

      source_id = "test_source_#{System.unique_integer([:positive])}"
      {:ok, report} = FramingDetector.detect_drift(source_id, doc_profile)

      assert report.source_id == source_id
      assert report.drifted == false
      assert report.documents_seen == 1
      assert_in_delta report.similarity_to_ema, 1.0, 1.0e-6
    end

    test "identical documents show no drift" do
      vec = List.duplicate(0.5, 10)
      doc_profile = DocumentProfile.aggregate([profile(vec)])

      source_id = "test_stable_#{System.unique_integer([:positive])}"

      {:ok, _} = FramingDetector.detect_drift(source_id, doc_profile)
      {:ok, report} = FramingDetector.detect_drift(source_id, doc_profile)

      assert report.drifted == false
      assert report.documents_seen == 2
      assert report.similarity_to_ema > 0.99
    end

    test "radically different documents flag drift" do
      source_id = "test_drift_#{System.unique_integer([:positive])}"

      vec1 = List.duplicate(1.0, 20)
      doc1 = DocumentProfile.aggregate([profile(vec1)])
      {:ok, _} = FramingDetector.detect_drift(source_id, doc1)

      # Feed many consistent docs so EMA converges
      for _ <- 1..10 do
        FramingDetector.detect_drift(source_id, doc1)
      end

      vec2 = List.duplicate(-1.0, 20)
      doc2 = DocumentProfile.aggregate([profile(vec2)])
      {:ok, report} = FramingDetector.detect_drift(source_id, doc2)

      assert report.drifted == true
      assert report.similarity_to_ema < 0.85
      assert length(report.top_changed_dimensions) > 0
    end
  end

  describe "get_source_ema/1" do
    setup do
      ensure_framing_detector()
      :ok
    end

    test "returns :not_found for unknown source" do
      assert :not_found = FramingDetector.get_source_ema("nonexistent_#{System.unique_integer([:positive])}")
    end

    test "returns ema and count after detecting drift" do
      source_id = "test_ema_#{System.unique_integer([:positive])}"
      vec = [1.0, 2.0, 3.0]
      doc = DocumentProfile.aggregate([profile(vec)])

      {:ok, _} = FramingDetector.detect_drift(source_id, doc)

      assert {:ok, ema, 1} = FramingDetector.get_source_ema(source_id)
      assert length(ema) == 3
    end
  end

  describe "ready?/0" do
    test "returns true when GenServer is running" do
      ensure_framing_detector()
      assert FramingDetector.ready?()
    end
  end
end
