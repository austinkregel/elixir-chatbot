defmodule Brain.Analysis.Types.IntentReviewCandidateTest do
  use ExUnit.Case, async: true

  alias Brain.Analysis.Types.IntentReviewCandidate

  # The world the test suite runs in, seeded by `mix test.models`.
  @world_id "default"

  describe "new/4" do
    test "records the world the text was analyzed in" do
      candidate =
        IntentReviewCandidate.new("turn the porch light purple", "smarthome.lights.color", 0.41,
          world_id: @world_id,
          conversation_id: "conv_1",
          second_score: 0.38,
          margin: 0.03
        )

      assert candidate.world_id == @world_id
      assert candidate.text == "turn the porch light purple"
      assert candidate.predicted_intent == "smarthome.lights.color"
      assert candidate.best_score == 0.41
      assert candidate.conversation_id == "conv_1"
      assert candidate.second_score == 0.38
      assert candidate.margin == 0.03
      assert candidate.status == :pending
    end

    test "raises without a :world_id" do
      assert_raise ArgumentError, ~r/IntentReviewCandidate.new needs :world_id/, fn ->
        IntentReviewCandidate.new("turn the porch light purple", "smarthome.lights.color", 0.41, [])
      end
    end

    test "raises when :world_id is not a world id" do
      for world_id <- [nil, "", :default] do
        assert_raise ArgumentError, ~r/IntentReviewCandidate option :world_id must be a world id/, fn ->
          IntentReviewCandidate.new("turn the porch light purple", "smarthome.lights.color", 0.41, world_id: world_id)
        end
      end
    end
  end
end
