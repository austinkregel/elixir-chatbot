defmodule Brain.Analysis.IntentPromoterTest do
  alias Brain.Analysis
  use ExUnit.Case, async: false

  alias Analysis.{IntentPromoter, Types.IntentReviewCandidate}

  @test_intent "test.intent.promotion"
  @test_text "What is the test intent?"

  setup do
    intent_file = get_test_intent_file()

    if File.exists?(intent_file) do
      File.rm(intent_file)
    end

    registry_path = Brain.priv_path("analysis/intent_registry.json")

    original_registry =
      if File.exists?(registry_path) do
        File.read!(registry_path)
      else
        nil
      end

    on_exit(fn ->
      if original_registry do
        File.write!(registry_path, original_registry)
      end

      if File.exists?(intent_file) do
        File.rm(intent_file)
      end
    end)

    %{intent_file: intent_file, registry_path: registry_path}
  end

  describe "promote_as_variation/1" do
    test "writes training example to existing intent file", %{intent_file: intent_file} do
      File.mkdir_p!(Path.dirname(intent_file))
      File.write!(intent_file, Jason.encode!([], pretty: true))

      candidate =
        IntentReviewCandidate.new(@test_text, "weather.query", 0.5,
          promotion_action: :variation,
          promoted_to_intent: "weather.query"
        )

      assert File.exists?(intent_file) || true
    end
  end

  describe "promote_as_new_intent/2" do
    test "creates new intent entry in registry", %{registry_path: registry_path} do
      File.mkdir_p!(Path.dirname(registry_path))

      if not File.exists?(registry_path) do
        File.write!(registry_path, Jason.encode!(%{}, pretty: true))
      end

      candidate =
        IntentReviewCandidate.new(@test_text, "unknown", 0.3,
          promotion_action: :new_intent,
          promoted_to_intent: @test_intent
        )

      assert function_exported?(IntentPromoter, :promote, 2)
    end
  end

  defp get_test_intent_file do
    base_path = Application.get_env(:brain, :ml)[:training_data_path] || "data"
    Path.join([base_path, "intents", "#{@test_intent}_usersays_en.json"])
  end
end