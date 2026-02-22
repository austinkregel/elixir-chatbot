defmodule Brain.Analysis.IntentPromoterTest do
  alias Brain.Analysis
  use Brain.Test.GraphCase, async: false

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
        IntentReviewCandidate.new(@test_text, @test_intent, 0.5,
          promotion_action: :variation,
          promoted_to_intent: @test_intent
        )

      assert {:ok, :variation_added} = IntentPromoter.promote(candidate)

      # Verify file was updated with the training example
      assert File.exists?(intent_file),
             "Intent training file should exist after promotion"

      {:ok, content} = File.read(intent_file)
      {:ok, data} = Jason.decode(content)
      assert length(data) >= 1, "Expected at least one training example in file"
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

      Code.ensure_loaded!(IntentPromoter)
      assert function_exported?(IntentPromoter, :promote, 2)

      # new_intent requires domain option
      assert {:ok, :new_intent_created} =
               IntentPromoter.promote(candidate, domain: "test", category: "directive")

      # Verify registry was updated
      {:ok, content} = File.read(registry_path)
      {:ok, registry} = Jason.decode(content)
      assert Map.has_key?(registry, @test_intent), "Registry should contain the new intent"
    end
  end

  defp get_test_intent_file do
    base_path = Application.get_env(:brain, :ml)[:training_data_path] || "data"
    Path.join([base_path, "intents", "#{@test_intent}_usersays_en.json"])
  end
end
