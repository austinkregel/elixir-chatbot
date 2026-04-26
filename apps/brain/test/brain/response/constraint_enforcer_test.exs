defmodule Brain.Response.ConstraintEnforcerTest do
  @moduledoc """
  Tests for `Brain.Response.ConstraintEnforcer`.

  The enforcer is a structural-only gate: it rejects empty, too-short,
  too-long, or degenerate (repetitive) output. All semantic validation
  has moved to `ResponseEvaluator` scoring dimensions.
  """
  use ExUnit.Case, async: true

  alias Brain.Response.ConstraintEnforcer
  alias Brain.Response.Primitive

  defp greeting_primitives do
    [Primitive.new(:greeting, :hello, %{})]
  end

  describe "validate/3 structural checks" do
    test "passes through a normal response" do
      assert {:ok, "Hello there friend"} =
               ConstraintEnforcer.validate("Hello there friend", greeting_primitives())
    end

    test "rejects an empty string" do
      assert {:rejected, _reason} =
               ConstraintEnforcer.validate("", greeting_primitives())
    end

    test "rejects whitespace-only string" do
      assert {:rejected, _reason} =
               ConstraintEnforcer.validate("   \n  ", greeting_primitives())
    end

    test "rejects a single-word response as too short" do
      assert {:rejected, reason} =
               ConstraintEnforcer.validate("Hi", greeting_primitives())

      assert reason =~ "too short"
    end

    test "rejects a response exceeding max length" do
      long_text = String.duplicate("word ", 500)
      assert {:rejected, reason} = ConstraintEnforcer.validate(long_text, greeting_primitives())
      assert reason =~ "too long"
    end

    test "rejects degenerate repetitive output" do
      degenerate = Enum.join(List.duplicate("hello", 20), " ")
      assert {:rejected, reason} = ConstraintEnforcer.validate(degenerate, greeting_primitives())
      assert reason =~ "Degenerate"
    end

    test "passes short responses that are not degenerate" do
      assert {:ok, "Hi there!"} =
               ConstraintEnforcer.validate("Hi there!", greeting_primitives())
    end

    test "accepts response regardless of primitive content (no NLP checks)" do
      primitives = [
        Primitive.new(:content, :factual, %{
          facts: [%{fact: "Paris is the capital of France"}],
          entity_context: [%{value: "Paris"}]
        })
      ]

      assert {:ok, _} =
               ConstraintEnforcer.validate(
                 "I enjoy sunny weather in the countryside",
                 primitives
               )
    end

    test "accepts with empty primitives list" do
      assert {:ok, "Some response text here"} =
               ConstraintEnforcer.validate("Some response text here", [])
    end
  end
end
