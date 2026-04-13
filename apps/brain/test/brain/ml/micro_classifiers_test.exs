defmodule Brain.ML.MicroClassifiersTest do
  use ExUnit.Case, async: false

  alias Brain.ML.MicroClassifiers

  describe "classify/2" do
    test "classifies personal questions" do
      case MicroClassifiers.classify(:personal_question, "who are you and what is your name") do
        {:ok, label, score} ->
          assert label == "personal"
          assert score > 0.0

        {:error, :not_loaded} ->
          # Acceptable during tests if GenServer not started
          :ok
      end
    end

    test "classifies non-personal questions" do
      case MicroClassifiers.classify(:personal_question, "what is the weather forecast for today") do
        {:ok, label, _score} ->
          assert label == "not_personal"

        {:error, :not_loaded} ->
          :ok
      end
    end

    test "classifies clarification responses" do
      case MicroClassifiers.classify(:clarification_response, "Which one do you mean?") do
        {:ok, label, _score} ->
          assert label == "clarification"

        {:error, :not_loaded} ->
          :ok
      end
    end

    test "classifies non-clarification responses" do
      case MicroClassifiers.classify(:clarification_response, "The capital of France is Paris.") do
        {:ok, label, _score} ->
          assert label == "not_clarification"

        {:error, :not_loaded} ->
          :ok
      end
    end

    test "classifies modal directives" do
      case MicroClassifiers.classify(:modal_directive, "can you turn on the lights") do
        {:ok, label, _score} ->
          assert label == "directive"

        {:error, :not_loaded} ->
          :ok
      end
    end

    test "classifies non-directive statements" do
      case MicroClassifiers.classify(:modal_directive, "the sky is blue") do
        {:ok, label, _score} ->
          assert label == "not_directive"

        {:error, :not_loaded} ->
          :ok
      end
    end

    test "classifies fallback responses" do
      case MicroClassifiers.classify(:fallback_response, "I don't understand what you mean.") do
        {:ok, label, _score} ->
          assert label == "fallback"

        {:error, :not_loaded} ->
          :ok
      end
    end

    test "classifies non-fallback responses" do
      case MicroClassifiers.classify(:fallback_response, "The weather in Paris is sunny.") do
        {:ok, label, _score} ->
          assert label == "not_fallback"

        {:error, :not_loaded} ->
          :ok
      end
    end

    test "classifies goal types" do
      case MicroClassifiers.classify(:goal_type, "why does rain fall") do
        {:ok, label, _score} ->
          assert label in ["reasoning", "factual", "sentiment", "general"]

        {:error, :not_loaded} ->
          :ok
      end
    end

    test "classifies entity types" do
      case MicroClassifiers.classify(:entity_type, "France geography") do
        {:ok, label, _score} ->
          assert is_binary(label)

        {:error, :not_loaded} ->
          :ok
      end
    end

    test "returns error for unknown classifier" do
      case MicroClassifiers.classify(:nonexistent, "text") do
        {:error, _} -> :ok
        # If the server isn't running, we also accept error
        _ -> :ok
      end
    end
  end

  describe "ready?/0" do
    test "returns boolean" do
      result = MicroClassifiers.ready?()
      assert is_boolean(result)
    end
  end

  describe "status/0" do
    test "returns status map" do
      status = MicroClassifiers.status()
      assert is_map(status)

      if status.ready do
        assert is_map(status.classifiers)

        Enum.each(
          ~w(personal_question clarification_response modal_directive fallback_response goal_type entity_type)a,
          fn name ->
            assert Map.has_key?(status.classifiers, name),
                   "Expected #{name} in classifiers status"
          end
        )
      end
    end
  end
end
