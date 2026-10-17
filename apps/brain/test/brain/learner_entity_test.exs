defmodule Brain.LearnerEntityTest do
  @moduledoc """
  What `Brain.Learner` learns from a decided entity of
  `Brain.ML.EntityExtractor.extract_entities/2`, built here in that shape so
  the test is about the learner, not the extractor.
  """
  use ExUnit.Case, async: true

  alias Brain.Learner

  @decided %{
    entity_type: "city",
    value: "Seattle",
    match: "Seattle",
    start_pos: 10,
    end_pos: 16,
    confidence: 0.83,
    source: :gazetteer,
    resolved: true
  }

  describe "learned_entity!/1" do
    test "learns the decided entity's value, type and confidence" do
      assert Learner.learned_entity!(@decided) == %{
               "name" => "Seattle",
               "type" => "city",
               "properties" => %{},
               "confidence" => 0.83
             }
    end

    test "an entity missing a key a decided entity carries raises, naming it" do
      for key <- [:value, :entity_type, :confidence] do
        assert_raise ArgumentError, ~r/has no #{inspect(key)}/, fn ->
          Learner.learned_entity!(Map.delete(@decided, key))
        end

        assert_raise ArgumentError, ~r/has no #{inspect(key)}/, fn ->
          Learner.learned_entity!(Map.put(@decided, key, nil))
        end
      end
    end

    test "a type under :entity rather than :entity_type is not read" do
      entity = @decided |> Map.delete(:entity_type) |> Map.put(:entity, "city")

      assert_raise ArgumentError, ~r/has no :entity_type/, fn ->
        Learner.learned_entity!(entity)
      end
    end
  end
end
