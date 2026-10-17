defmodule Brain.Analysis.EntityDisambiguatorInferrerTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.{EntityDisambiguator, EntityTypeScorer, TypeHierarchy}
  alias Brain.Worlds.TypeInferrer

  # A world no pattern was learned in: the type inferrer infers nothing, so the
  # context's expected types decide. The intent's expected types are weighed by
  # the slot type preferences loaded for English.
  @world "test_disambiguator_inferrer"

  setup do
    if Process.whereis(TypeHierarchy) == nil, do: start_supervised!({TypeHierarchy, name: TypeHierarchy})
    TypeInferrer.init()
    :ok
  end

  describe "infer_type_with_type_inferrer/3" do
    test "with no context, the entity keeps its own type" do
      entity = %{entity_type: "city", value: "Paris"}

      result = EntityDisambiguator.infer_type_with_type_inferrer(entity, [{"in", "ADP"}, {"Paris", "PROPN"}], %{world_id: @world})

      assert result.entity_type == "city"
      assert result.disambiguation_source == :type_inferrer
    end

    test "with an intent, the entity takes the most weighted type the intent's slots list that it is compatible with" do
      entity = %{entity_type: "city", value: "Paris"}
      context = %{world_id: @world, intent: "weather.query"}

      weights = context.intent |> EntityTypeScorer.expected_types() |> EntityTypeScorer.Expected.weights()
      assert weights != %{}

      expected =
        if Map.has_key?(weights, "city") do
          "city"
        else
          weights
          |> Enum.sort_by(fn {type, weight} -> {-weight, type} end)
          |> Enum.find_value(fn {type, _weight} -> if TypeHierarchy.compatible?("city", type), do: type end)
        end

      assert is_binary(expected)

      result = EntityDisambiguator.infer_type_with_type_inferrer(entity, [{"in", "ADP"}, {"Paris", "PROPN"}], context)

      assert result.entity_type == expected
    end
  end
end
