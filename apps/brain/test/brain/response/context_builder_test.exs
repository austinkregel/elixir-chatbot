defmodule Brain.Response.ContextBuilderTest do
  use ExUnit.Case, async: true

  alias Brain.Analysis.InternalModel
  alias Brain.Response.ContextBuilder

  describe "build/2" do
    test "raises when opts name no world" do
      model = %InternalModel{raw_input: "Hello", analyses: []}

      assert_raise ArgumentError, ~r/ContextBuilder.build requires a :world_id option/, fn ->
        ContextBuilder.build(model, [])
      end
    end

    test "raises when the world is not a world id" do
      model = %InternalModel{raw_input: "Hello", analyses: []}

      assert_raise ArgumentError, ~r/ContextBuilder.build requires a :world_id option/, fn ->
        ContextBuilder.build(model, world_id: "")
      end
    end
  end
end
