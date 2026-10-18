defmodule Brain.ML.TrainingSeedTest do
  use ExUnit.Case, async: false

  alias Brain.ML.TrainingSeed

  describe "get!/0" do
    test "returns the configured seed" do
      assert TrainingSeed.get!() == Application.get_env(:brain, :ml)[:training_seed]
    end

    test "raises when the seed is not configured as an integer" do
      ml = Application.get_env(:brain, :ml)
      on_exit(fn -> Application.put_env(:brain, :ml, ml) end)

      Application.put_env(:brain, :ml, Keyword.delete(ml, :training_seed))

      assert_raise RuntimeError, ~r/:training_seed/, fn -> TrainingSeed.get!() end
    end
  end

  describe "shuffle/2 and pick/2" do
    test "the same seed gives the same order and the same picks" do
      list = Enum.to_list(1..50)

      {first, rand_a} = TrainingSeed.shuffle(list, TrainingSeed.state(3))
      {second, rand_b} = TrainingSeed.shuffle(list, TrainingSeed.state(3))

      assert first == second
      assert Enum.sort(first) == list
      assert TrainingSeed.pick(list, rand_a) == TrainingSeed.pick(list, rand_b)
    end

    test "leave the calling process's random state untouched" do
      :rand.seed(:exsss, {1, 2, 3})
      expected = :rand.uniform()

      :rand.seed(:exsss, {1, 2, 3})
      {_, rand} = TrainingSeed.shuffle(Enum.to_list(1..50), TrainingSeed.state(3))
      TrainingSeed.pick(Enum.to_list(1..50), rand)

      assert :rand.uniform() == expected
    end
  end
end
