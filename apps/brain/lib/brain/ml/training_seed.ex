defmodule Brain.ML.TrainingSeed do
  @moduledoc """
  The seed for stochastic training steps, and the random state derived from it.

  Every trainer that draws random numbers takes its seed from here and uses a
  state of its own (`:rand.seed_s/2`, `:rand.uniform_s/1,2`). None of them calls
  `:rand.seed/2`, which reseeds the calling process and so decides what every
  later draw in that process gets; that made one trainer's determinism depend
  on which trainer ran before it.
  """

  @doc "The configured training seed. Raises if it is not configured."
  @spec get!() :: integer()
  def get! do
    case Application.get_env(:brain, :ml, [])[:training_seed] do
      seed when is_integer(seed) ->
        seed

      other ->
        raise "Brain.ML.TrainingSeed: :training_seed under config :brain, :ml must be an " <>
                "integer, got #{inspect(other)}"
    end
  end

  @doc "A fresh random state for `seed`, independent of the process's own state."
  @spec state(integer()) :: :rand.state()
  def state(seed) when is_integer(seed), do: :rand.seed_s(:exsss, {seed, seed, seed})

  @doc """
  Shuffles `list` using `rand`, returning the shuffled list and the advanced
  state. The seeded counterpart of `Enum.shuffle/1`, which draws from the
  process's state.
  """
  @spec shuffle(list(), :rand.state()) :: {list(), :rand.state()}
  def shuffle(list, rand) do
    {keyed, rand} =
      Enum.map_reduce(list, rand, fn item, rand ->
        {key, rand} = :rand.uniform_s(rand)
        {{key, item}, rand}
      end)

    {keyed |> Enum.sort_by(&elem(&1, 0)) |> Enum.map(&elem(&1, 1)), rand}
  end

  @doc """
  Picks one element of a non-empty `list` using `rand`, returning it and the
  advanced state. The seeded counterpart of `Enum.random/1`.
  """
  @spec pick(nonempty_list(), :rand.state()) :: {term(), :rand.state()}
  def pick([_ | _] = list, rand) do
    {index, rand} = :rand.uniform_s(length(list), rand)
    {Enum.at(list, index - 1), rand}
  end
end
