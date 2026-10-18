defmodule Mix.Tasks.TrainLattice do
  @moduledoc """
  Standalone task to generate lattice phrase inventory.

  Equivalent to running stage 7 of `mix train`.

  ## Usage

      mix train_lattice [--verbose]
  """

  use Mix.Task

  @shortdoc "Generate lattice phrase inventory (standalone)"

  def run(args) do
    Mix.Tasks.GenLatticeData.run(args)
  end
end
