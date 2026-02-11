defmodule FourthWall do
  @moduledoc """
  Codebase maintenance tooling for the ChatBot umbrella project.

  FourthWall provides automated code fixers for Credo issues, with a focus on
  safety and correctness. All operations are dry-run by default and require
  an explicit `--apply` flag to modify files.

  ## Core Principles

  1. **Dry-run by default** - Never write files without explicit `--apply`
  2. **TDD** - All fixers are developed test-first
  3. **95%+ coverage** - Required before running on real codebase
  4. **Corruption detection** - Safety checks prevent known corruption patterns

  ## Available Tasks

  - `mix credo_fix` - Master orchestrator for all fixers
  - `mix credo_fix.trailing_whitespace` - Remove trailing whitespace
  - `mix credo_fix.large_numbers` - Add underscores to large numbers
  - `mix credo_fix.length_check` - Replace `length(x) == 0` with `x == []`
  - `mix credo_fix.map_join` - Convert `Enum.map |> Enum.join` to `Enum.map_join`
  - `mix credo_fix.alias_usage` - Add aliases for frequently used modules
  """

  @doc """
  Returns the version of FourthWall.
  """
  def version, do: "0.1.0"
end
