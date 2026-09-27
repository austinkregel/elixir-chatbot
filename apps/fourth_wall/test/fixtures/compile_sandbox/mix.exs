defmodule CompileSandbox.MixProject do
  use Mix.Project

  @moduledoc """
  A project for `Mix.Tasks.CredoFix.UnusedAlias` to compile under test.

  `collect_unused_aliases/1` cleans and recompiles a project in a child
  process to read the compiler's warnings. Pointed at the umbrella, that
  clean deletes the build the running test suite loads its modules from, so
  `config :fourth_wall, credo_fix_compile_dir:` points here instead: a real
  project, with nothing in it that another test needs.
  """

  def project do
    [
      app: :compile_sandbox,
      version: "0.1.0",
      elixir: "~> 1.14",
      deps: []
    ]
  end

  def application, do: [extra_applications: []]
end
