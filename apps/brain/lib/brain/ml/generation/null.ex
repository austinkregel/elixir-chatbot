defmodule Brain.ML.Generation.Null do
  @moduledoc """
  The no-generation backend: an instance that *chose* to run Brain for analysis
  only, with no text-generation model wired in.

      config :brain, :generation, backend: :null

  `generate/2` returns `{:error, :no_backend}`. This is an honest, chosen state —
  not a degradation. Callers that need prose must surface it plainly ("no
  generation backend configured"); they must never paper over it with a canned
  "could you rephrase?" that impersonates a comprehension problem.
  """
  @behaviour Brain.ML.Generation.Backend

  @impl true
  def name, do: :null

  @impl true
  def children, do: []

  @impl true
  def generate(_messages, _opts \\ []), do: {:error, :no_backend}

  @impl true
  def ready?, do: false
end
