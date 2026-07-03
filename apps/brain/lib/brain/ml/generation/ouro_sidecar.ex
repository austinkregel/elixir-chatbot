defmodule Brain.ML.Generation.OuroSidecar do
  @moduledoc """
  Generation via a self-hosted Ouro model process (`scripts/ouro_server.py`).

  This is the **opt-in**, self-hosted backend — Ouro's recursive-transformer design
  is excellent at composing coherent text under a tight memory budget. But it is a
  multi-GB local process, so it is a deliberate choice, never the default: a Fleet
  should point at ONE shared endpoint rather than launch a sidecar per agent.

      config :brain, :generation, backend: :ouro_sidecar

  This adapter wraps the existing `Brain.ML.Ouro.Model` GenServer and its
  `SidecarLauncher` (which now reuses a healthy sidecar already listening on the
  configured port instead of spawning a redundant one per BEAM). Those two
  processes are only added to the supervision tree when this backend is selected.
  """
  @behaviour Brain.ML.Generation.Backend

  alias Brain.ML.Ouro.Model

  @impl true
  def name, do: :ouro_sidecar

  # Only when Ouro is the chosen backend do its (heavy) processes join the tree.
  @impl true
  def children, do: [Brain.ML.Ouro.Model, Brain.ML.Ouro.SidecarLauncher]

  @impl true
  def generate(messages, opts \\ []) do
    case Model.generate(messages, opts) do
      {:ok, text} -> {:ok, text}
      :fallback -> {:error, :sidecar_not_ready}
      {:error, reason} -> {:error, reason}
      other -> {:error, {:unexpected, other}}
    end
  end

  @impl true
  def ready?, do: Model.ready?()
end
