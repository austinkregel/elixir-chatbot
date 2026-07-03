defmodule Brain.ML.Generation do
  @moduledoc """
  The generation seam — Brain's pluggable, optional text-generation backend.

  Brain does *analysis*. Turning a realization packet into prose is a separate
  capability supplied by a backend chosen per instance — **not** a hard dependency
  baked into Brain. Not every Brain needs a heavyweight local model: the Fleet
  default points every agent at ONE shared OpenAI-compatible endpoint (Ollama) over
  HTTP, so N agents cost one server, not N × a multi-GB model process (which would
  OOM the machine).

      # default — shared, cheap, nothing launched:
      config :brain, :generation,
        backend: :openai_compatible,
        openai_compatible: [base_url: "http://localhost:11434/v1", model: "llama3.2"]

      # opt-in self-hosted Ouro (one process, reused across the machine):
      config :brain, :generation, backend: :ouro_sidecar

      # analysis-only, no generator:
      config :brain, :generation, backend: :null

  Backends implement `Brain.ML.Generation.Backend`. `generate/2` returns
  `{:ok, text}` | `{:error, reason}`. A *configured* backend that is unavailable
  returns `{:error, reason}`, which the caller surfaces **loudly** — a generation
  outage is never disguised as a comprehension failure (no graceful degradation).
  """

  alias Brain.ML.Generation.{OpenAICompatible, OuroSidecar, Null}

  @default_backend :openai_compatible

  @doc "The `:generation` config (keyword list)."
  def config, do: Application.get_env(:brain, :generation, [])

  @doc "The configured backend key (defaults to `:openai_compatible`)."
  def backend, do: config()[:backend] || @default_backend

  @doc "Resolve a backend key (or a custom module) to its implementation module."
  def backend_module(b \\ backend()) do
    case b do
      :openai_compatible -> OpenAICompatible
      :ouro_sidecar -> OuroSidecar
      :null -> Null
      mod when is_atom(mod) -> mod
    end
  end

  @doc """
  Generate prose from ChatML `messages`. Returns `{:ok, text}` | `{:error, reason}`.
  """
  def generate(messages, opts \\ []), do: backend_module().generate(messages, opts)

  @doc "Cheap liveness probe for the configured backend."
  def ready?, do: backend_module().ready?()

  @doc "The configured backend's name (for logs/telemetry/status)."
  def name, do: backend_module().name()

  @doc """
  Supervision-tree children the configured backend needs at boot. Empty for the
  HTTP backends (nothing to launch) — which is exactly why the default costs no
  processes and no memory. Only `:ouro_sidecar` contributes the heavy Ouro
  processes, and only when it is explicitly chosen.
  """
  def children, do: backend_module().children()
end
