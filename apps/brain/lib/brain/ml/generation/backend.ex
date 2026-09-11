defmodule Brain.ML.Generation.Backend do
  @moduledoc """
  Contract for a generation backend: turn a realization packet (ChatML
  messages) into prose.

  Generation — turning a realization packet (ChatML messages) into prose — is a
  *pluggable, optional* capability of Brain, not a hard dependency. A backend is
  chosen by configuration (see `Brain.ML.Generation`); Brain itself only does
  analysis. This is what lets a Fleet of many agents share ONE inference endpoint
  instead of each spinning up a heavyweight local model.

  Contract:

    * `generate/2` returns `{:ok, text}` or `{:error, reason}` — the exact shape the
      realizer already expects. A backend that is *configured but unavailable*
      returns `{:error, reason}`; the caller surfaces that loudly (no graceful
      degradation), it must never be disguised as a comprehension failure.
    * `ready?/0` is a cheap liveness probe (used by health/status surfaces).
    * `name/0` identifies the backend for logs and telemetry.
    * `children/0` are the supervision-tree children this backend needs at boot —
      empty for stateless HTTP backends (nothing to launch), so choosing them costs
      no processes and no memory.
  """

  @callback generate(messages :: [map()], opts :: keyword()) ::
              {:ok, String.t()} | {:error, term()}
  @callback ready?() :: boolean()
  @callback name() :: atom()
  @callback children() :: [module() | {module(), term()} | Supervisor.child_spec()]
end

defmodule Brain.ML.Generation.BackendError do
  @moduledoc """
  Raised when a *configured* generation backend cannot produce output.

  This is deliberately loud. A generation outage is a real failure — it is NOT a
  comprehension problem and must never be dressed up as one ("could you rephrase?").
  Surfacing it as an error is the no-graceful-degradation contract: if we cannot
  generate, we say so plainly rather than emitting a plausible-sounding placeholder.
  """
  defexception [:backend, :reason]

  @impl true
  def message(%__MODULE__{backend: backend, reason: reason}) do
    "generation backend #{inspect(backend)} unavailable: #{inspect(reason)} " <>
      "(this is a backend outage, not a comprehension failure)"
  end
end
