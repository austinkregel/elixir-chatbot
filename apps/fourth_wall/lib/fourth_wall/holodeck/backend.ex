defmodule FourthWall.Holodeck.Backend do
  @moduledoc """
  Contract for a holodeck backend — the thing that actually holds an officer's
  workspace and runs commands in it.

  `FourthWall.Holodeck` owns lifecycle, keying, and the confinement policy; a
  backend owns the mechanism. Splitting them keeps the boundary honest in two
  ways:

    * the real boundary (`Backend.Docker`) is a container with a read-only root,
      a non-root uid, no network, and one writable mount — the kernel, not a
      predicate;
    * everything above it (per-soul keying, path confinement, the tool gate) is
      exercised against `Backend.Memory` in tests without a Docker daemon, and
      against `Backend.Unavailable` by default so nothing runs a container until
      an operator explicitly turns the real backend on.

  A `handle` is whatever the backend needs to address one workspace later (the
  Docker backend uses the container name); the caller treats it as opaque.
  """

  @type soul_id :: String.t()
  @type handle :: term()
  @type abs_path :: String.t()
  @type exec_result :: %{stdout: String.t(), exit_status: non_neg_integer()}

  @doc "Create the workspace for a soul, returning an opaque handle for it."
  @callback create(soul_id(), keyword()) :: {:ok, handle()} | {:error, term()}

  @doc "Tear the workspace down, reclaiming any storage it held."
  @callback destroy(handle()) :: :ok | {:error, term()}

  @doc "Read a file at an already-confined absolute path."
  @callback read(handle(), abs_path()) :: {:ok, String.t()} | {:error, term()}

  @doc "Write a file at an already-confined absolute path, creating parents."
  @callback write(handle(), abs_path(), String.t()) :: :ok | {:error, term()}

  @doc "List a directory at an already-confined absolute path."
  @callback list(handle(), abs_path()) :: {:ok, String.t()} | {:error, term()}

  @doc "Run an already-validated argv in the workspace, returning its output."
  @callback exec(handle(), [String.t()]) :: {:ok, exec_result()} | {:error, term()}
end
