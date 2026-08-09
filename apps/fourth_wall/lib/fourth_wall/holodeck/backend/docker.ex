defmodule FourthWall.Holodeck.Backend.Docker do
  @moduledoc """
  The real holodeck boundary: one Docker container per soul.

  Every workspace is a container run with a **read-only root filesystem**, a
  **non-root uid** (the image's `officer` user), **no network** (`--network
  none`), the repository mounted **read-only** at `/repo`, and a single writable
  mount — a per-soul named volume at `/workspace`. The container's only job is to
  stay alive (`sleep infinity`) so the ship can `docker exec` into it.

  The isolation is the container's, not this module's: nothing here trusts a
  path or a command, but nothing here is what stops an escape either — the kernel
  is. `FourthWall.Holodeck.Policy` still validates paths and commands first, as
  defense-in-depth and to give named refusals.

  Uses the `docker` CLI via `System.cmd/3` (the same shape as
  `Brain.ML.Ouro.SidecarLauncher`), so the ship needs the Docker socket. The
  socket is root-equivalent and belongs to the ship alone — the officer only ever
  proposes tools and never sees it.

  Ownership note: the container runs as the image's non-root `officer` user, and
  `/workspace` is owned by that user in the image, so a fresh named volume
  inherits that ownership on first mount and the officer can write to it. We do
  **not** pass `--user`, which would break that.
  """

  @behaviour FourthWall.Holodeck.Backend

  require Logger

  @type handle :: %{soul_id: String.t(), container: String.t(), volume: String.t()}

  @impl true
  def create(soul_id, opts) when is_binary(soul_id) do
    handle = %{
      soul_id: soul_id,
      container: container_name(soul_id),
      volume: volume_name(soul_id)
    }

    case ensure_running(handle, opts) do
      :ok -> {:ok, handle}
      {:error, _} = error -> error
    end
  end

  @impl true
  def destroy(%{container: container, volume: volume}) do
    # Best-effort and order-tolerant: remove the container, then its volume. A
    # missing container is success (already reaped); a volume still attached to
    # something is left rather than forced.
    _ = docker(["rm", "-f", container])
    _ = docker(["volume", "rm", volume])
    :ok
  end

  @impl true
  def read(%{container: container}, abs_path) do
    case docker(["exec", container, "cat", abs_path]) do
      {:ok, content} -> {:ok, content}
      {:error, {:docker_failed, _status, out}} -> {:error, {:read_failed, String.trim(out)}}
      other -> other
    end
  end

  @impl true
  def write(%{container: container}, abs_path, content) when is_binary(content) do
    dir = Path.dirname(abs_path)
    tmp = Path.join(System.tmp_dir!(), "holodeck-" <> random_suffix())

    try do
      File.write!(tmp, content)

      with {:ok, _} <- docker(["exec", container, "mkdir", "-p", dir]),
           {:ok, _} <- docker(["cp", tmp, container <> ":" <> abs_path]) do
        :ok
      else
        {:error, {:docker_failed, _status, out}} -> {:error, {:write_failed, String.trim(out)}}
        other -> other
      end
    after
      File.rm(tmp)
    end
  end

  @impl true
  def list(%{container: container}, abs_path) do
    case docker(["exec", container, "ls", "-la", abs_path]) do
      {:ok, listing} -> {:ok, listing}
      {:error, {:docker_failed, _status, out}} -> {:error, {:list_failed, String.trim(out)}}
      other -> other
    end
  end

  @impl true
  def exec(%{container: container}, argv) when is_list(argv) do
    # A non-zero exit is the *command's* result, not a fault — `ls missing` is a
    # legitimate answer. We return stdout and the status; only a failure to reach
    # the container is an error, and `Holodeck` guarantees it exists first.
    {out, status} =
      System.cmd("docker", ["exec", "-w", "/workspace", container | argv], stderr_to_stdout: true)

    {:ok, %{stdout: out, exit_status: status}}
  rescue
    e -> {:error, {:docker_unavailable, Exception.message(e)}}
  end

  # ── Container lifecycle ────────────────────────────────────────────────────

  defp ensure_running(%{container: container} = handle, opts) do
    case docker(["inspect", "-f", "{{.State.Running}}", container]) do
      {:ok, "true" <> _} -> :ok
      {:ok, _stopped} -> start_ok(container)
      {:error, _absent} -> run_container(handle, opts)
    end
  end

  defp start_ok(container) do
    case docker(["start", container]) do
      {:ok, _} -> :ok
      error -> error
    end
  end

  defp run_container(%{container: container, volume: volume}, opts) do
    image = Keyword.fetch!(opts, :image)
    repo = Keyword.fetch!(opts, :repo_path)
    network = Keyword.get(opts, :network, "none")

    args =
      [
        "run",
        "-d",
        "--name",
        container,
        "--read-only",
        "--network",
        network,
        "--workdir",
        "/workspace",
        "--tmpfs",
        "/tmp",
        "--env",
        "HOME=/workspace",
        "-v",
        repo <> ":/repo:ro",
        "-v",
        volume <> ":/workspace",
        image,
        "sleep",
        "infinity"
      ]

    case docker(args) do
      {:ok, _id} ->
        :ok

      {:error, {:docker_failed, _status, out}} ->
        Logger.error("Holodeck.Docker: failed to start #{container} — #{String.trim(out)}")
        {:error, {:create_failed, String.trim(out)}}

      other ->
        other
    end
  end

  # ── Naming ─────────────────────────────────────────────────────────────────

  @doc false
  def container_name(soul_id), do: "holodeck-" <> sanitize(soul_id)

  @doc false
  def volume_name(soul_id), do: "holodeck-ws-" <> sanitize(soul_id)

  # Docker names allow [a-zA-Z0-9][a-zA-Z0-9_.-]+. Fold anything else to `-` so a
  # soul id with a colon or slash still yields a legal, stable name.
  defp sanitize(soul_id) do
    soul_id
    |> String.replace(~r/[^a-zA-Z0-9_.-]/, "-")
    |> String.trim_leading("-")
  end

  # ── docker CLI ─────────────────────────────────────────────────────────────

  defp docker(args) do
    {out, status} = System.cmd("docker", args, stderr_to_stdout: true)

    case status do
      0 -> {:ok, String.trim(out)}
      _ -> {:error, {:docker_failed, status, out}}
    end
  rescue
    e -> {:error, {:docker_unavailable, Exception.message(e)}}
  end

  defp random_suffix, do: :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
end
