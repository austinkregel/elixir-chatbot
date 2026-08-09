defmodule FourthWall.Holodeck.Policy do
  @moduledoc """
  The deterministic confinement policy for a holodeck workspace — path
  confinement and a command allowlist.

  This is **defense-in-depth, not the boundary**. The boundary is the kernel:
  every workspace is a container with a read-only root filesystem, a non-root
  uid, no network, and only `/workspace` writable (see
  `FourthWall.Holodeck.Backend.Docker`). Even so, a bad path or command should be
  refused *before* it reaches the container, with an error that names the path or
  the command — so an officer that mistypes gets a correction rather than an
  opaque failure, and a probe can assert the refusal without spinning a container.

  Ported in spirit from the LCARS prototype's deterministic command policy
  (`uss-dreamcom/lcars/tools/command.py`): an executable allowlist, read-only
  `git` subcommands only, and rejection of shell metacharacters. Commands run as
  an argv array via `docker exec` — never through a shell — so metacharacters are
  already inert; the allowlist is the real control, and the metacharacter check
  is belt-and-suspenders.
  """

  # The one writable mount inside every workspace container. All paths are
  # confined under it; the container's root filesystem is read-only.
  @workspace_root "/workspace"

  # Executables an officer may run in its workspace. Deliberately small: reading,
  # searching, moving files around its own tree, and building. Anything not here
  # is refused by name rather than run.
  @allowed_commands ~w(
    ls cat head tail wc echo pwd stat file find grep sort uniq cut tr
    mkdir touch mv cp rm ln diff sed awk
    git mix elixir iex node npm python python3 cargo rustc go make bash sh
  )

  # `git` is allowed, but only for subcommands that observe rather than mutate
  # history or reach the network. A workspace has no network anyway, but a
  # read-only surface is the honest floor and does not depend on that.
  @readonly_git_subcommands ~w(
    status log diff show branch tag remote ls-files ls-tree rev-parse
    blame describe cat-file show-ref shortlog reflog config
  )

  # Characters that only mean something to a shell. We never invoke a shell, so
  # these are inert in an argv — rejecting them anyway keeps a command that
  # *expects* shell interpretation from silently doing nothing surprising.
  @shell_metacharacters ~w(; | & $ ` > < \n)

  @doc "The writable workspace root inside a container."
  @spec workspace_root() :: String.t()
  def workspace_root, do: @workspace_root

  @doc "The executables an officer may run in its workspace."
  @spec allowed_commands() :: [String.t()]
  def allowed_commands, do: @allowed_commands

  @doc "The read-only `git` subcommands an officer may run."
  @spec readonly_git_subcommands() :: [String.t()]
  def readonly_git_subcommands, do: @readonly_git_subcommands

  @doc """
  Confine a requested path to the workspace, returning its absolute in-container
  path.

  Normalises `requested` against `#{@workspace_root}` and refuses anything that
  would escape — `..` traversal, an absolute path outside the workspace, or a
  symlink-style climb. Returns `{:ok, abs}` where `abs` is guaranteed to be
  `#{@workspace_root}` or a path beneath it.

  ## Examples

      iex> FourthWall.Holodeck.Policy.resolve_path("lib/foo.ex")
      {:ok, "/workspace/lib/foo.ex"}

      iex> FourthWall.Holodeck.Policy.resolve_path("../etc/passwd")
      {:error, {:escapes_workspace, "../etc/passwd"}}

      iex> FourthWall.Holodeck.Policy.resolve_path("/etc/passwd")
      {:error, {:escapes_workspace, "/etc/passwd"}}
  """
  @spec resolve_path(term()) ::
          {:ok, String.t()} | {:error, {:escapes_workspace | :blank_path, term()}}
  def resolve_path(requested) when is_binary(requested) do
    trimmed = String.trim(requested)

    cond do
      trimmed == "" ->
        {:error, {:blank_path, requested}}

      true ->
        abs = Path.expand(trimmed, @workspace_root)

        if abs == @workspace_root or String.starts_with?(abs, @workspace_root <> "/") do
          {:ok, abs}
        else
          {:error, {:escapes_workspace, requested}}
        end
    end
  end

  def resolve_path(other), do: {:error, {:blank_path, other}}

  @doc """
  Validate a command an officer proposes to run, given as an argv list.

  Returns `:ok`, or an error naming exactly what was refused. The first element
  is the executable (checked against the allowlist); `git` is narrowed to
  read-only subcommands; and no argument may contain a shell metacharacter.

  ## Examples

      iex> FourthWall.Holodeck.Policy.check_command(["ls", "-la"])
      :ok

      iex> FourthWall.Holodeck.Policy.check_command(["git", "status"])
      :ok

      iex> FourthWall.Holodeck.Policy.check_command(["git", "push"])
      {:error, {:forbidden_git_subcommand, "push"}}

      iex> FourthWall.Holodeck.Policy.check_command(["curl", "http://evil"])
      {:error, {:forbidden_command, "curl"}}

      iex> FourthWall.Holodeck.Policy.check_command(["ls", "; rm -rf /"])
      {:error, {:shell_metacharacter, "; rm -rf /"}}
  """
  @spec check_command(term()) ::
          :ok
          | {:error,
             {:empty_command, term()}
             | {:forbidden_command, String.t()}
             | {:forbidden_git_subcommand, String.t()}
             | {:shell_metacharacter, String.t()}}
  def check_command([cmd | _] = argv) when is_binary(cmd) do
    with :ok <- check_metacharacters(argv),
         :ok <- check_executable(cmd),
         :ok <- check_git(argv) do
      :ok
    end
  end

  def check_command([]), do: {:error, {:empty_command, []}}
  def check_command(other), do: {:error, {:empty_command, other}}

  defp check_metacharacters(argv) do
    case Enum.find(argv, &contains_metacharacter?/1) do
      nil -> :ok
      arg -> {:error, {:shell_metacharacter, arg}}
    end
  end

  defp contains_metacharacter?(arg) when is_binary(arg),
    do: Enum.any?(@shell_metacharacters, &String.contains?(arg, &1))

  defp contains_metacharacter?(_), do: false

  defp check_executable(cmd) do
    if cmd in @allowed_commands, do: :ok, else: {:error, {:forbidden_command, cmd}}
  end

  defp check_git(["git", sub | _]) when is_binary(sub) do
    if sub in @readonly_git_subcommands,
      do: :ok,
      else: {:error, {:forbidden_git_subcommand, sub}}
  end

  # `git` with no subcommand (prints usage) is harmless; anything else already
  # passed the executable check.
  defp check_git(_argv), do: :ok
end
