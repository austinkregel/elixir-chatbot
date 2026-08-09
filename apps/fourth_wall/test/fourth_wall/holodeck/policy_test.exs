defmodule FourthWall.Holodeck.PolicyTest do
  use ExUnit.Case, async: true

  alias FourthWall.Holodeck.Policy

  doctest FourthWall.Holodeck.Policy

  describe "resolve_path/1 confines to the workspace" do
    test "a plain relative path resolves under /workspace" do
      assert {:ok, "/workspace/lib/foo.ex"} = Policy.resolve_path("lib/foo.ex")
      assert {:ok, "/workspace"} = Policy.resolve_path(".")
    end

    test "traversal out of the workspace is refused" do
      for escape <- [
            "../etc/passwd",
            "foo/../../bar",
            "./../..",
            "lib/../../secret",
            "a/b/../../../x"
          ] do
        assert {:error, {:escapes_workspace, ^escape}} = Policy.resolve_path(escape)
      end
    end

    test "an absolute path outside the workspace is refused" do
      assert {:error, {:escapes_workspace, "/etc/passwd"}} = Policy.resolve_path("/etc/passwd")
    end

    test "an absolute path that is already inside the workspace is kept" do
      assert {:ok, "/workspace/lib/x.ex"} = Policy.resolve_path("/workspace/lib/x.ex")
    end

    test "blank or non-binary paths are refused, not treated as the root" do
      assert {:error, {:blank_path, _}} = Policy.resolve_path("")
      assert {:error, {:blank_path, _}} = Policy.resolve_path("   ")
      assert {:error, {:blank_path, _}} = Policy.resolve_path(nil)
    end
  end

  describe "check_command/1 gates what may run" do
    test "an allowlisted command passes" do
      assert :ok = Policy.check_command(["ls", "-la"])
      assert :ok = Policy.check_command(["cat", "README.md"])
    end

    test "a command not on the allowlist is refused by name" do
      assert {:error, {:forbidden_command, "curl"}} = Policy.check_command(["curl", "http://x"])
      assert {:error, {:forbidden_command, "wget"}} = Policy.check_command(["wget", "x"])
    end

    test "git is limited to read-only subcommands" do
      assert :ok = Policy.check_command(["git", "status"])
      assert :ok = Policy.check_command(["git", "log", "--oneline"])
      assert {:error, {:forbidden_git_subcommand, "push"}} = Policy.check_command(["git", "push"])

      assert {:error, {:forbidden_git_subcommand, "commit"}} =
               Policy.check_command(["git", "commit"])
    end

    test "shell metacharacters are refused even in an argv (defense-in-depth)" do
      assert {:error, {:shell_metacharacter, _}} = Policy.check_command(["ls", "; rm -rf /"])
      assert {:error, {:shell_metacharacter, _}} = Policy.check_command(["cat", "a | b"])
      assert {:error, {:shell_metacharacter, _}} = Policy.check_command(["echo", "$(whoami)"])
    end

    test "the metacharacter check outranks the allowlist" do
      # A forbidden command with a metacharacter is caught as the metacharacter
      # first — the argv is refused whole, whichever reason lands first.
      assert {:error, {:shell_metacharacter, _}} = Policy.check_command(["curl", "http://x; ls"])
    end

    test "an empty command is refused" do
      assert {:error, {:empty_command, _}} = Policy.check_command([])
      assert {:error, {:empty_command, _}} = Policy.check_command("not a list")
    end
  end
end
