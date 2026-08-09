defmodule FourthWall.HolodeckTest do
  use ExUnit.Case, async: false

  alias FourthWall.Holodeck
  alias FourthWall.Holodeck.Backend.Memory

  setup do
    # A private Holodeck backed by the in-memory backend, addressed through the
    # process-dictionary override the client honours. No Docker, real behaviour.
    name = :"holodeck_test_#{System.unique_integer([:positive])}"
    {:ok, pid} = Holodeck.start_link(backend: Memory, name: name)
    Process.put(:holodeck_server, name)
    on_exit(fn -> Process.delete(:holodeck_server) end)
    %{pid: pid, soul: "soul-#{System.unique_integer([:positive])}"}
  end

  describe "ensure/1" do
    test "creates a workspace and is idempotent", %{soul: soul} do
      assert {:ok, ctx1} = Holodeck.ensure(soul)
      assert {:ok, ctx2} = Holodeck.ensure(soul)
      assert ctx1.handle == ctx2.handle
    end

    test "refuses a soul-less officer rather than keying on nil" do
      assert {:error, :no_workspace_identity} = Holodeck.ensure(nil)
    end
  end

  describe "read/write/list round-trip" do
    test "a file written is a file read", %{soul: soul} do
      assert :ok = Holodeck.write(soul, "lib/foo.ex", "defmodule Foo")
      assert {:ok, "defmodule Foo"} = Holodeck.read(soul, "lib/foo.ex")
    end

    test "list shows what was written", %{soul: soul} do
      :ok = Holodeck.write(soul, "lib/a.ex", "a")
      :ok = Holodeck.write(soul, "lib/b.ex", "b")
      assert {:ok, listing} = Holodeck.list(soul, "lib")
      assert listing =~ "/workspace/lib/a.ex"
      assert listing =~ "/workspace/lib/b.ex"
    end

    test "a path escaping the workspace is refused before the backend", %{soul: soul} do
      assert {:error, {:escapes_workspace, "../x"}} = Holodeck.write(soul, "../x", "y")
      assert {:error, {:escapes_workspace, "/etc/passwd"}} = Holodeck.read(soul, "/etc/passwd")
    end
  end

  describe "run/2" do
    test "a validated command reaches the backend", %{soul: soul} do
      assert {:ok, %{stdout: "ls -la", exit_status: 0}} = Holodeck.run(soul, ["ls", "-la"])
    end

    test "a forbidden command is refused without creating a workspace", %{soul: soul} do
      assert {:error, {:forbidden_command, "curl"}} = Holodeck.run(soul, ["curl", "http://x"])
    end
  end

  describe "isolation between souls" do
    test "one soul cannot see another's files" do
      :ok = Holodeck.write("soul-a", "secret.txt", "a's secret")
      assert {:ok, "a's secret"} = Holodeck.read("soul-a", "secret.txt")
      assert {:error, {:read_failed, _}} = Holodeck.read("soul-b", "secret.txt")
    end
  end

  describe "reap/1" do
    test "tears down a soul's workspace", %{soul: soul} do
      :ok = Holodeck.write(soul, "f.txt", "data")
      assert :ok = Holodeck.reap(soul)
      # After reap the files are gone; a fresh ensure sees an empty workspace.
      assert {:error, {:read_failed, _}} = Holodeck.read(soul, "f.txt")
    end

    test "is safe to call for a soul that never had a workspace" do
      assert :ok = Holodeck.reap("never-existed-#{System.unique_integer([:positive])}")
    end
  end
end
