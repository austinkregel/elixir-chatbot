defmodule Fleet.WorkspaceToolTest do
  use ExUnit.Case, async: false

  alias Fleet.{Authority, Proposal, Tool}
  alias FourthWall.Holodeck

  describe "the registry" do
    test "workspace reads carry no info_class, so an officer reads its own space without clearance" do
      assert %Tool{effect: :read, info_class: nil} = Tool.registry()["workspace.read"]
      assert %Tool{effect: :read, info_class: nil} = Tool.registry()["workspace.list"]
    end

    test "writes and shell.run are :mutate, authority-gated" do
      assert %Tool{effect: :mutate} = Tool.registry()["workspace.write"]
      assert %Tool{effect: :mutate} = Tool.registry()["shell.run"]
    end

    test "no workspace tool leaves the ship (no egress)" do
      for name <- ~w(workspace.read workspace.list workspace.write shell.run) do
        assert Tool.registry()[name].egress == []
      end
    end

    test "descriptions never mention how the workspace is confined" do
      for name <- ~w(workspace.read workspace.list workspace.write shell.run) do
        desc = Tool.registry()[name].description
        refute desc =~ ~r/container|docker|sandbox|chroot/i
      end
    end
  end

  describe "the gate (pure decide/2)" do
    test "a write is refused without its authority and allowed with it" do
      proposal = %Proposal{
        tool: "workspace.write",
        requirement: "save my work",
        args: %{"path" => "a.ex", "content" => "x"}
      }

      assert {:refuse, {:ungranted, _}} = Fleet.Dispatcher.decide(proposal, MapSet.new())

      grants = MapSet.new([Authority.tool("workspace.write")])
      assert {:allow, %Tool{name: "workspace.write"}} = Fleet.Dispatcher.decide(proposal, grants)
    end

    test "a malformed write (missing content) is a malformed_call, not a handler fault" do
      proposal = %Proposal{
        tool: "workspace.write",
        requirement: "save my work",
        args: %{"path" => "a.ex"}
      }

      grants = MapSet.new([Authority.tool("workspace.write")])
      assert {:refuse, {:malformed_call, errors}} = Fleet.Dispatcher.decide(proposal, grants)
      assert {:missing, "content"} in errors
    end
  end

  describe "the handlers when the holodeck is unavailable" do
    test "an honest refusal, not a fake write" do
      # Fleet.Application starts the holodeck with the Unavailable backend by
      # default, so with no override the handler reports the capability is off.
      write = Tool.registry()["workspace.write"].handler

      assert {:error, :holodeck_unavailable} =
               write.(%{"path" => "a.ex", "content" => "x"}, %{soul_id: "s1"})
    end
  end

  describe "the handlers key on the officer's own soul" do
    setup do
      name = :"holodeck_ws_tool_#{System.unique_integer([:positive])}"
      {:ok, _pid} = Holodeck.start_link(backend: Fleet.WorkspaceTestBackend, name: name)
      Process.put(:holodeck_server, name)
      on_exit(fn -> Process.delete(:holodeck_server) end)
      :ok
    end

    test "a write then read round-trips for the same soul" do
      write = Tool.registry()["workspace.write"].handler
      read = Tool.registry()["workspace.read"].handler

      assert {:ok, %{written: "lib/a.ex"}} =
               write.(%{"path" => "lib/a.ex", "content" => "hello"}, %{soul_id: "sa"})

      assert {:ok, "hello"} = read.(%{"path" => "lib/a.ex"}, %{soul_id: "sa"})
    end

    test "another officer cannot read the first's file" do
      write = Tool.registry()["workspace.write"].handler
      read = Tool.registry()["workspace.read"].handler

      write.(%{"path" => "secret", "content" => "mine"}, %{soul_id: "owner"})
      assert {:error, {:read_failed, _}} = read.(%{"path" => "secret"}, %{soul_id: "intruder"})
    end

    test "identity falls back to agent_id for a soul-less officer" do
      write = Tool.registry()["workspace.write"].handler
      read = Tool.registry()["workspace.read"].handler

      write.(%{"path" => "f", "content" => "v"}, %{soul_id: nil, agent_id: "anon"})
      assert {:ok, "v"} = read.(%{"path" => "f"}, %{soul_id: nil, agent_id: "anon"})
    end

    test "shell.run validates the command before executing" do
      run = Tool.registry()["shell.run"].handler
      assert {:ok, %{exit_status: 0}} = run.(%{"command" => ["ls", "-la"]}, %{soul_id: "sa"})

      assert {:error, {:forbidden_command, "curl"}} =
               run.(%{"command" => ["curl", "x"]}, %{soul_id: "sa"})
    end
  end
end
