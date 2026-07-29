defmodule Fleet.HailTest do
  @moduledoc """
  Proves HAIL is conversation, not command: hailing an agent runs its soul's
  cognition and returns an answer to the caller, but creates no assignment, confers
  no authority, and works even when the agent is relieved of duty.
  """
  use Fleet.FleetCase, async: false

  @moduletag :integration

  alias Fleet.CrewSupervisor

  setup do
    test_pid = self()
    handler_id = "hail-#{System.unique_integer([:positive])}"

    :telemetry.attach(handler_id, [:chat_bot, :officer, :event],
      fn _e, meas, meta, pid -> send(pid, {:tele, meta.event, meas, meta}) end, test_pid)

    on_exit(fn ->
      :telemetry.detach(handler_id)
      for {_, pid, _, _} <- CrewSupervisor.list(), is_pid(pid), do: CrewSupervisor.retire(pid)
    end)

    :ok
  end

  defp commission do
    sid = "hail-#{System.unique_integer([:positive])}"
    soul = %Brain.Soul{id: sid, name: "Agent #{sid}", constitution: "You are helpful.", genome: %{}}
    {:ok, _pid, id} = CrewSupervisor.start_officer(soul_id: sid, soul: soul, tick_interval: 50)
    assert_receive {:tele, :soul_hydrated, _, %{soul_id: ^sid}}, 5_000
    id
  end

  test "hailing returns a reply and creates no assignment" do
    id = commission()

    assert {:ok, reply} = Fleet.hail_sync(id, "Who are you?", 120_000)
    assert reply.agent_id == id
    assert reply.question == "Who are you?"
    # A reply carries either an answer (string) or a surfaced error — never silence.
    assert is_binary(reply[:answer]) or is_binary(reply[:error])

    # The defining property: conversation left no assignment behind.
    status = Fleet.status(id)
    assert is_nil(status.directive)
    assert is_nil(status.assignment_status)
  end

  test "a relieved agent can still be hailed — talking is not duty" do
    id = commission()
    Fleet.relieve(id)
    assert_receive {:tele, :relieved, _, _}, 5_000

    assert {:ok, reply} = Fleet.hail_sync(id, "Report your status.", 120_000)
    assert reply.agent_id == id
    assert is_binary(reply[:answer]) or is_binary(reply[:error])
  end

  test "the hail is audited as a conversation, not an order" do
    id = commission()
    assert {:ok, _reply} = Fleet.hail_sync(id, "Standing by?", 120_000)

    # A HAIL record was written; no ORDER/assignment record was.
    kinds =
      Atlas.Schemas.CommandRecord
      |> Atlas.Repo.all()
      |> Enum.filter(&(&1.from_agent == "admiral" or &1.to_agent == id or &1.from_agent == id))
      |> Enum.map(& &1.kind)

    assert "hail" in kinds
    refute "order" in kinds
  end
end
