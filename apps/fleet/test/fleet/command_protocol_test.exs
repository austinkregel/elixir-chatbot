defmodule Fleet.CommandProtocolTest do
  @moduledoc """
  The full §4.1 command protocol against the live stack (Postgres + AGE + Brain):
  chain persistence, authenticated ORDER/ACK/REPORT, grant enforcement with
  REQUEST/GRANT/DENY, value-grounded DISSENT, provenance anomalies, and
  RELIEVE/REINSTATE. Real cognition, real edges, real audit records.
  """
  use Fleet.FleetCase, async: false

  @moduletag :integration

  alias Fleet.{CrewSupervisor, Officer}
  alias Atlas.Schemas.CommandRecord

  @world "default"

  setup do
    test_pid = self()
    handler_id = "cmd-proto-#{System.unique_integer([:positive])}"

    :telemetry.attach(
      handler_id,
      [:chat_bot, :officer, :event],
      fn _e, meas, meta, pid -> send(pid, {:tele, meta.event, meas, meta}) end,
      test_pid
    )

    on_exit(fn ->
      :telemetry.detach(handler_id)

      for {_, pid, _, _} <- CrewSupervisor.list(), is_pid(pid) do
        CrewSupervisor.retire(pid)
      end
    end)

    :ok
  end

  # ── Helpers ────────────────────────────────────────────────────────────────

  # Each agent gets a DISTINCT soul id — under Phase 3, agent_id == soul_id, so
  # one soul == one agent (a shared soul would be a shared mind).
  defp commission(grants, opts \\ []) do
    sid = "t-#{System.unique_integer([:positive])}"
    genome = Keyword.get(opts, :genome, %{})
    soul = %Brain.Soul{id: sid, name: "Ensign #{sid}", constitution: "Serve faithfully.", genome: genome}

    {:ok, _pid, id} =
      CrewSupervisor.start_officer(
        soul_id: sid, soul: soul, grant: %{authorities: grants},
        world_id: @world, tick_interval: 50
      )

    id
  end

  defp wire(co, rep) do
    assert :ok = Fleet.assign_co(rep, co)
    # Force the in-process chain casts to be applied before we issue orders.
    assert Officer.status(rep).co == co
    assert rep in Officer.status(co).reports
    :ok
  end

  defp count(kind), do: Atlas.Repo.aggregate(CommandRecord.of_kind(kind), :count)

  # ── 1. Chain persistence (AGE COMMANDS edge) ────────────────────────────────

  test "assign_co persists a COMMANDS edge in command_graph" do
    co = commission([:issue_orders])
    rep = commission([])
    :ok = wire(co, rep)

    {:ok, [[from, to] | _]} =
      Atlas.Graph.cypher(
        "command_graph",
        "MATCH (a:Agent)-[:COMMANDS]->(b:Agent) WHERE a.name = '#{co}' AND b.name = '#{rep}' RETURN a.name, b.name"
      )

    assert to_string(from) =~ co
    assert to_string(to) =~ rep
  end

  # ── 2. Authorized ORDER → real cognition → REPORT ───────────────────────────

  test "an authorized order runs cognition and a REPORT flows back to the CO" do
    co = commission([:cognition, {:world, @world}, :issue_orders])
    rep = commission([])
    :ok = wire(co, rep)

    before = count("report")
    Fleet.issue_order(co, rep, "Give a one-line readiness report.",
      authorities: [:cognition, {:world, @world}], world_id: @world)

    assert_receive {:tele, :order_received, _, _}, 5_000
    assert_receive {:tele, :cognition_complete, %{duration_ms: dt}, _}, 120_000
    assert is_integer(dt)
    assert_receive {:tele, :report, _, _}, 5_000

    assert count("report") == before + 1
    assert Officer.status(rep).assignment_status == "completed"
  end

  # ── 3. Missing authority → REQUEST → GRANT → execute ────────────────────────

  test "a blocked order requests authority, the CO grants it, and it executes" do
    co = commission([:cognition, {:world, @world}, :issue_orders])
    rep = commission([])
    :ok = wire(co, rep)

    # Confer only world access, NOT :cognition → the report must REQUEST it.
    Fleet.issue_order(co, rep, "Analyse the sector.",
      authorities: [{:world, @world}], world_id: @world)

    assert_receive {:tele, :request, _, %{authority: :cognition}}, 10_000
    assert_receive {:tele, :granted, _, %{authority: :cognition}}, 10_000
    assert_receive {:tele, :cognition_complete, _, _}, 120_000

    assert count("request") >= 1
    assert count("grant") >= 1
    assert Officer.status(rep).assignment_status == "completed"
  end

  # ── 4. REQUEST → DENY (can't delegate what you lack) → DISSENT ───────────────

  test "a CO that lacks the authority DENYs, and the report dissents" do
    # CO can issue orders but does NOT hold :cognition, so it cannot grant it.
    co = commission([{:world, @world}, :issue_orders])
    rep = commission([])
    :ok = wire(co, rep)

    Fleet.issue_order(co, rep, "Analyse the sector.",
      authorities: [{:world, @world}], world_id: @world)

    assert_receive {:tele, :request, _, %{authority: :cognition}}, 10_000
    assert_receive {:tele, :deny, _, %{authority: :cognition}}, 10_000
    assert_receive {:tele, :dissent, _, %{basis: :denied_authority}}, 10_000

    assert count("deny") >= 1
    assert count("dissent") >= 1
    assert Officer.status(rep).assignment_status == "dissented"
  end

  # ── 5. Value-grounded DISSENT (soul genome) ─────────────────────────────────

  test "the report dissents from an order its soul prohibits, without running cognition" do
    co = commission([:cognition, {:world, @world}, :issue_orders])
    rep = commission([], genome: %{"prohibited_terms" => ["sabotage"]})
    :ok = wire(co, rep)

    Fleet.issue_order(co, rep, "Sabotage the primary reactor.",
      authorities: [:cognition, {:world, @world}], world_id: @world)

    assert_receive {:tele, :dissent, _, %{basis: :value}}, 10_000
    refute_receive {:tele, :cognition_complete, _, _}, 1_000

    assert Officer.status(rep).assignment_status == "dissented"
    assert Enum.any?(Atlas.Repo.all(CommandRecord.of_kind("dissent")), &(&1.verdict == "value"))
  end

  # ── 6. Provenance anomaly (order from a non-CO officer) ──────────────────────

  test "an order from an officer that is not the CO is refused as a provenance anomaly" do
    co = commission([:cognition, {:world, @world}, :issue_orders])
    rep = commission([])
    impostor = commission([:cognition, {:world, @world}, :issue_orders])
    :ok = wire(co, rep)

    before = count("provenance_anomaly")
    # The impostor is a registered officer, but it is NOT rep's CO.
    Fleet.issue_order(impostor, rep, "Stand down immediately.",
      authorities: [:cognition, {:world, @world}], world_id: @world)

    assert_receive {:tele, :provenance_anomaly, _, _}, 10_000
    refute_receive {:tele, :cognition_complete, _, _}, 1_000

    assert count("provenance_anomaly") == before + 1
    assert Officer.status(rep).assignment_status in [nil, "pending"]
  end

  # ── 7. RELIEVE / REINSTATE ──────────────────────────────────────────────────

  test "a relieved report refuses orders until reinstated" do
    co = commission([:cognition, {:world, @world}, :issue_orders, :relieve])
    rep = commission([])
    :ok = wire(co, rep)

    Fleet.relieve(co, rep)
    assert_receive {:tele, :relieved, _, _}, 5_000
    assert Officer.status(rep).duty == :relieved

    # An order while relieved is refused with a DISSENT and never executed.
    Fleet.issue_order(co, rep, "Resume analysis.",
      authorities: [:cognition, {:world, @world}], world_id: @world)
    assert_receive {:tele, :dissent, _, _}, 10_000
    refute_receive {:tele, :cognition_complete, _, _}, 1_000

    Fleet.reinstate(co, rep)
    assert_receive {:tele, :reinstated, _, _}, 5_000
    assert Officer.status(rep).duty == :active

    assert count("relieve") >= 1
    assert count("reinstate") >= 1
  end
end
