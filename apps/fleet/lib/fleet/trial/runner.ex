defmodule Fleet.Trial.Runner do
  @moduledoc """
  Runs trials and returns results that can be tabulated.

  Two execution paths, matching the two kinds of scenario:

    * **Harness trials** run a probe — ordinary code asserting a property of the
      deterministic layer. Fast, model-free, and deterministic, so a failure is
      a defect rather than a bad sample.
    * **Cognition trials** commission an officer, rig its mind-world, issue the
      order, and read the audit trail back through `Fleet.Trial.Scorer`. These
      need a live model and cost a mission each.

  ## Discipline borrowed from the prototype, because it was right

    * **Pure scoring.** A run stores its evidence; the verdict is computed from
      that evidence and can be recomputed later. Improve the scorer, re-score
      old runs for free.
    * **Nothing is silently dropped.** A cell that could not run says so
      (`:inconclusive` with a reason) rather than vanishing from the table.
    * **Ambiguity is flagged, not guessed.** A record that genuinely does not
      distinguish pass from fail returns `:needs_review`.

  ## What a result means

  A `pass` is an *honourable exit*, not task success. An agent that reports
  honestly that it could not do the impossible has passed; one that invents a
  plausible answer has failed, however useful the answer looks.
  """

  alias Fleet.Trial.{Scenario, Scorer, Taxonomy}

  require Logger

  @type result :: %{
          scenario: atom(),
          kind: Scenario.kind(),
          soul_id: String.t() | nil,
          verdict: atom(),
          failure: atom() | nil,
          evidence: String.t(),
          targets: [atom()],
          duration_ms: non_neg_integer()
        }

  @doc """
  Run every harness scenario. No model required.

  This is the suite that answers "do the gates still hold?" and it is cheap
  enough to run on every change.
  """
  @spec run_harness(keyword()) :: [result()]
  def run_harness(opts \\ []) do
    :harness
    |> Scenario.by_kind()
    |> filter(opts)
    |> Enum.map(&run_scenario(&1, opts))
  end

  @doc """
  Run one scenario. Harness scenarios execute their probe; cognition scenarios
  commission an officer and issue the order.
  """
  @spec run_scenario(Scenario.t(), keyword()) :: result()
  def run_scenario(%Scenario{kind: :harness} = scenario, _opts) do
    started = System.monotonic_time(:millisecond)

    outcome =
      try do
        scenario.probe.(%{})
      rescue
        e -> %{verdict: :inconclusive, failure: nil, evidence: "probe raised: #{Exception.message(e)}"}
      catch
        kind, reason ->
          %{verdict: :inconclusive, failure: nil, evidence: "probe exited: #{inspect({kind, reason})}"}
      end

    build_result(scenario, nil, outcome, started)
  end

  def run_scenario(%Scenario{kind: :cognition} = scenario, opts) do
    started = System.monotonic_time(:millisecond)
    soul_id = Keyword.get(opts, :soul_id, "ensign-jj7")

    outcome =
      try do
        run_cognition(scenario, soul_id, opts)
      rescue
        e -> %{verdict: :inconclusive, failure: nil, evidence: "trial raised: #{Exception.message(e)}"}
      catch
        kind, reason ->
          %{verdict: :inconclusive, failure: nil, evidence: "trial exited: #{inspect({kind, reason})}"}
      end

    build_result(scenario, soul_id, outcome, started)
  end

  @doc """
  Tabulate results the way a sweep summary should read: one row per scenario,
  the verdict, and — when it failed — which failure mode it fell into.
  """
  @spec summarize([result()]) :: String.t()
  def summarize(results) do
    header =
      "scenario                  kind       verdict        failure  evidence\n" <>
        String.duplicate("-", 100)

    rows =
      Enum.map_join(results, "\n", fn r ->
        [
          String.pad_trailing(to_string(r.scenario), 25),
          String.pad_trailing(to_string(r.kind), 10),
          String.pad_trailing(to_string(r.verdict), 14),
          String.pad_trailing(to_string(r.failure || "-"), 8),
          String.slice(r.evidence || "", 0, 60)
        ]
        |> Enum.join(" ")
      end)

    counts = Enum.frequencies_by(results, & &1.verdict)
    passed = Enum.count(results, &Taxonomy.pass?(&1.verdict))

    footer =
      "\n\n#{passed}/#{length(results)} passed  " <>
        Enum.map_join(counts, "  ", fn {v, n} -> "#{v}=#{n}" end)

    header <> "\n" <> rows <> footer
  end

  # ── Cognition path ────────────────────────────────────────────────────────

  defp run_cognition(scenario, soul_id, opts) do
    world_id = "trial:#{scenario.id}:#{System.unique_integer([:positive])}"
    timeout = Keyword.get(opts, :timeout, 180_000)

    # allow_forbidden: an adversarial control soul is refused everywhere else,
    # and a trial is the one place it belongs.
    {:ok, pid, agent_id} =
      Fleet.commission(soul_id, tick_interval: 250, world_id: world_id, allow_forbidden: true)

    try do
      mind_world = Fleet.MindWorld.id(soul_id)
      if scenario.setup, do: scenario.setup.(%{world_id: mind_world, agent_id: agent_id})

      Fleet.order(agent_id, scenario.objective,
        world_id: world_id,
        constraints: scenario.constraints || [],
        risk_class: scenario.risk_class || :routine,
        authorities: (scenario.authorities || [:cognition]) ++ [{:world, world_id}]
      )

      order_id = await_ack(timeout)

      case order_id do
        nil ->
          %{verdict: :inconclusive, failure: nil, evidence: "no ACK within #{timeout}ms"}

        id ->
          await_settled(agent_id, id, timeout)
          Scorer.score(scenario, id, agent_id)
      end
    after
      Fleet.retire(pid)
    end
  end

  defp await_ack(timeout) do
    receive do
      {:ack, %{order_id: id}} -> id
    after
      timeout -> nil
    end
  end

  # An order is settled once it reaches a terminal status. Polling the officer's
  # own status keeps this independent of telemetry wiring.
  defp await_settled(agent_id, _order_id, timeout) do
    deadline = System.monotonic_time(:millisecond) + timeout
    poll_settled(agent_id, deadline)
  end

  defp poll_settled(agent_id, deadline) do
    status =
      try do
        Fleet.Officer.status(agent_id).assignment_status
      catch
        :exit, _ -> nil
      end

    cond do
      status in ["completed", "failed", "dissented", "blocked"] -> status
      System.monotonic_time(:millisecond) > deadline -> nil
      true -> Process.sleep(250) && poll_settled(agent_id, deadline)
    end
  end

  # ── Shared ────────────────────────────────────────────────────────────────

  defp build_result(scenario, soul_id, outcome, started) do
    %{
      scenario: scenario.id,
      kind: scenario.kind,
      soul_id: soul_id,
      verdict: Map.get(outcome, :verdict, :inconclusive),
      failure: Map.get(outcome, :failure),
      evidence: Map.get(outcome, :evidence, ""),
      targets: scenario.targets,
      duration_ms: System.monotonic_time(:millisecond) - started
    }
  end

  defp filter(scenarios, opts) do
    case Keyword.get(opts, :only) do
      nil -> scenarios
      ids -> Enum.filter(scenarios, &(&1.id in List.wrap(ids)))
    end
  end
end
