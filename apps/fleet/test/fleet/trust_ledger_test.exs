defmodule Fleet.TrustLedgerTest do
  @moduledoc """
  The trust ledger: derived from the append-only record, honest about what it
  cannot yet measure.
  """
  use Fleet.FleetCase

  alias Fleet.{Order, Service, TrustLedger}

  @moduletag :integration

  defp soul_id, do: "ensign-ledger-#{System.unique_integer([:positive])}"

  defp order(id, soul), do: Order.new(id: id, from: :admiral, objective: "task #{id}", world_id: soul)

  test "a soul with no service history measures nothing rather than guessing" do
    ledger = TrustLedger.compute(soul_id())

    assert ledger.verified_accuracy.state == :unmeasured
    assert ledger.verified_accuracy.value == nil
    assert ledger.calibration.state == :unmeasured
    assert ledger.evidence_quality.state == :unmeasured
    assert ledger.anomaly_record.state == :unmeasured

    # No measurable factor means no score — not a neutral 0.5.
    assert TrustLedger.score(ledger) == nil
  end

  test "completed orders raise verified accuracy; dissents and failures lower it" do
    soul = soul_id()
    Service.commission(soul, %{agent_id: soul, rank: "ensign"})

    Service.record_order_outcome(soul, order("l1", soul), :completed, %{outcome: "done"})
    Service.record_order_outcome(soul, order("l2", soul), :completed, %{outcome: "done"})
    Service.record_order_outcome(soul, order("l3", soul), :failed, %{reason: "backend down"})

    ledger = TrustLedger.compute(soul)

    assert ledger.verified_accuracy.state == :measured
    assert_in_delta ledger.verified_accuracy.value, 2 / 3, 0.001
    assert ledger.orders.completed == 2
    assert ledger.orders.failed == 1
    assert ledger.verified_accuracy.basis =~ "2 completed of 3"
  end

  # A blocked order is in flight, not a failure. Counting it against the officer
  # would punish asking for clarification, which is the behaviour we want.
  test "a blocked order does not count against accuracy" do
    soul = soul_id()
    Service.commission(soul, %{agent_id: soul, rank: "ensign"})

    Service.record_order_outcome(soul, order("b1", soul), :completed, %{outcome: "done"})
    Service.record_order_outcome(soul, order("b2", soul), :blocked, %{reason: "needs a date"})

    ledger = TrustLedger.compute(soul)

    assert ledger.verified_accuracy.value == 1.0
    assert ledger.orders.blocked == 1
  end

  test "the ledger is computed from the record, so it cannot be self-reported" do
    soul = soul_id()
    Service.commission(soul, %{agent_id: soul, rank: "ensign"})
    Service.record_order_outcome(soul, order("s1", soul), :completed, %{outcome: "done"})

    a = TrustLedger.compute(soul)
    b = TrustLedger.compute(soul)

    assert a.verified_accuracy.value == b.verified_accuracy.value
    assert a.orders == b.orders
  end

  test "score averages only measured factors" do
    soul = soul_id()
    Service.commission(soul, %{agent_id: soul, rank: "ensign"})
    Service.record_order_outcome(soul, order("sc1", soul), :completed, %{outcome: "done"})

    ledger = TrustLedger.compute(soul)
    score = TrustLedger.score(ledger)

    assert is_float(score)
    assert score >= 0.0 and score <= 1.0
  end

  test "the ledger tool is gated to XO and above, not readable by its subject" do
    {:ok, tool} = Fleet.Tool.lookup("trust.read")

    assert tool.info_class == :trust_ledger
    assert tool.effect == :read
    assert Fleet.InfoClass.min_billet(:trust_ledger) == :executive_officer
  end
end
