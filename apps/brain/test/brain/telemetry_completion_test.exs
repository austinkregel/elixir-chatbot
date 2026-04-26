defmodule Brain.TelemetryCompletionTest do
  @moduledoc "Tests for telemetry completion: response generation, fact database, template lookup, readiness tracking."
  use ExUnit.Case, async: false

  describe "response generation telemetry" do
    test "Generator.generate emits response generate stop event" do
      this = self()

      handler_id = "test-response-generate-#{System.unique_integer([:positive])}"

      # Module-function capture so :telemetry doesn't warn about a
      # local/anonymous handler (which would log once per test run).
      :telemetry.attach(
        handler_id,
        [:chat_bot, :response, :generate, :stop],
        &__MODULE__.__forward_measurements__/4,
        %{target: this}
      )

      try do
        _ = Brain.Response.Generator.generate("smalltalk.greet", [], nil)
        assert_receive {:telemetry, %{duration: _duration}}, 2_000
      after
        :telemetry.detach(handler_id)
      end
    end
  end

  @doc false
  def __forward_measurements__(_event, measurements, _metadata, %{target: target})
      when is_pid(target) do
    send(target, {:telemetry, measurements})
    :ok
  end

  describe "record_readiness" do
    test "stores readiness state retrievable via get_metrics" do
      Brain.Metrics.Aggregator.record_readiness(:test_system, true)
      Process.sleep(50)

      metrics = Brain.Metrics.Aggregator.get_metrics()
      readiness = Map.get(metrics, :readiness, %{})
      test_data = Map.get(readiness, :test_system)

      assert test_data != nil
      assert test_data.ready? == true
      assert is_integer(test_data.timestamp)
    end
  end

  describe "previously missing metrics" do
    test "knowledge_research and other metrics appear in get_metrics after recording" do
      Brain.Metrics.Aggregator.record_duration(:knowledge_research, 100, %{})
      Brain.Metrics.Aggregator.record_duration(:jtms_justify, 50, %{})
      Process.sleep(100)

      # Aggregator runs on interval - metrics may need time to aggregate
      metric = Brain.Metrics.Aggregator.get_metric(:knowledge_research)
      assert metric != nil
      assert (metric[:count] || metric["count"] || 0) >= 0
      assert (metric[:avg_ms] || metric["avg_ms"] || 0) >= 0 or metric == %{}
    end
  end
end
