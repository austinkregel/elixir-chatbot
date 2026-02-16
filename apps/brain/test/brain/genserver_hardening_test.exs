defmodule Brain.GenServerHardeningTest do
  @moduledoc "Tests for GenServer hardening: ready?/0, timeouts, @impl annotations."
  use ExUnit.Case, async: false

  describe "ready?/0 for application-started GenServers" do
    @tag :gen_server_hardening
    test "Memory.Store ready? returns true when started" do
      assert Brain.Memory.Store.ready?() == true
    end

    @tag :gen_server_hardening
    test "HeuristicStore ready? returns true when started" do
      assert Brain.Analysis.HeuristicStore.ready?() == true
    end

    @tag :gen_server_hardening
    test "Metrics.Aggregator ready? returns true when started" do
      assert Brain.Metrics.Aggregator.ready?() == true
    end

    @tag :gen_server_hardening
    test "TemplateStore ready? returns true when started" do
      assert Brain.Response.TemplateStore.ready?() == true
    end

    @tag :gen_server_hardening
    test "CredentialVault ready? returns true when started" do
      assert Brain.Services.CredentialVault.ready?() == true
    end

    @tag :gen_server_hardening
    test "ContradictionHandler ready? returns true when started" do
      assert Brain.Epistemic.ContradictionHandler.ready?() == true
    end
  end

  describe "TemplateStore with timeout" do
    @tag :gen_server_hardening
    test "get_templates returns list (not crash) when store is ready" do
      result = Brain.Response.TemplateStore.get_templates("smalltalk.greet")
      assert is_list(result)
    end
  end

  describe "CredentialVault with timeout" do
    @tag :gen_server_hardening
    test "store returns :ok or error tuple when vault is ready" do
      result = Brain.Services.CredentialVault.store(:test, :key, "value", [])
      assert result == :ok or match?({:error, _}, result)
    end
  end

  describe "ContradictionHandler catch-all handle_info" do
    @tag :gen_server_hardening
    test "ignores unexpected messages without crashing" do
      send(Brain.Epistemic.ContradictionHandler, {:unexpected, :message})
      send(Brain.Epistemic.ContradictionHandler, :another_unexpected)
      # If we get here without crash, the catch-all works
      assert Brain.Epistemic.ContradictionHandler.ready?() == true
    end
  end

  describe "subprocess ready? with dynamic process" do
    @tag :gen_server_hardening
    test "CliSubprocess ready? returns true for started subprocess, false for unknown" do
      {:ok, _pid, subprocess_id} =
        Brain.Subprocesses.Supervisor.start_cli_subprocess(
          subprocess_id: "test-genserver-#{System.unique_integer([:positive])}",
          memory_snapshot: %{}
        )

      assert Brain.Subprocesses.CliSubprocess.ready?(subprocess_id) == true
      # Unknown subprocess returns false (catches exit)
      assert Brain.Subprocesses.CliSubprocess.ready?("nonexistent-#{System.unique_integer([:positive])}") == false
    end
  end
end
