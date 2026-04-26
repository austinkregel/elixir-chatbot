defmodule Brain.Test.AtlasSandbox do
  @moduledoc false

  # GenServers and supervisors that may run `Atlas.Repo` work outside the test
  # process. Extend when Sandbox leak logs show a new long-lived client.
  # Lives under lib/ (not test/support) so umbrella apps like chat_web can
  # reference it from their ConnCase.
  @atlas_genservers [
    Brain.Services.CredentialVault,
    Brain.Epistemic.SourceAuthority,
    Brain.Epistemic.BeliefStore,
    Brain.Memory.Store,
    Brain.Knowledge.ReviewQueue,
    Brain.Epistemic.UserModelStore,
    Brain.Knowledge.SourceReliability,
    Brain.FactDatabase,
    Brain.ML.Gazetteer,
    Brain,
    Brain.Epistemic.JTMS,
    Brain.Analysis.ComprehensionAssessor
  ]

  @task_supervisors [
    Brain.AtlasTaskSupervisor,
    Brain.Knowledge.AgentSupervisor
  ]

  @doc "Registered names that receive `Sandbox.allow(Atlas.Repo, owner, pid)`."
  def atlas_genservers, do: @atlas_genservers

  @doc """
  Allow persistent Brain/Atlas workers and task supervisors to use the test
  sandbox connection checked out by `owner_pid`.
  """
  def allow_for_test_owner!(owner_pid) when is_pid(owner_pid) do
    for name <- @atlas_genservers do
      if pid = Process.whereis(name) do
        Ecto.Adapters.SQL.Sandbox.allow(Atlas.Repo, owner_pid, pid)
      end
    end

    for name <- @task_supervisors do
      if pid = Process.whereis(name) do
        Ecto.Adapters.SQL.Sandbox.allow(Atlas.Repo, owner_pid, pid)
      end
    end

    :ok
  end

  @doc """
  Standard teardown: drain async Atlas work, then stop the sandbox owner.
  """
  def drain_and_stop_owner(owner_pid) when is_pid(owner_pid) do
    try do
      Brain.AtlasIntegration.drain()
    rescue
      _ -> :ok
    catch
      _, _ -> :ok
    end

    Ecto.Adapters.SQL.Sandbox.stop_owner(owner_pid)
  end

  @doc """
  Checkout sandbox, enable AGE + `atlas_test` search_path, allow workers.
  Returns the owner PID (use in `on_exit` with `drain_and_stop_owner/1`).
  """
  def checkout_and_configure!(tags) when is_map(tags) do
    try do
      Ecto.Adapters.SQL.Sandbox.mode(Atlas.Repo, :manual)
    rescue
      _ -> :ok
    catch
      :exit, _ -> :ok
    end

    pid =
      try do
        Ecto.Adapters.SQL.Sandbox.start_owner!(Atlas.Repo, shared: not tags[:async])
      rescue
        MatchError ->
          Ecto.Adapters.SQL.Sandbox.mode(Atlas.Repo, :manual)
          Process.sleep(50)
          Ecto.Adapters.SQL.Sandbox.start_owner!(Atlas.Repo, shared: not tags[:async])
      end

    Ecto.Adapters.SQL.query!(Atlas.Repo, "LOAD 'age'", [])

    Ecto.Adapters.SQL.query!(
      Atlas.Repo,
      ~s(SET search_path = atlas_test, ag_catalog, "$user", public),
      []
    )

    allow_for_test_owner!(pid)
    pid
  end
end
