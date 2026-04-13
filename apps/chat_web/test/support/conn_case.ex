defmodule ChatWeb.ConnCase do
  @moduledoc "This module defines the test case to be used by\ntests that require setting up a connection.\n\nSuch tests rely on `Phoenix.ConnTest` and also\nimport other functionality to make it easier\nto build common data structures.\n"

  alias ChatWeb.Endpoint
  alias Phoenix.ConnTest
  use ExUnit.CaseTemplate

  using do
    quote do
      import Plug.Conn
      import Phoenix.ConnTest
      import ChatWeb.ConnCase

      alias ChatWeb.Router.Helpers, as: Routes
      @endpoint ChatWeb.Endpoint
      @router ChatWeb.Router
      import Phoenix.VerifiedRoutes, only: [sigil_p: 2]
    end
  end

  setup tags do
    # Checkout sandbox so Brain GenServers can access Atlas during tests
    if Process.whereis(Atlas.Repo) do
      # Reset to manual mode first to clear stale shared connections
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

      atlas_genservers = [
        Brain.Services.CredentialVault,
        Brain.Epistemic.SourceAuthority,
        Brain.Epistemic.BeliefStore,
        Brain.Memory.Store,
        Brain.Knowledge.ReviewQueue,
        Brain.Epistemic.UserModel,
        Brain.Knowledge.SourceReliability,
        Brain.Analysis.IntentReviewQueue,
        Brain.FactDatabase
      ]

      for name <- atlas_genservers do
        if genserver_pid = Process.whereis(name) do
          Ecto.Adapters.SQL.Sandbox.allow(Atlas.Repo, pid, genserver_pid)
        end
      end

      if task_sup = Process.whereis(Brain.AtlasTaskSupervisor) do
        Ecto.Adapters.SQL.Sandbox.allow(Atlas.Repo, pid, task_sup)
      end

      on_exit(fn ->
        try do
          Brain.AtlasIntegration.drain()
        rescue
          _ -> :ok
        catch
          _, _ -> :ok
        end

        Ecto.Adapters.SQL.Sandbox.stop_owner(pid)
      end)
    end

    unless tags[:skip_endpoint] do
      ensure_endpoint_started()
    end

    %{conn: ConnTest.build_conn()}
  end

  defp ensure_endpoint_started do
    case Process.whereis(ChatWeb.Endpoint) do
      nil ->
        case Endpoint.start_link() do
          {:ok, _pid} ->
            wait_for_endpoint_ready()

          {:error, {:already_started, _pid}} ->
            wait_for_endpoint_ready()

          {:error, reason} ->
            raise "Failed to start endpoint: #{inspect(reason)}"
        end

      _pid ->
        wait_for_endpoint_ready()
    end
  end

  defp wait_for_endpoint_ready(attempts \\ 20)

  defp wait_for_endpoint_ready(0) do
    raise "Endpoint ETS table not ready after waiting"
  end

  defp wait_for_endpoint_ready(attempts) do
    try do
      _ = Endpoint.config(:secret_key_base)
      :ok
    rescue
      ArgumentError ->
        Process.sleep(10)
        wait_for_endpoint_ready(attempts - 1)
    end
  end
end
