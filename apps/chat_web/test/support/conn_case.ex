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
      pid = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)

      on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(pid) end)
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
