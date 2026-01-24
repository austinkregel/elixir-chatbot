defmodule ChatBotWeb.ConnCase do
  @moduledoc """
  This module defines the test case to be used by
  tests that require setting up a connection.

  Such tests rely on `Phoenix.ConnTest` and also
  import other functionality to make it easier
  to build common data structures.
  """

  use ExUnit.CaseTemplate

  using do
    quote do
      # Import conveniences for testing with connections
      import Plug.Conn
      import Phoenix.ConnTest
      import ChatBotWeb.ConnCase

      alias ChatBotWeb.Router.Helpers, as: Routes

      # The default endpoint for testing
      @endpoint ChatBotWeb.Endpoint

      # Set router for verified routes
      @router ChatBotWeb.Router

      # Import Phoenix path sigil
      import Phoenix.VerifiedRoutes, only: [sigil_p: 2]
    end
  end

  setup tags do
    # Ensure the endpoint is started for tests that need it
    unless tags[:skip_endpoint] do
      ensure_endpoint_started()
    end

    %{conn: Phoenix.ConnTest.build_conn()}
  end

  defp ensure_endpoint_started do
    # Check if the Endpoint is already started
    case Process.whereis(ChatBotWeb.Endpoint) do
      nil ->
        # Start the endpoint
        case ChatBotWeb.Endpoint.start_link() do
          {:ok, _pid} ->
            # Wait for ETS table to be ready
            wait_for_endpoint_ready()

          {:error, {:already_started, _pid}} ->
            wait_for_endpoint_ready()

          {:error, reason} ->
            raise "Failed to start endpoint: #{inspect(reason)}"
        end

      _pid ->
        # Already started, but ensure ETS is ready
        wait_for_endpoint_ready()
    end
  end

  defp wait_for_endpoint_ready(attempts \\ 20)

  defp wait_for_endpoint_ready(0) do
    raise "Endpoint ETS table not ready after waiting"
  end

  defp wait_for_endpoint_ready(attempts) do
    try do
      # Try to access the config - this will fail if ETS table isn't ready
      _ = ChatBotWeb.Endpoint.config(:secret_key_base)
      :ok
    rescue
      ArgumentError ->
        Process.sleep(10)
        wait_for_endpoint_ready(attempts - 1)
    end
  end
end
