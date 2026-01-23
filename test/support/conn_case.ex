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
      case ChatBotWeb.Endpoint.start_link() do
        {:ok, _pid} -> :ok
        {:error, {:already_started, _pid}} -> :ok
        {:error, _reason} -> :ok
      end
    end

    %{conn: Phoenix.ConnTest.build_conn()}
  end
end
