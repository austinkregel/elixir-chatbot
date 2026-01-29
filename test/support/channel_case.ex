defmodule ChatBotWeb.ChannelCase do
  @moduledoc """
  This module defines the test case to be used by
  channel tests.

  Such tests rely on `Phoenix.ChannelTest` and also
  import other functionality to make it easier
  to build common data structures.
  """

  use ExUnit.CaseTemplate

  using do
    quote do
      # Import conveniences for testing with channels
      import Phoenix.ChannelTest
      import ChatBotWeb.ChannelCase

      # The default endpoint for testing
      @endpoint ChatBotWeb.Endpoint
    end
  end

  setup tags do
    # Ensure the endpoint is started for channel tests
    unless tags[:skip_endpoint] do
      ensure_endpoint_started()
    end

    :ok
  end

  defp ensure_endpoint_started do
    case Process.whereis(ChatBotWeb.Endpoint) do
      nil ->
        case ChatBotWeb.Endpoint.start_link() do
          {:ok, _pid} -> wait_for_endpoint_ready()
          {:error, {:already_started, _pid}} -> wait_for_endpoint_ready()
          {:error, reason} -> raise "Failed to start endpoint: #{inspect(reason)}"
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
      _ = ChatBotWeb.Endpoint.config(:secret_key_base)
      :ok
    rescue
      ArgumentError ->
        Process.sleep(10)
        wait_for_endpoint_ready(attempts - 1)
    end
  end
end
