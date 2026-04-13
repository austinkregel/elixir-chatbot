defmodule ChatWeb.ChannelCase do
  @moduledoc "This module defines the test case to be used by\nchannel tests.\n\nSuch tests rely on `Phoenix.ChannelTest` and also\nimport other functionality to make it easier\nto build common data structures.\n"

  alias ChatWeb.Endpoint
  use ExUnit.CaseTemplate

  using do
    quote do
      import Phoenix.ChannelTest
      import ChatWeb.ChannelCase
      @endpoint ChatWeb.Endpoint
    end
  end

  setup tags do
    unless tags[:skip_endpoint] do
      ensure_endpoint_started()
    end

    :ok
  end

  defp ensure_endpoint_started do
    case Process.whereis(ChatWeb.Endpoint) do
      nil ->
        case Endpoint.start_link() do
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
      _ = Endpoint.config(:secret_key_base)
      :ok
    rescue
      ArgumentError ->
        Process.sleep(10)
        wait_for_endpoint_ready(attempts - 1)
    end
  end
end