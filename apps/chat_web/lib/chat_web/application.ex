defmodule ChatWeb.Application do
  @moduledoc false
  alias ChatWeb.Endpoint
  use Application

  @impl true
  def start(_type, _args) do
    children = [
      ChatWeb.Telemetry,
      {DNSCluster, [query: Application.get_env(:chat_web, :dns_cluster_query) || :ignore]},
      ChatWeb.Endpoint
    ]

    opts = [strategy: :one_for_one, name: ChatWeb.Supervisor]
    Supervisor.start_link(children, opts)
  end

  @impl true
  def config_change(changed, _new, removed) do
    Endpoint.config_change(changed, removed)
    :ok
  end
end