defmodule ChatWeb.Router do
  use ChatWeb, :router

  pipeline :browser do
    plug :accepts, ["html"]
    plug :fetch_session
    plug :fetch_live_flash
    plug :put_root_layout, html: {ChatWeb.Layouts, :root}
    plug :protect_from_forgery
    plug :put_secure_browser_headers
  end

  pipeline :api do
    plug :accepts, ["json"]
  end

  # This is a Fleet-management workbench: the Fleet console is the whole UI.
  scope "/", ChatWeb do
    pipe_through :browser

    live_session :world_context,
      on_mount: [{ChatWeb.WorldContext, :default}] do
      # `/` is the console; `/fleet` kept as a stable alias.
      live "/", Admin.FleetLive
      live "/fleet", Admin.FleetLive
      # The ship's black box — comprehensive systems visibility.
      live "/systems", Admin.SystemsLive
    end
  end

  # Enable LiveDashboard and Swoosh mailbox preview in development
  if Application.compile_env(:chat_web, :dev_routes) do
    import Phoenix.LiveDashboard.Router

    scope "/dev" do
      pipe_through :browser

      live_dashboard "/dashboard",
        metrics: ChatWeb.Telemetry,
        router: ChatWeb.Router

      forward "/mailbox", Plug.Swoosh.MailboxPreview
    end
  end
end
