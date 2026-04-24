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

  # Main application routes with world context
  scope "/", ChatWeb do
    pipe_through :browser

    # Landing page (no world context needed)
    get "/", PageController, :home

    # World-aware LiveViews with shared context
    live_session :world_context,
      on_mount: [{ChatWeb.WorldContext, :default}] do
      live "/chat", ChatLive
      live "/chat/:conversation_id", ChatLive
      live "/explorer", ExplorerLive
      live "/dashboard", DashboardLive
      live "/settings", SettingsLive
      live "/code", CodeAnalysisLive
      live "/sessions", SessionsLive
      live "/sessions/:session_id", SessionsLive
      live "/knowledge-review", Admin.KnowledgeReviewLive
      live "/accuracy", AccuracyLive
    end

    # Legacy route redirects
    get "/admin", PageController, :redirect_to_settings
    get "/worlds", PageController, :redirect_to_explorer
    get "/worlds/:world_id/entities", PageController, :redirect_to_explorer
    get "/memories", PageController, :redirect_to_explorer
  end

  # Legacy operational dashboard route (redirect to new location)
  scope "/ops", ChatWeb do
    pipe_through :browser

    # Redirect old dashboard URL to new one
    get "/dashboard", PageController, :redirect_dashboard
  end

  # Test endpoints
  scope "/api", ChatWeb do
    pipe_through :api

    post "/test-learning", TestController, :test_learning
    get "/test-knowledge", TestController, :test_knowledge
    post "/add-test-knowledge", TestController, :add_test_knowledge
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
