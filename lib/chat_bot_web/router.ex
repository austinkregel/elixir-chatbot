defmodule ChatBotWeb.Router do
  use ChatBotWeb, :router

  pipeline :browser do
    plug :accepts, ["html"]
    plug :fetch_session
    plug :fetch_live_flash
    plug :put_root_layout, html: {ChatBotWeb.Layouts, :root}
    plug :protect_from_forgery
    plug :put_secure_browser_headers
  end

  pipeline :api do
    plug :accepts, ["json"]
  end

  scope "/", ChatBotWeb do
    pipe_through :browser

    get "/", PageController, :home
    live "/chat", ChatLive
    live "/admin", AdminLive
  end

  # Test endpoints
  scope "/api", ChatBotWeb do
    pipe_through :api

    post "/test-learning", TestController, :test_learning
    get "/test-knowledge", TestController, :test_knowledge
    post "/add-test-knowledge", TestController, :add_test_knowledge
  end

  # Other scopes may use custom stacks.
  # scope "/api", ChatBotWeb do
  #   pipe_through :api
  # end

  # Enable LiveDashboard and Swoosh mailbox preview in development
  if Application.compile_env(:chat_bot, :dev_routes) do
    # If you want to use the LiveDashboard in production, you should put
    # it behind authentication and allow only admins to access it.
    # If your application does not have an admins-only section yet,
    # you can use Plug.BasicAuth to set up some basic authentication
    # as long as you are also using SSL (which you should anyway).
    import Phoenix.LiveDashboard.Router

    scope "/dev" do
      pipe_through :browser

      live_dashboard "/dashboard", metrics: ChatBotWeb.Telemetry
      forward "/mailbox", Plug.Swoosh.MailboxPreview
    end
  end
end
