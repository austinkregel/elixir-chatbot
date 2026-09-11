defmodule ChatWeb.PageController do
  use ChatWeb, :controller

  def home(conn, _params) do
    redirect(conn, to: ~p"/chat")
  end

  def redirect_dashboard(conn, _params) do
    redirect(conn, to: ~p"/dashboard")
  end

  def redirect_to_settings(conn, _params) do
    redirect(conn, to: ~p"/settings")
  end

  def redirect_to_explorer(conn, _params) do
    redirect(conn, to: ~p"/explorer")
  end
end
