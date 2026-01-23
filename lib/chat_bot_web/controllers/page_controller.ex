defmodule ChatBotWeb.PageController do
  use ChatBotWeb, :controller

  def home(conn, _params) do
    redirect(conn, to: ~p"/chat")
  end
end
