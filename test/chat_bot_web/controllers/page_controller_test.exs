defmodule ChatBotWeb.PageControllerTest do
  use ChatBotWeb.ConnCase

  # These tests require the full application endpoint
  @moduletag :integration
  @moduletag skip: "Requires full application (run with mix test --include integration)"

  test "GET / redirects to /chat", %{conn: conn} do
    conn = get(conn, "/")
    assert redirected_to(conn) == "/chat"
  end
end
