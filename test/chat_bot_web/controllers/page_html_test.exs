defmodule ChatBotWeb.PageHTMLTest do
  use ExUnit.Case, async: true

  describe "ChatBotWeb.PageHTML" do
    test "module is defined" do
      assert Code.ensure_loaded?(ChatBotWeb.PageHTML)
    end

    test "uses ChatBotWeb :html" do
      # The module should have embedded templates
      # Check that it has the expected function for home template
      assert function_exported?(ChatBotWeb.PageHTML, :home, 1)
    end

    test "home/1 renders template" do
      assigns = %{}

      # Call the template function
      result = ChatBotWeb.PageHTML.home(assigns)

      # Result should be a rendered template (Phoenix.LiveView.Rendered struct)
      assert is_struct(result) or is_binary(result) or is_list(result)
    end
  end
end
