defmodule ChatWeb.PageHTMLTest do
  alias ChatWeb.PageHTML
  use ExUnit.Case, async: false

  describe "ChatWeb.PageHTML" do
    test "module is defined" do
      assert Code.ensure_loaded?(ChatWeb.PageHTML)
    end

    test "uses ChatWeb :html" do
      assert function_exported?(ChatWeb.PageHTML, :home, 1)
    end

    test "home/1 renders template" do
      assigns = %{}
      result = PageHTML.home(assigns)
      assert is_struct(result) or is_binary(result) or is_list(result)
    end
  end
end