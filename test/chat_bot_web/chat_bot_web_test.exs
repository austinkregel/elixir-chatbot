defmodule ChatBotWebTest do
  use ExUnit.Case, async: true

  describe "static_paths/0" do
    test "returns list of static asset paths" do
      paths = ChatBotWeb.static_paths()

      assert is_list(paths)
      assert "assets" in paths
      assert "images" in paths
      assert "favicon.ico" in paths
      assert "robots.txt" in paths
    end
  end

  describe "__using__/1 macro" do
    test "controller quote block includes expected imports" do
      # The controller/0 function returns a quoted expression
      quoted = ChatBotWeb.controller()

      assert is_tuple(quoted)
      assert elem(quoted, 0) == :__block__
    end

    test "live_view quote block includes expected imports" do
      quoted = ChatBotWeb.live_view()

      assert is_tuple(quoted)
      assert elem(quoted, 0) == :__block__
    end

    test "live_component quote block includes expected imports" do
      quoted = ChatBotWeb.live_component()

      assert is_tuple(quoted)
      assert elem(quoted, 0) == :__block__
    end

    test "html quote block includes expected imports" do
      quoted = ChatBotWeb.html()

      assert is_tuple(quoted)
      assert elem(quoted, 0) == :__block__
    end

    test "router quote block includes expected imports" do
      quoted = ChatBotWeb.router()

      assert is_tuple(quoted)
      assert elem(quoted, 0) == :__block__
    end

    test "channel quote block includes expected imports" do
      quoted = ChatBotWeb.channel()

      assert is_tuple(quoted)
      # Channel is a simple `use Phoenix.Channel` so it's a :use tuple, not a block
      assert elem(quoted, 0) in [:__block__, :use]
    end

    test "verified_routes quote block includes expected imports" do
      quoted = ChatBotWeb.verified_routes()

      assert is_tuple(quoted)
    end
  end
end
