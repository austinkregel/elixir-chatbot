defmodule ChatWeb.LayoutsTest do
  use ExUnit.Case, async: false
  import Phoenix.LiveViewTest

  alias ChatWeb.Layouts

  describe "app/1" do
    test "renders with required flash attribute" do
      assigns = %{flash: %{}, inner_block: []}

      # Use render_component to test the component
      html = render_component(&Layouts.app/1, assigns)

      assert html =~ "navbar"
      assert html =~ "phoenixframework.org"
      assert html =~ "Get Started"
    end

    test "renders main element" do
      assigns = %{flash: %{}, inner_block: []}

      html = render_component(&Layouts.app/1, assigns)

      assert html =~ "<main"
    end

    test "includes theme toggle" do
      assigns = %{flash: %{}, inner_block: []}

      html = render_component(&Layouts.app/1, assigns)

      assert html =~ "phx:set-theme"
    end
  end

  describe "flash_group/1" do
    test "renders with empty flash" do
      assigns = %{flash: %{}}

      html = render_component(&Layouts.flash_group/1, assigns)

      assert html =~ "flash-group"
      assert html =~ "aria-live=\"polite\""
    end

    test "renders with custom id" do
      assigns = %{flash: %{}, id: "custom-flash"}

      html = render_component(&Layouts.flash_group/1, assigns)

      assert html =~ "custom-flash"
    end

    test "includes client error flash" do
      assigns = %{flash: %{}}

      html = render_component(&Layouts.flash_group/1, assigns)

      assert html =~ "client-error"
      # Text is HTML-encoded in the output
      assert html =~ "find the internet" or html =~ "can&#39;t find the internet"
    end

    test "includes server error flash" do
      assigns = %{flash: %{}}

      html = render_component(&Layouts.flash_group/1, assigns)

      assert html =~ "server-error"
      assert html =~ "Something went wrong"
    end
  end

  describe "theme_toggle/1" do
    test "renders theme toggle buttons" do
      assigns = %{}

      html = render_component(&Layouts.theme_toggle/1, assigns)

      assert html =~ "phx:set-theme"
      assert html =~ ~s(data-phx-theme="system")
      assert html =~ ~s(data-phx-theme="light")
      assert html =~ ~s(data-phx-theme="dark")
    end

    test "includes system, light, and dark options" do
      assigns = %{}

      html = render_component(&Layouts.theme_toggle/1, assigns)

      # Check for the three theme icons
      assert html =~ "hero-computer-desktop-micro"
      assert html =~ "hero-sun-micro"
      assert html =~ "hero-moon-micro"
    end
  end
end
