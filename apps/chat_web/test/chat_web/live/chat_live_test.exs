defmodule ChatWeb.ChatLiveTest do
  use ChatWeb.ConnCase, async: false
  import Phoenix.LiveViewTest

  # These tests require the full application to be running
  # Run with: mix test --only integration
  @moduletag :integration
  @moduletag skip: "Requires full application (run with mix test --include integration)"

  describe "mounting" do
    test "mounts successfully", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/chat")

      assert html =~ "Chat Bot"
      assert html =~ "Echo"
      assert html =~ "cheerful"
    end

    test "shows initial status", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/chat")

      assert html =~ "Status:"
      assert html =~ "Conversations:"
      assert html =~ "Memory:"
      assert html =~ "entries"
    end
  end

  describe "conversation management" do
    test "creates new conversation when sending first message", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/chat")

      # Send a message
      view
      |> form("#message-form", %{input: "Hello, Echo!"})
      |> render_submit()

      # Should create conversation and show response
      html = render(view)
      assert html =~ "Hello, Echo!"
      assert html =~ "Hello! I&#39;m Echo, and I&#39;m happy to help!"
    end

    test "shows conversation in sidebar after creation", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/chat")

      # Send a message to create conversation
      view
      |> form("#message-form", %{input: "Test message"})
      |> render_submit()

      html = render(view)
      assert html =~ "Conversation"
      assert html =~ "0 messages"
    end

    test "can end conversation", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/chat")

      # Create conversation
      view
      |> form("#message-form", %{input: "Test message"})
      |> render_submit()

      # End conversation
      view
      |> element("button", "End Conversation")
      |> render_click()

      html = render(view)
      assert html =~ "Start a new conversation"
    end
  end

  describe "message handling" do
    test "displays user and assistant messages", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/chat")

      # Send message
      view
      |> form("#message-form", %{input: "Hello there!"})
      |> render_submit()

      html = render(view)

      # Should show user message
      assert html =~ "Hello there!"

      # Should show assistant response
      assert html =~ "Hello! I&#39;m Echo, and I&#39;m happy to help!"
    end

    test "clears input after sending message", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/chat")

      # Send message
      view
      |> form("#message-form", %{input: "Test input"})
      |> render_submit()

      # Input should be cleared
      assert has_element?(view, "input[value='']")
    end
  end

  describe "error handling" do
    test "shows error for invalid operations", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/chat")

      # Try to send empty message
      view
      |> form("#message-form", %{input: ""})
      |> render_submit()

      # Should still process the empty message
      html = render(view)
      assert html =~ "You said:"
    end
  end
end
