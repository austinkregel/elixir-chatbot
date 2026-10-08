defmodule ChatWeb.ChatLiveTest do
  use ChatWeb.ConnCase, async: false
  import Phoenix.LiveViewTest
  import Brain.TestHelpers, only: [start_brain_services: 0, eventually: 4]

  @moduletag :integration

  # The person's message renders in a right-aligned column; the assistant's
  # reply, which arrives when the Brain finishes evaluating, in a left-aligned
  # one on the surface ground.
  @user_bubble "div.flex-col.items-end"
  @assistant_bubble "div.flex-col.items-start > div.bg-surface"

  setup do
    start_brain_services()
    :ok
  end

  # The reply is sent to the LiveView from a task once `Brain.evaluate/3`
  # returns, so it is waited for, up to 15 seconds.
  defp await_assistant_reply(view) do
    eventually(fn -> has_element?(view, @assistant_bubble) end, & &1, 150, 100)
    view |> element(@assistant_bubble) |> render()
  end

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

      # The conversation is created and named in the header, the message is
      # shown, and the assistant's reply follows.
      html = render(view)
      refute html =~ "Start a new conversation"
      assert html =~ ~r/<h2[^>]*>\s*Conversation [0-9a-f]{8}\s*<\/h2>/
      assert has_element?(view, @user_bubble, "Hello, Echo!")

      reply = await_assistant_reply(view)
      assert reply =~ ~r/whitespace-pre-wrap">\s*\S/
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

      # Ending a conversation writes shared state, so the trigger opens the
      # confirm panel and only its confirm button ends the conversation.
      view
      |> element("#end-conversation")
      |> render_click()

      assert has_element?(view, "#end-conversation-confirm")

      view
      |> element("#end-conversation-confirm-confirm")
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

      # Should show user message
      assert has_element?(view, @user_bubble, "Hello there!")

      # Should show assistant response
      reply = await_assistant_reply(view)
      assert reply =~ ~r/whitespace-pre-wrap">\s*\S/
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

  describe "message validation" do
    @blank_message_error "A message needs at least one character that is not a space."

    # The input's `title` states the same rule for the browser's own hint, so
    # the validation message is looked for in its caption below the input.
    @error_caption "#message-form p"

    test "the message input requires a character that is not whitespace", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/chat")

      assert has_element?(view, "#message-input[required][minlength='1']")
      assert view |> element("#message-input") |> render() =~ ~S(pattern=".*\S.*")
    end

    test "Send stays disabled while the input is blank or whitespace", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/chat")

      assert has_element?(view, "#message-form button[type='submit'][disabled]")

      view |> form("#message-form", %{input: "   "}) |> render_change()
      assert has_element?(view, "#message-form button[type='submit'][disabled]")

      view |> form("#message-form", %{input: "hi"}) |> render_change()
      refute has_element?(view, "#message-form button[type='submit'][disabled]")
    end

    for {label, blank} <- [{"empty", ""}, {"whitespace-only", " \t\n "}] do
      test "rejects an #{label} message without sending it", %{conn: conn} do
        {:ok, view, _html} = live(conn, "/chat")

        # A submit that bypasses the browser's validation reaches the server,
        # which is the guard that holds.
        html = render_submit(view, "send_message", %{"input" => unquote(blank)})

        assert has_element?(view, @error_caption, @blank_message_error)
        assert has_element?(view, "#message-input[aria-invalid='true']")
        refute has_element?(view, @user_bubble)
        assert html =~ "Start a new conversation"
        assert Process.alive?(view.pid)
      end
    end

    test "typing a message clears the validation message", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/chat")

      refute has_element?(view, @error_caption)

      render_submit(view, "send_message", %{"input" => "  "})
      assert has_element?(view, @error_caption, @blank_message_error)

      view |> form("#message-form", %{input: "hello"}) |> render_change()
      refute has_element?(view, @error_caption)
      refute has_element?(view, "#message-input[aria-invalid]")
    end
  end

  describe "evaluation errors" do
    # `Brain.evaluate/3` returns `{:error, {:generation_failed, message}}` only
    # when evaluation raises, and no input reliably makes it raise, so this
    # sends the LiveView the result message its evaluation task sends.
    test "a generation failure shows as an error and the page survives", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/chat")

      view
      |> form("#message-form", %{input: "Hello there!"})
      |> render_submit()

      "/chat/" <> conversation_id = assert_patch(view)
      await_assistant_reply(view)

      send(
        view.pid,
        {:evaluation_complete,
         %{
           conversation_id: conversation_id,
           message_id: "failed-message",
           result: {:error, {:generation_failed, "expected a map, got: nil"}}
         }}
      )

      html = render(view)
      assert html =~ "Generating a response failed: expected a map, got: nil"
      assert Process.alive?(view.pid)
    end

    test "an error of another shape shows as its term", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/chat")

      view
      |> form("#message-form", %{input: "Hello there!"})
      |> render_submit()

      "/chat/" <> conversation_id = assert_patch(view)
      await_assistant_reply(view)

      send(
        view.pid,
        {:evaluation_complete,
         %{conversation_id: conversation_id, message_id: "failed-message", result: {:error, {:timeout, 5000}}}}
      )

      html = render(view)
      assert html =~ "Evaluating the message failed: {:timeout, 5000}"
      assert Process.alive?(view.pid)
    end
  end
end
