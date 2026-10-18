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

  # The time zone the message form's BrowserTimeZone hook sends with a
  # message (assets/js/app.js), as a browser in UTC sends it: no changes of
  # offset from 2000 to 2100. The hook writes it into the form's hidden
  # time_zone input as the form is submitted, so a test gives it to
  # render_submit/2 as the hook's value; form/3 takes only what a person types.
  @browser_zone Jason.encode!(%{
                  "name" => "Etc/UTC",
                  "at" => DateTime.to_unix(~U[2026-10-14 12:00:00Z], :millisecond),
                  "offset" => 0,
                  "from" => DateTime.to_unix(~U[2000-01-01 00:00:00Z], :millisecond),
                  "until" => DateTime.to_unix(~U[2100-01-01 00:00:00Z], :millisecond),
                  "offset_at_from" => 0,
                  "transitions" => []
                })

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

  describe "language" do
    test "an English browser, regional variant included, mounts in English", %{conn: conn} do
      conn = Plug.Conn.put_req_header(conn, "accept-language", "en-GB,en;q=0.9,fr;q=0.8")
      {:ok, view, _html} = live(conn, "/chat")

      assert :sys.get_state(view.pid).socket.assigns.language == "en"
    end

    test "a browser whose top choice is unsupported fails, naming the language", %{conn: conn} do
      conn = Plug.Conn.put_req_header(conn, "accept-language", "fr-FR,en;q=0.9")

      assert_raise ArgumentError, ~r/"fr" is not supported/, fn -> live(conn, "/chat") end
    end
  end

  describe "conversation management" do
    test "creates new conversation when sending first message", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/chat")

      # Send a message
      view
      |> form("#message-form", %{input: "Hello, Echo!"})
      |> render_submit(%{time_zone: @browser_zone})

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
      |> render_submit(%{time_zone: @browser_zone})

      html = render(view)
      assert html =~ "Conversation"
      assert html =~ "0 messages"
    end

    test "can end conversation", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/chat")

      # Create conversation
      view
      |> form("#message-form", %{input: "Test message"})
      |> render_submit(%{time_zone: @browser_zone})

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
      |> render_submit(%{time_zone: @browser_zone})

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
      |> render_submit(%{time_zone: @browser_zone})

      # Input should be cleared
      assert has_element?(view, "#message-input[value='']")
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
      |> render_submit(%{time_zone: @browser_zone})

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
      |> render_submit(%{time_zone: @browser_zone})

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

  describe "the browser's time zone" do
    test "the message form fills it with the BrowserTimeZone hook, for the span of years configured", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/chat")

      years = Application.fetch_env!(:chat_web, :browser_zone_years)
      assert has_element?(view, ~s(#message-form[phx-hook="BrowserTimeZone"][data-zone-years="#{years}"]))
      assert has_element?(view, ~s(#message-form input[type="hidden"][name="time_zone"]))
    end

    test "a message sent without it fails, naming what is missing, and is not evaluated", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/chat")
      Process.flag(:trap_exit, true)

      assert {{%ArgumentError{message: message}, _stacktrace}, _call} =
               catch_exit(render_submit(view, "send_message", %{"input" => "set an alarm for 7 am", "time_zone" => ""}))

      assert message =~ "carries the browser's time zone as time_zone"
    end

    test "a message's time is shown in the browser's zone, by the LocalTime hook", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/chat")

      view
      |> form("#message-form", %{input: "Hello there!"})
      |> render_submit(%{time_zone: @browser_zone})

      assert has_element?(view, ~s(#{@user_bubble} time[phx-hook="LocalTime"][data-format="time"][datetime]))
    end
  end

  describe "alarms, reminders, timers and calendar items that fire" do
    defp user_id(view), do: :sys.get_state(view.pid).socket.assigns.current_user_id

    defp fired(changes) do
      Map.merge(
        %{
          id: Ecto.UUID.generate(),
          world_id: "test_chat_live_notifications",
          kind: :alarm,
          said: %{time: "7 am", date: "tomorrow"},
          due_at: ~U[2026-10-13 12:00:00.000000Z],
          fired_at: ~U[2026-10-13 12:00:00.000000Z],
          late: false,
          status: "ringing"
        },
        changes
      )
    end

    test "one told to the user's topic shows on the chat page, with its due time in the browser's zone, and can be dismissed",
         %{conn: conn} do
      {:ok, view, _html} = live(conn, "/chat")
      refute has_element?(view, "#scheduler-notifications")

      alarm = fired(%{})
      Phoenix.PubSub.broadcast(Brain.PubSub, Atlas.WebUsers.topic(user_id(view)), {:scheduler_fired, alarm})

      notification = "#notification-#{alarm.id}"
      assert has_element?(view, notification, "Alarm")
      assert has_element?(view, notification, "Ringing")
      assert has_element?(view, notification, "tomorrow · 7 am")
      assert has_element?(view, ~s(#{notification} time[phx-hook="LocalTime"][datetime="2026-10-13T12:00:00.000Z"]))
      refute has_element?(view, notification, "Fired late")

      view |> element("#{notification} button[phx-click='dismiss_notification']") |> render_click()
      refute has_element?(view, notification)
      assert Process.alive?(view.pid)
    end

    test "one that fired late says so, with when it fired", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/chat")

      reminder =
        fired(%{kind: :reminder, said: %{text: "call home", time: "6 pm", date: ""}, status: nil, late: true, fired_at: ~U[2026-10-13 19:00:00.000000Z]})

      Phoenix.PubSub.broadcast(Brain.PubSub, Atlas.WebUsers.topic(user_id(view)), {:scheduler_fired, reminder})

      notification = "#notification-#{reminder.id}"
      assert has_element?(view, notification, "Reminder")
      assert has_element?(view, notification, "call home · 6 pm")
      assert has_element?(view, notification, "Fired late")
      assert has_element?(view, ~s(#{notification} time[datetime="2026-10-13T19:00:00.000Z"]))
    end

    test "one told to another user's topic does not show", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/chat")
      other = Atlas.WebUsers.create!("test_chat_live_notifications")

      Phoenix.PubSub.broadcast(Brain.PubSub, Atlas.WebUsers.topic(other.id), {:scheduler_fired, fired(%{})})

      refute has_element?(view, "#scheduler-notifications")
    end
  end
end
