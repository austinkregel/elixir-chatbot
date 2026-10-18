defmodule Brain.Services.SchedulingRepliesTest do
  use ExUnit.Case, async: false

  alias Brain.Response.Synthesizer
  alias Brain.Services.{Dispatcher, LocalScheduler, SchedulingDriver}

  # The replies to alarm, reminder, timer and calendar intents, worded by
  # `Synthesizer.synthesize_from_domain/5` from what the world's scheduling
  # driver returned when the intent was dispatched. Each test works in a fresh
  # `test_` world that has chosen `LocalScheduler`.

  # An acknowledgment that continues a reply's sentence asks the POS tagger
  # whether the reply's first word is a proper noun
  # (`Synthesizer.join_after/2`); the tagger reads WordNet through its
  # lexical features.
  setup_all do
    Brain.TestHelpers.start_pos_tagger!()
  end

  setup tags do
    owner = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)
    on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)

    # The scheduler's clock stands at its configured start until a test moves
    # it (`Brain.Test.ManualClock`).
    Brain.Test.ManualClock.reset!()
    on_exit(&Brain.Test.ManualClock.reset!/0)

    world = "test_scheduling_replies_#{System.unique_integer([:positive])}"
    on_exit(fn -> LocalScheduler.clear_world(world) end)

    assert :ok = Dispatcher.assign_scheduling_driver(world, LocalScheduler)
    Process.put(:scheduling_replies_web_user, Atlas.WebUsers.create!(world).id)
    %{world: world}
  end

  defp intent_domain(intent), do: intent |> String.split(".", parts: 2) |> hd()

  # The turn intents are dispatched in, a Monday morning read in UTC, unless a
  # test says otherwise, said by the test's web user.
  @turn_time ~U[2026-10-12 09:00:00Z]

  defp turn(world, turn_time) do
    %{
      world_id: world,
      turn_time: turn_time,
      time_zone: Brain.Temporal.Zone.utc(),
      web_user_id: Process.get(:scheduling_replies_web_user)
    }
  end

  defp dispatch!(intent, slots, world, turn_time \\ @turn_time) do
    assert {:ok, fields} = Dispatcher.dispatch(intent, slots, turn(world, turn_time))
    fields
  end

  # The reply to a dispatched intent, from the fields the driver returned and
  # the operation it performed, kept in the unified context's enrichment as
  # `Brain.Response.ContextBuilder` keeps them, with the slots it was given as
  # the turn's entities. Below a confidence of 0.7 no acknowledgment is put
  # before it, so the reply is a template as filled.
  defp reply(intent, slots, fields, confidence) do
    entities = for {slot, value} <- slots, do: %{entity_type: Atom.to_string(slot), value: value}
    {:ok, operation} = SchedulingDriver.operation(intent)

    assert {:ok, text} =
             Synthesizer.synthesize_from_domain(intent_domain(intent), intent, entities, confidence,
               context: %{enrichment: %{enrichment_status: :success, enriched_data: fields, enrichment_operation: operation}}
             )

    text
  end

  # Every reply over enough draws to pick each template of a frame.
  defp replies(intent, slots, fields, confidence) do
    for _draw <- 1..20, uniq: true, do: reply(intent, slots, fields, confidence)
  end

  describe "the replies that printed broken text" do
    test "a reminder created words that it is set, and never prints the boolean it was returned with",
         %{world: world} do
      slots = %{reminder_text: "call mom", time: "5pm"}
      fields = dispatch!("reminder.create", slots, world)
      assert fields.reminder_set == true

      for text <- replies("reminder.create", slots, fields, 0.5) do
        assert text in ["Reminder set for 5pm: call mom.", "Your reminder for 5pm is set: call mom."]
      end
    end

    test "an alarm set with no time says what it was set for, or only that it is set", %{world: world} do
      slots = %{date: "tomorrow"}
      fields = dispatch!("alarm.set", slots, world)
      assert %{alarm_time: "", alarm_date: "tomorrow"} = fields

      for text <- replies("alarm.set", slots, fields, 0.5) do
        assert text in ["Your alarm is set for tomorrow.", "Alarm set for tomorrow."]
      end

      fields = dispatch!("alarm.set", %{}, world)

      for text <- replies("alarm.set", %{}, fields, 0.5) do
        assert text in ["Your alarm is set.", "Alarm set."]
      end
    end

    test "a timer check that finds none says so, and one that finds a timer gives its duration and the time left",
         %{world: world} do
      fields = dispatch!("timer.check", %{}, world)

      for text <- replies("timer.check", %{}, fields, 0.5) do
        assert text in ["There's no timer running.", "No timer is running."]
      end

      dispatch!("timer.set", %{duration: "5 minutes"}, world)
      fields = dispatch!("timer.check", %{}, world, DateTime.add(@turn_time, 90))
      assert fields == %{timer_status: "running", duration: "5 minutes", time_left: ["3 minutes", "30 seconds"]}

      for text <- replies("timer.check", %{}, fields, 0.5) do
        assert text in [
                 "There's a timer running, set for 5 minutes, with 3 minutes, 30 seconds left.",
                 "A timer set for 5 minutes is running: 3 minutes, 30 seconds left."
               ]
      end
    end

    test "a timer check after its duration has passed says it has run out", %{world: world} do
      dispatch!("timer.set", %{duration: "5 minutes"}, world)
      fields = dispatch!("timer.check", %{}, world, DateTime.add(@turn_time, 600))
      assert fields == %{timer_status: "finished", duration: "5 minutes", time_left: []}

      for text <- replies("timer.check", %{}, fields, 0.5) do
        assert text in ["The timer set for 5 minutes has run out.", "Your timer for 5 minutes has run out."]
      end
    end

    test "a calendar query names the soonest event still to come, or says none is", %{world: world} do
      dispatch!("calendar.schedule", %{event: "dentist", date: "Friday", time: "3 pm"}, world)
      dispatch!("calendar.schedule", %{event: "standup", date: "tomorrow", time: "9 am"}, world)

      fields = dispatch!("calendar.query", %{}, world)

      for text <- replies("calendar.query", %{}, fields, 0.5) do
        assert text in [
                 "Next on your calendar: standup, tomorrow at 9 am.",
                 "Coming up on your calendar: standup, tomorrow at 9 am."
               ]
      end

      fields = dispatch!("calendar.query", %{}, world, ~U[2026-10-17 09:00:00Z])

      for text <- replies("calendar.query", %{}, fields, 0.5) do
        assert text in ["Nothing on your calendar is still to come.", "Your calendar has nothing coming up."]
      end
    end

    test "an acknowledgment that ends a sentence leaves the reply's capital, and one that continues it lowercases it",
         %{world: world} do
      slots = %{date: "tomorrow"}
      fields = dispatch!("alarm.set", slots, world)

      for text <- replies("alarm.set", slots, fields, 0.95) do
        assert text in [
                 "Your alarm is set for tomorrow.",
                 "Alarm set for tomorrow.",
                 "Sure, your alarm is set for tomorrow.",
                 "Sure, alarm set for tomorrow.",
                 "Done. Your alarm is set for tomorrow.",
                 "Done. Alarm set for tomorrow."
               ]
      end
    end

    test "a calendar query that finds nothing says the calendar is empty", %{world: world} do
      fields = dispatch!("calendar.query", %{}, world)
      assert fields.event_count == 0

      for text <- replies("calendar.query", %{}, fields, 0.5) do
        assert text in ["You have no events on your calendar.", "There's nothing on your calendar."]
      end
    end
  end

  # Every slot an operation reads, so each operation stores, finds or cancels
  # an item described by every field it has.
  @slots %{
    time: "7 am",
    date: "tomorrow",
    duration: "5 minutes",
    reminder_text: "call the dentist",
    event: "team standup"
  }

  # Every scheduling operation, each with what it finds or does: checks with
  # nothing stored; items set with every slot and with none; checks that find
  # one item and then two (one of them described by nothing); the items with a
  # time of day coming due (`:due`) and checked again, the alarm ringing and
  # the reminder gone off; snoozes of the ringing alarm with no length said,
  # with one, and with none ringing; and cancels that remove one item and then
  # several. A calendar cancel removes exactly one event, so it is named.
  @steps [
    {:none_found, "alarm.check", %{}},
    {:none_found, "reminder.check", %{}},
    {:none_found, "timer.check", %{}},
    {:none_found, "calendar.query", %{}},
    {:set, "alarm.set", @slots},
    {:set, "reminder.create", @slots},
    {:set, "timer.set", @slots},
    {:set, "calendar.schedule", @slots},
    {:found, "alarm.check", %{}},
    {:found, "reminder.check", %{}},
    {:found, "timer.check", %{}},
    {:found, "calendar.query", %{}},
    {:cancelled, "alarm.cancel", @slots},
    {:cancelled, "reminder.cancel", @slots},
    {:cancelled, "timer.cancel", @slots},
    {:cancelled, "calendar.cancel", @slots},
    {:set, "alarm.set", @slots},
    {:set, "reminder.create", @slots},
    {:set, "timer.set", @slots},
    {:set, "calendar.schedule", @slots},
    {:set, "alarm.set", %{}},
    {:set, "reminder.create", %{}},
    {:set, "timer.set", %{}},
    {:set, "calendar.schedule", %{}},
    {:found, "alarm.check", %{}},
    {:found, "reminder.check", %{}},
    {:found, "timer.check", %{}},
    {:found, "calendar.query", %{}},
    :due,
    {:ringing, "alarm.check", %{}},
    {:gone_off, "reminder.check", %{}},
    {:missing_duration, "alarm.snooze", %{}},
    {:snoozed, "alarm.snooze", %{duration: "5 minutes"}},
    {:no_ringing_alarm, "alarm.snooze", %{}},
    {:cancelled, "alarm.cancel", %{}},
    {:cancelled, "reminder.cancel", %{}},
    {:cancelled, "timer.cancel", %{}},
    {:cancelled, "calendar.cancel", %{event: "team standup"}}
  ]

  # After the steps, each at the turn given: a timer and an event set, and
  # checked a week later, when the timer has run out and the event has passed.
  @later_steps [
    {:set, "timer.set", @slots, ~U[2026-10-12 09:00:00Z]},
    {:set, "calendar.schedule", @slots, ~U[2026-10-12 09:00:00Z]},
    {:finished, "timer.check", %{}, ~U[2026-10-19 09:00:00Z]},
    {:none_upcoming, "calendar.query", %{}, ~U[2026-10-19 09:00:00Z]}
  ]

  defp domain_file(domain) do
    "knowledge/domains/#{domain}.json"
    |> Brain.priv_path()
    |> File.read!()
    |> Jason.decode!()
  end

  # Every template of the domain's enriched frames with its placeholders filled
  # from `fields`, a list as its items joined by ", ".
  defp enriched_sentences(config, fields) do
    values = Map.new(fields, fn {field, value} -> {Atom.to_string(field), value} end)

    for {_frame, frame_config} <- Map.fetch!(config, "enriched_response_frames"),
        template <- Map.fetch!(frame_config, "templates") do
      Regex.replace(~r/\$([a-z_]+)/, template, fn whole, name ->
        case Map.fetch(values, name) do
          {:ok, value} when is_list(value) -> Enum.join(value, ", ")
          {:ok, value} -> to_string(value)
          :error -> whole
        end
      end)
    end
  end

  # The reply without the acknowledgment put before it, with its first letter
  # as the template has it.
  defp without_acknowledgment(text, acknowledgments) do
    case Enum.find(acknowledgments, &String.starts_with?(text, &1 <> " ")) do
      nil ->
        text

      acknowledgment ->
        rest = String.replace_prefix(text, acknowledgment <> " ", "")
        String.upcase(String.first(rest)) <> String.slice(rest, 1..-1//1)
    end
  end

  # When the items set with `@slots` have come due: 7 am on the Tuesday after
  # the turn.
  @all_due ~U[2026-10-13 07:00:00Z]

  test "the steps dispatch every scheduling operation" do
    assert @steps |> Enum.reject(&(&1 == :due)) |> Enum.map(&elem(&1, 1)) |> Enum.uniq() |> Enum.sort() ==
             SchedulingDriver.intents()
  end

  test "every reply to every scheduling operation and outcome is worded from a frame written for it, and reads whole",
       %{world: world} do
    steps = Enum.map(@steps, fn step -> if step == :due, do: :due, else: Tuple.insert_at(step, 3, @turn_time) end)

    for step <- steps ++ @later_steps do
      case step do
        # The scheduler's clock reaches the items' time, and they fire.
        :due ->
          Brain.Test.ManualClock.set!(@all_due)
          :ok = LocalScheduler.wake()

        {outcome, intent, slots, turn_time} ->
          assert_reply(outcome, intent, Dispatcher.dispatch(intent, slots, turn(world, turn_time)))
      end
    end
  end

  defp assert_reply(outcome, intent, result) do
    case {outcome, result} do
      # A snooze with no alarm ringing, or with no length said, is answered as
      # a service reply, not from a domain frame.
      {error, result} when error in [:no_ringing_alarm, :missing_duration] ->
        assert result == {:error, error}

      {_outcome, {:ok, fields}} ->
        config = domain_file(intent_domain(intent))
        sentences = enriched_sentences(config, fields)
        acknowledgments = Map.fetch!(config, "acknowledgment_prefixes")

        for text <- replies(intent, %{}, fields, 0.95) do
          context = "#{intent} (#{outcome}) returned #{inspect(fields)} and was answered #{inspect(text)}"

          assert without_acknowledgment(text, acknowledgments) in sentences,
                 "#{context}, which no enriched frame of #{intent_domain(intent)}.json words"

          refute text =~ "$", "#{context}, with a placeholder left in"
          refute text =~ "  ", "#{context}, with a doubled space"
          refute text =~ " .", "#{context}, with a space before a period"
          refute text =~ ~r/\b(true|false)\b/i, "#{context}, with a boolean printed"
        end

      {_outcome, other} ->
        flunk("#{intent} (#{outcome}) returned #{inspect(other)}")
    end
  end
end
