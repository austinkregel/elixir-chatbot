defmodule Brain.Services.SchedulingRepliesTest do
  use ExUnit.Case, async: false

  alias Brain.Response.Synthesizer
  alias Brain.Services.{Dispatcher, LocalScheduler, SchedulingDriver}

  # The replies to alarm, reminder, timer and calendar intents, worded by
  # `Synthesizer.synthesize_from_domain/5` from what the world's scheduling
  # driver returned when the intent was dispatched. Each test works in a fresh
  # `test_` world that has chosen `LocalScheduler`.

  setup tags do
    owner = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)
    on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)

    world = "test_scheduling_replies_#{System.unique_integer([:positive])}"
    on_exit(fn -> LocalScheduler.clear_world(world) end)

    assert :ok = Dispatcher.assign_scheduling_driver(world, LocalScheduler)
    %{world: world}
  end

  defp intent_domain(intent), do: intent |> String.split(".", parts: 2) |> hd()

  defp dispatch!(intent, slots, world) do
    assert {:ok, fields} = Dispatcher.dispatch(intent, slots, %{world_id: world})
    fields
  end

  # The reply to a dispatched intent, from the fields the driver returned, with
  # the slots it was given as the turn's entities. Below a confidence of 0.7 no
  # acknowledgment is put before it, so the reply is a template as filled.
  defp reply(intent, slots, fields, confidence) do
    entities = for {slot, value} <- slots, do: %{entity_type: Atom.to_string(slot), value: value}

    assert {:ok, text} =
             Synthesizer.synthesize_from_domain(intent_domain(intent), intent, entities, confidence,
               context: %{enriched_data: fields}
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

    test "a timer check that finds none says so, and one that finds a timer gives the duration it was set with",
         %{world: world} do
      fields = dispatch!("timer.check", %{}, world)

      for text <- replies("timer.check", %{}, fields, 0.5) do
        assert text in ["There's no timer running.", "No timer is running."]
      end

      dispatch!("timer.set", %{duration: "5 minutes"}, world)
      fields = dispatch!("timer.check", %{}, world)
      assert fields == %{timer_status: "running", duration: "5 minutes"}

      for text <- replies("timer.check", %{}, fields, 0.5) do
        assert text in ["There's a timer running, set for 5 minutes.", "A timer is running, set for 5 minutes."]
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
  # one item and then two (one of them described by nothing); cancels that
  # remove one item and then several; and a snooze, which finds no ringing
  # alarm. A calendar cancel removes exactly one event, so it is named.
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
    {:no_ringing_alarm, "alarm.snooze", %{}},
    {:cancelled, "alarm.cancel", %{}},
    {:cancelled, "reminder.cancel", %{}},
    {:cancelled, "timer.cancel", %{}},
    {:cancelled, "calendar.cancel", %{event: "team standup"}}
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

  test "the steps dispatch every scheduling operation" do
    assert @steps |> Enum.map(&elem(&1, 1)) |> Enum.uniq() |> Enum.sort() == SchedulingDriver.intents()
  end

  test "every reply to every scheduling operation and outcome is worded from a frame written for it, and reads whole",
       %{world: world} do
    for {outcome, intent, slots} <- @steps do
      case {outcome, Dispatcher.dispatch(intent, slots, %{world_id: world})} do
        # Alarms are stored and never rung (`Brain.Services.LocalScheduler`);
        # the error is answered as a service reply, not from a domain frame.
        {:no_ringing_alarm, result} ->
          assert result == {:error, :no_ringing_alarm}

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
end
