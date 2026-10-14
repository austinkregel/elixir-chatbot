defmodule Brain.Services.FreeTextRemindersTest do
  # A reminder is set, checked and cancelled by what it is about: the text each
  # turn names is captured into its reminder_text slot
  # (`Brain.Analysis.SlotDetector` with the loaded
  # `Brain.Analysis.FreeTextCapture` model), and the slots are dispatched to
  # the local scheduler. Items are stored in Atlas, inside this test's sandbox
  # transaction, in a fresh `test_` world cleared when the test ends.
  use ExUnit.Case, async: false

  alias Brain.Analysis.SlotDetector
  alias Brain.Services.{Dispatcher, LocalScheduler}

  # The turn every utterance below is said in, read in UTC.
  @turn_time ~U[2026-10-12 09:00:00Z]

  setup tags do
    owner = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)
    on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)

    world = "test_free_text_reminders_#{System.unique_integer([:positive])}"
    web_user = Atlas.WebUsers.create!(world).id
    on_exit(fn -> LocalScheduler.clear_world(world) end)
    :ok = Dispatcher.assign_scheduling_driver(world, LocalScheduler)

    %{context: %{world_id: world, turn_time: @turn_time, time_zone: Brain.Temporal.Zone.utc(), web_user_id: web_user}}
  end

  # The slots `text` fills for `intent`, as the response layer dispatches them:
  # each filled slot's value under its name.
  defp said(intent, text) do
    intent
    |> SlotDetector.detect([], side_effects: false, text: text, language: "en")
    |> Map.fetch!(:filled_slots)
    |> Map.new(fn {slot, %{value: value}} -> {String.to_existing_atom(slot), value} end)
  end

  test "a reminder set by what it is about is found by a check that names it and removed by a cancel that names it",
       %{context: context} do
    created = said("reminder.create", "Can you remind me to call John")
    assert created == %{reminder_text: "call John"}

    assert {:ok, %{reminder_set: true, reminder_text: "call John"}} = Dispatcher.dispatch("reminder.create", created, context)
    {:ok, _} = Dispatcher.dispatch("reminder.create", %{reminder_text: "feed the cat"}, context)

    assert {:ok, %{reminder_texts: ["call John"], reminder_count: 1}} =
             Dispatcher.dispatch("reminder.check", said("reminder.check", "Did I set a reminder to call John"), context)

    assert {:ok, %{reminder_texts: ["call John"], cancelled_count: 1}} =
             Dispatcher.dispatch("reminder.cancel", said("reminder.cancel", "cancel my reminder to call John"), context)

    assert {:ok, %{reminder_texts: ["feed the cat"], reminder_count: 1}} = Dispatcher.dispatch("reminder.check", %{}, context)
  end
end
