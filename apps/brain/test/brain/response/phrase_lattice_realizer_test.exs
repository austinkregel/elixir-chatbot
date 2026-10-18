defmodule Brain.Response.PhraseLatticeRealizerTest do
  use ExUnit.Case, async: false

  alias Brain.Response.{PhraseLatticeRealizer, Primitive, Synthesizer}

  # The world the test suite runs in; the inventory is the one the application
  # loaded (`mix gen_lattice_data`).
  @world_id "default"

  # What `Brain.Services.LocalScheduler.check_timer/3` returns for each timer a
  # check can find, and `cancel_timer/3` for a cancel that removed one.
  @checks %{
    none: %{timer_status: "none", duration: "", time_left: []},
    running: %{timer_status: "running", duration: "5 minutes", time_left: ["4 minutes", "59 seconds"]},
    finished: %{timer_status: "finished", duration: "5 minutes", time_left: []}
  }
  @cancelled %{timer_status: "cancelled", cancelled_count: 1, timer_durations: []}

  # The timer domain's cancel frame (timer.json "cancel").
  @cancel_wording ["Timer cancelled.", "I've stopped the timer."]

  # The operation the scheduler performed for each intent
  # (`Brain.Services.SchedulingDriver.operation/1`).
  @operations %{"timer.check" => :check_timer, "timer.cancel" => :cancel_timer}

  defp realize(intent, text, enriched_data) do
    primitive = Primitive.new(:content, :enriched, %{intent: intent})

    PhraseLatticeRealizer.realize([primitive],
      world_id: @world_id,
      intent: intent,
      analysis: %{intent: intent, text: text, entities: [], confidence: 0.9, speech_act: %{category: :directive}},
      unified_context: %{
        enrichment: %{
          enriched_data: enriched_data,
          enrichment_status: :success,
          enrichment_operation: Map.fetch!(@operations, intent)
        }
      }
    )
  end

  defp chosen(data, operation), do: Synthesizer.chosen_frame("timer", [], %{enriched_data: data, operation: operation})

  describe "a domain frame is offered only for a turn the synthesizer would word from it" do
    test "a timer check is never worded as the cancel it did not do, whatever it found" do
      for {found, data} <- @checks do
        assert {:ok, _rendered, reply} = realize("timer.check", "How long on the timer", data)

        refute reply in @cancel_wording,
               "a check that found #{found} was worded as a cancel: #{inspect(reply)}"

        refute match?({:response, "cancel"}, chosen(data, :check_timer))
      end
    end

    test "a turn misread as a timer check is never worded as a cancel" do
      for {found, data} <- @checks do
        assert {:ok, _rendered, reply} = realize("timer.check", "describe what fetch_data does", data)

        refute reply in @cancel_wording,
               "a misread turn whose check found #{found} was worded as a cancel: #{inspect(reply)}"
      end
    end

    test "a check that found no timer says so from the frame the synthesizer chooses" do
      assert chosen(@checks.none, :check_timer) == {:enriched, "none_found"}

      assert {:ok, _rendered, reply} = realize("timer.check", "How long on the timer", @checks.none)
      refute reply in ["There's a timer running.", "A timer is running.", "Timer is running."]
    end

    # The synthesizer's cancelled_one frame. The timer.cancel intent's own
    # template (templates.json, "Canceling the timer.") is not offered: a
    # domain frame is chosen for the turn, and it is the reply.
    test "a cancel that did cancel is worded as one" do
      assert chosen(@cancelled, :cancel_timer) == {:enriched, "cancelled_one"}

      assert {:ok, _rendered, reply} = realize("timer.cancel", "Cancel the timer", @cancelled)
      assert reply in ["Cancelled the timer.", "The timer is cancelled."]
    end
  end

  describe "a check's reply reports what the check found" do
    # Each check's result frame (timer.json), every template of it.
    @found %{
      none: ["There's no timer running.", "No timer is running."],
      running: [
        "There's a timer running, set for 5 minutes, with 4 minutes, 59 seconds left.",
        "A timer set for 5 minutes is running: 4 minutes, 59 seconds left."
      ],
      finished: ["The timer set for 5 minutes has run out.", "Your timer for 5 minutes has run out."]
    }

    test "the result frame outranks the timer.check intent's own template, which announces a check" do
      for {found, data} <- @checks do
        assert {:ok, _rendered, reply} = realize("timer.check", "How long on the timer", data)
        assert reply in @found[found], "a check that found #{found} was worded #{inspect(reply)}"
      end
    end

    # time_left is a list, one part per unit (`LocalScheduler.check_timer/3`):
    # the lattice says it as the synthesizer does (`Formatting.sayable/1`).
    test "a list the check returned fills a placeholder, as the synthesizer fills it" do
      assert chosen(@checks.running, :check_timer) == {:enriched, "running_with_time_left"}
      assert Brain.Response.Formatting.sayable(@checks.running.time_left) == "4 minutes, 59 seconds"
      assert Brain.Response.Formatting.sayable([]) == nil
      assert Brain.Response.Formatting.sayable(["4 minutes", " "]) == nil
      assert Brain.Response.Formatting.sayable(:unknown) == nil
    end
  end
end
