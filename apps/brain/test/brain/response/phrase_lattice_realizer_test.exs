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
    running: %{timer_status: "running", duration: "5 minutes", time_left: "4 minutes"},
    finished: %{timer_status: "finished", duration: "5 minutes", time_left: []}
  }
  @cancelled %{timer_status: "cancelled", cancelled_count: 1, timer_durations: []}

  # The timer domain's cancel frame (timer.json "cancel").
  @cancel_wording ["Timer cancelled.", "I've stopped the timer."]

  defp realize(intent, text, enriched_data) do
    primitive = Primitive.new(:content, :enriched, %{intent: intent})

    PhraseLatticeRealizer.realize([primitive],
      world_id: @world_id,
      intent: intent,
      analysis: %{intent: intent, text: text, entities: [], confidence: 0.9, speech_act: %{category: :directive}},
      unified_context: %{enrichment: %{enriched_data: enriched_data, enrichment_status: :success}}
    )
  end

  describe "a domain frame is offered only for a turn the synthesizer would word from it" do
    test "a timer check is never worded as the cancel it did not do, whatever it found" do
      for {found, data} <- @checks do
        assert {:ok, _rendered, reply} = realize("timer.check", "How long on the timer", data)

        refute reply in @cancel_wording,
               "a check that found #{found} was worded as a cancel: #{inspect(reply)}"

        refute match?({:response, "cancel"}, Synthesizer.chosen_frame("timer", [], data))
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
      assert Synthesizer.chosen_frame("timer", [], @checks.none) == {:enriched, "none_found"}

      assert {:ok, _rendered, reply} = realize("timer.check", "How long on the timer", @checks.none)
      refute reply in ["There's a timer running.", "A timer is running.", "Timer is running."]
    end

    test "a cancel that did cancel is worded as one" do
      assert Synthesizer.chosen_frame("timer", [], @cancelled) == {:enriched, "cancelled_one"}

      assert {:ok, _rendered, reply} = realize("timer.cancel", "Cancel the timer", @cancelled)

      # The synthesizer's cancelled_one frame, or the timer.cancel intent's own
      # template (templates.json), which the lattice offers that intent.
      assert reply in ["Cancelled the timer.", "The timer is cancelled.", "Canceling the timer."]
    end
  end
end
