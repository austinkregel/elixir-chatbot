defmodule Mix.Tasks.RebuildGoldStandardTest do
  use ExUnit.Case, async: true

  alias Brain.Corpus.{Export, Label}
  alias Mix.Tasks.RebuildGoldStandard

  # One usersays row as the task reads it: filed under its file's label, with
  # the follow-up pair when the row came from a contextual variant.
  defp phrase(text, stem, followup \\ nil) do
    %{
      text: text,
      norm: Export.normalise(text),
      label: Label.canonical(stem),
      source_file: stem <> Export.usersays_suffix(),
      provenance: "dialogflow",
      followup:
        case followup do
          nil -> nil
          {context, variant} -> %{"followup_context" => context, "followup_variant" => variant}
        end
    }
  end

  defp keys_in_file_order(json) do
    json
    |> Jason.decode!(objects: :ordered_objects)
    |> Enum.map(fn %Jason.OrderedObject{values: values} -> Enum.map(values, &elem(&1, 0)) end)
  end

  describe "build_entries/1 on a text more than one intent claims" do
    test "source_file and follow-up describe the row's intent, not another candidate" do
      phrases = [
        phrase("back", "music.player.skip_backward"),
        phrase("back", "news.search", {"newssearch_followup", "previous"}),
        phrase("back", "web.search", {"websearch_followup", "previous"})
      ]

      {[entry], [conflict]} = RebuildGoldStandard.build_entries(phrases)

      assert entry == %{
               "text" => "back",
               "intent" => "music.player.skip_backward",
               "labeled_by" => "dialogflow",
               "source_file" => "music.player.skip_backward_usersays_en.json",
               "status" => "needs_review",
               "candidates" => ["music.player.skip_backward", "news.search", "web.search"]
             }

      assert conflict["sources"] == [
               "music.player.skip_backward_usersays_en.json",
               "news.search_usersays_en.json",
               "web.search_usersays_en.json"
             ]
    end

    test "an intent's file sorts before its sub-intent's, whatever the filename suffix would say" do
      phrases = [
        phrase("speakers too", "smarthome.device.volume.check", {"volume_check", "context_volume_check - comment_device"}),
        phrase("speakers too", "smarthome.device.volume", {"device_volume", "context_device_volume - comment_device"})
      ]

      {[entry], [conflict]} = RebuildGoldStandard.build_entries(phrases)

      assert entry["intent"] == "smarthome.device.volume"
      assert entry["source_file"] == "smarthome.device.volume_usersays_en.json"
      assert entry["followup_context"] == "device_volume"
      assert entry["followup_variant"] == "context_device_volume - comment_device"

      assert conflict["sources"] == [
               "smarthome.device.volume_usersays_en.json",
               "smarthome.device.volume.check_usersays_en.json"
             ]
    end

    test "a candidate's own sub-intent file is not named as the row's source" do
      phrases = [
        phrase("too warm", "smarthome.heating.down.implicit"),
        phrase("too warm", "smarthome.heating.down", {"heating", "context_heating - comment_not cold enough"})
      ]

      {[entry], _} = RebuildGoldStandard.build_entries(phrases)

      assert entry["intent"] == "smarthome.heating.down"
      assert entry["source_file"] == "smarthome.heating.down_usersays_en.json"
      assert entry["followup_context"] == "heating"
    end
  end

  describe "build_entries/1 on a text one intent claims" do
    test "the follow-up occurrence wins over the standalone one" do
      phrases = [
        phrase("it's too bright", "smarthome.device.brightness.down"),
        phrase("it's too bright", "smarthome.lights.brightness.down", {"lights_brightness", "context_lights_brightness - comment_not dim enough"})
      ]

      {[entry], []} = RebuildGoldStandard.build_entries(phrases)

      assert entry["intent"] == "smarthome.device.brightness.down"
      assert entry["followup_context"] == "lights_brightness"
    end

    test "follow-ups tie on variant name, then on context name" do
      phrases = [
        phrase("next", "smarthome.lights.x", {"z_context", "same variant"}),
        phrase("next", "smarthome.device.x", {"a_context", "same variant"})
      ]

      {[entry], []} = RebuildGoldStandard.build_entries(phrases)

      assert entry["followup_context"] == "a_context"
    end
  end

  describe "determinism" do
    setup do
      phrases = [
        phrase("back", "music.player.skip_backward"),
        phrase("back", "news.search", {"newssearch_followup", "previous"}),
        phrase("back", "web.search", {"websearch_followup", "previous"}),
        phrase("speakers too", "smarthome.device.volume.check", {"volume_check", "context_volume_check - comment_device"}),
        phrase("speakers too", "smarthome.device.volume", {"device_volume", "context_device_volume - comment_device"}),
        phrase("next", "smarthome.lights.x", {"z_context", "same variant"}),
        phrase("next", "smarthome.device.x", {"a_context", "same variant"}),
        phrase("set an alarm", "alarm.set")
      ]

      %{phrases: phrases}
    end

    test "the files are the same bytes whatever order the phrases were read in", %{phrases: phrases} do
      encode = fn input ->
        {entries, conflicts} = RebuildGoldStandard.build_entries(input)
        {RebuildGoldStandard.encode_entries!(entries), RebuildGoldStandard.encode_conflicts!(conflicts)}
      end

      reference = encode.(phrases)

      assert encode.(Enum.reverse(phrases)) == reference
      assert encode.(Enum.sort_by(phrases, & &1.source_file)) == reference
      assert encode.(Enum.sort_by(phrases, & &1.source_file, :desc)) == reference
    end

    test "gold rows are written in the documented key order", %{phrases: phrases} do
      {entries, _} = RebuildGoldStandard.build_entries(phrases)

      assert keys_in_file_order(RebuildGoldStandard.encode_entries!(entries)) == [
               ~w(candidates intent labeled_by source_file status text),
               ~w(intent labeled_by followup_context followup_variant source_file text),
               ~w(intent labeled_by source_file text),
               ~w(candidates intent labeled_by followup_context followup_variant source_file status text)
             ]
    end

    test "ambiguous texts are written in the documented key order", %{phrases: phrases} do
      {_, conflicts} = RebuildGoldStandard.build_entries(phrases)

      assert keys_in_file_order(RebuildGoldStandard.encode_conflicts!(conflicts)) == [
               ~w(text sources candidates),
               ~w(text sources candidates)
             ]
    end

    test "the key order does not depend on how the map was built" do
      pairs = [
        {"text", "set an alarm"},
        {"intent", "alarm.set"},
        {"labeled_by", "dialogflow"},
        {"source_file", "alarm.set_usersays_en.json"}
      ]

      expected = """
      [
        {
          "intent": "alarm.set",
          "labeled_by": "dialogflow",
          "source_file": "alarm.set_usersays_en.json",
          "text": "set an alarm"
        }
      ]
      """

      assert RebuildGoldStandard.encode_entries!([Map.new(pairs)]) == expected
      assert RebuildGoldStandard.encode_entries!([Map.new(Enum.reverse(pairs))]) == expected
    end

    test "a key outside the written order raises" do
      row = %{"text" => "x", "intent" => "a.b", "confidence" => 1.0}

      assert_raise ArgumentError, ~r/"confidence"/, fn -> RebuildGoldStandard.encode_entries!([row]) end
    end
  end
end
