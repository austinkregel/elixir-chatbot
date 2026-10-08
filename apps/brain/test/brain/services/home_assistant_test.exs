defmodule Brain.Services.HomeAssistantTest do
  @moduledoc """
  How Home Assistant reads an intent against `priv/services/ha_action_verbs.json`:
  an action it calls a service for, a query that reads state, or an intent it
  does not handle. The intents are ones `priv/analysis/intent_registry.json`
  defines in Home Assistant's domains.

  An intent with no verb in the map used to be treated as a state query, so
  `smarthome.device.volume.mute` asked Home Assistant for media-player states.
  """
  use ExUnit.Case, async: true

  alias Brain.Services.HomeAssistant

  describe "classify/1" do
    test "a verb mapped to services is an action" do
      assert {:action, "switch", ["turn_on", "turn_off", "toggle"]} = HomeAssistant.classify("smarthome.switch")
      assert {:action, "cancel", _} = HomeAssistant.classify("calendar.cancel")
      assert {:action, "create", _} = HomeAssistant.classify("reminder.create")
    end

    test "a two-segment verb is preferred over its last segment" do
      assert {:action, "player.pause", ["media_pause"]} = HomeAssistant.classify("music.player.pause")
      assert {:action, "player.play", ["media_play"]} = HomeAssistant.classify("music.player.play")
    end

    test "a verb mapped to null is a query" do
      assert :query = HomeAssistant.classify("smarthome.device_check")
      assert :query = HomeAssistant.classify("music.search")
      assert :query = HomeAssistant.classify("timer.check")
    end

    test "a query verb anywhere in the name wins over an action verb" do
      assert :query = HomeAssistant.classify("smarthome.device.switch.check")
      assert :query = HomeAssistant.classify("smarthome.device.switch.check.on")
      assert :query = HomeAssistant.classify("smarthome.locks.check.lock")
    end

    test "an intent with no verb in the map is unsupported, not a query" do
      assert :unsupported = HomeAssistant.classify("smarthome.device.volume.mute")
      assert :unsupported = HomeAssistant.classify("smarthome.locks.lock")
      assert :unsupported = HomeAssistant.classify("device.control")
    end
  end

  describe "writes?/1" do
    test "is true only for an action" do
      assert HomeAssistant.writes?("smarthome.switch")
      refute HomeAssistant.writes?("smarthome.device.switch.check.on")
      refute HomeAssistant.writes?("smarthome.device.volume.mute")
    end
  end

  describe "enrich/3" do
    test "an unsupported intent is reported, and Home Assistant is not called" do
      credentials = %{url: "http://home-assistant.invalid", access_token: "unused"}

      assert {:error, {:unsupported_intent, "smarthome.device.volume.mute"}} =
               HomeAssistant.enrich("smarthome.device.volume.mute", %{}, credentials)
    end
  end
end
