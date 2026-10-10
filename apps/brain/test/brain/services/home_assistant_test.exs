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

  describe "as a scheduling driver" do
    test "reads every scheduling operation's intent as an action or a query, so none is unsupported" do
      for intent <- Brain.Services.SchedulingDriver.intents() do
        refute HomeAssistant.classify(intent) == :unsupported, intent
      end
    end

    test "calls a service for setting, snoozing, creating, scheduling and cancelling, and only reads for checking and querying" do
      for intent <- ["alarm.set", "alarm.snooze", "timer.set", "reminder.create", "calendar.schedule", "calendar.cancel"] do
        assert HomeAssistant.writes?(intent), intent
      end

      for intent <- ["timer.check", "calendar.query"] do
        refute HomeAssistant.writes?(intent), intent
      end
    end

    test "does not claim the scheduling domains as its own; they reach it only as a world's driver" do
      for domain <- Brain.Services.SchedulingDriver.domains() do
        refute domain in HomeAssistant.supported_domains(), domain
      end
    end
  end

  describe "enrich/4" do
    test "an unsupported intent is reported, and Home Assistant is not called" do
      credentials = %{url: "http://home-assistant.invalid", access_token: "unused"}

      assert {:error, {:unsupported_intent, "smarthome.device.volume.mute"}} =
               HomeAssistant.enrich("smarthome.device.volume.mute", %{}, credentials, "default")
    end
  end
end
