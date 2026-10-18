defmodule Brain.Analysis.IntentUtilsTest do
  use ExUnit.Case, async: true

  alias Brain.Analysis.IntentUtils

  test "same_domain_prefix? for matching domains" do
    assert IntentUtils.same_domain_prefix?("weather.query", "weather.condition")
    refute IntentUtils.same_domain_prefix?("weather.query", "smalltalk.greet")
  end

  test "domain_prefix/1" do
    assert IntentUtils.domain_prefix("calendar.query") == "calendar"
  end
end
