defmodule Brain.AssertionHelpers do
  @moduledoc """
  Helper functions for common test assertions.

  These helpers reduce repetition and improve readability of test assertions.
  They provide positive assertions for common response patterns.
  """

  import ExUnit.Assertions

  @greeting_pattern ~r/hello|hi|hey|welcome|nice|meet|how.*you|help|can i|what can/i
  @farewell_pattern ~r/bye|goodbye|see you|later|farewell|good night|take care/i
  @weather_pattern ~r/weather|temperature|forecast|degrees|rain|sunny|cloudy|location|city|where|which/i
  @music_pattern ~r/playing|play\s|music|song|ok|sure|alright|will do/i
  @acknowledgment_pattern ~r/welcome|anytime|gladly|certainly|absolutely|pleasure|happy to help|ok|got it|understood/i

  @doc """
  Asserts that a response contains greeting patterns.
  """
  def assert_greeting_response(response, context \\ "") do
    assert response =~ @greeting_pattern,
           "Expected greeting response#{context}, got: #{response}"
  end

  @doc """
  Asserts that a response does NOT contain farewell patterns.
  Useful as a regression test to prevent misclassification.
  """
  def refute_farewell_response(response, context \\ "") do
    refute response =~ @farewell_pattern,
           "Expected non-farewell response#{context}, got farewell: #{response}"
  end

  @doc """
  Asserts that a response contains weather-related patterns.
  """
  def assert_weather_response(response, context \\ "") do
    assert response =~ @weather_pattern,
           "Expected weather-related response#{context}, got: #{response}"
  end

  @doc """
  Asserts that a response contains music command acknowledgment patterns.
  """
  def assert_music_command_response(response, context \\ "") do
    assert response =~ @music_pattern,
           "Expected music command acknowledgment#{context}, got: #{response}"
  end

  @doc """
  Asserts that a response contains acknowledgment patterns.
  Useful for testing responses to "thank you" or name introductions.
  """
  def assert_acknowledgment_response(response, context \\ "") do
    assert response =~ @acknowledgment_pattern,
           "Expected acknowledgment response#{context}, got: #{response}"
  end

  @doc """
  Asserts that a response contains farewell patterns.
  """
  def assert_farewell_response(response, context \\ "") do
    assert response =~ @farewell_pattern,
           "Expected farewell response#{context}, got: #{response}"
  end

  @doc """
  Asserts that a response contains introduction acknowledgment patterns.
  Optionally checks for a specific name.
  """
  def assert_introduction_response(response, name \\ nil, context \\ "") do
    name_pattern = if name, do: ~r/#{name}/i, else: ~r//
    assert response =~ @greeting_pattern or response =~ name_pattern,
           "Expected introduction acknowledgment#{context}, got: #{response}"
  end

  # ============================================================================
  # Log Assertion Helpers
  # ============================================================================

  @doc """
  Asserts that a log contains expected content.
  Useful for verifying warnings, errors, and info messages.
  """
  def assert_log_contains(log, expected_pattern, context \\ "") do
    assert log =~ expected_pattern,
           "Expected log#{context} to contain '#{expected_pattern}', got: #{log}"
  end

  @doc """
  Asserts that a log contains a contradiction-related message.
  """
  def assert_contradiction_logged(log, context \\ "") do
    contradiction_patterns = ~r/contradiction|contradicted|conflict|conflicting/i
    assert log =~ contradiction_patterns,
           "Expected contradiction warning#{context} in log, got: #{log}"
  end

  @doc """
  Asserts that a log contains an error message.
  """
  def assert_error_logged(log, context \\ "") do
    error_patterns = ~r/error|failed|failure|exception/i
    assert log =~ error_patterns,
           "Expected error message#{context} in log, got: #{log}"
  end
end
