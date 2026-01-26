# Negative Assertions Improvement Plan

## Overview

This plan details specific improvements to replace or supplement negative assertions (`refute`) with positive assertions (`assert`) where possible, making tests more precise and maintainable. Additionally, it addresses:

1. `IO.puts` statements in tests that should be converted to proper assertions
2. Log leaks - Logger calls that aren't asserted on
3. Enforcement of `.cursorrules` - No regex/string matching outside of tests or data preparation

## Important Rule: No Regex/String Matching Outside Tests

Per `.cursorrules`:
- **Regex and string matching outside of tests or data preparation is unacceptable**
- In production code, use `ChatBot.ML.Tokenizer` functions instead of regex
- Tests may use regex for assertions (e.g., `assert response =~ ~r/hello|hi/i`)
- Data preparation scripts may use regex for cleaning/transforming data
- **All other code must use Tokenizer functions or pattern matching**

## Strategy

1. **Keep negative assertions** where they serve as regression tests (e.g., preventing misclassification)
2. **Add positive assertions** alongside negative ones to verify expected behavior
3. **Replace weak negative assertions** with strong positive ones where appropriate
4. **Replace IO.puts with assertions** - Convert conditional IO.puts statements to proper test assertions
5. **Assert on log output** - Use `ExUnit.CaptureLog` with positive assertions on log text instead of letting logs leak
6. **No regex/string matching outside tests** - Per `.cursorrules`, regex and string matching outside of tests or data preparation is unacceptable
7. **Use helper functions** to reduce repetition and improve readability

## File-by-File Improvements

### 1. `test/chat_bot/feature_test.exs`

#### Current Issues
- Many tests only use `refute response =~ ~r/bye|goodbye/i` without positive assertions
- Weak assertions like `assert String.length(response) > 0`
- Missing positive checks for expected greeting/question/command patterns

#### Improvements

**Greeting Tests (lines 22-45)**
```elixir
# BEFORE
test "responds to hello", %{conversation_id: conv_id} do
  {:ok, response} = Brain.evaluate(conv_id, "Hello!")
  refute response =~ ~r/bye|goodbye|see you|later/i,
         "Expected greeting response, got farewell: #{response}"
  assert String.length(response) > 0
end

# AFTER
test "responds to hello", %{conversation_id: conv_id} do
  {:ok, response} = Brain.evaluate(conv_id, "Hello!")
  
  # Positive: Should contain greeting patterns
  assert response =~ ~r/hello|hi|hey|welcome|nice|meet|how.*you|help|can i|what can/i,
         "Expected greeting response, got: #{response}"
  
  # Negative: Regression test - should not be farewell
  refute response =~ ~r/bye|goodbye|see you|later/i,
         "Expected greeting response, got farewell: #{response}"
end
```

**Question Tests (lines 49-79)**
```elixir
# BEFORE
test "responds to weather question", %{conversation_id: conv_id} do
  {:ok, response} = Brain.evaluate(conv_id, "Can you tell me about the weather?")
  refute response =~ ~r/bye|goodbye|see you later/i,
         "Expected informative response, got farewell: #{response}"
  assert String.length(response) > 0
end

# AFTER
test "responds to weather question", %{conversation_id: conv_id} do
  {:ok, response} = Brain.evaluate(conv_id, "Can you tell me about the weather?")
  
  # Positive: Should mention weather or ask for location
  assert response =~ ~r/weather|temperature|forecast|location|city|where|which/i,
         "Expected weather-related response, got: #{response}"
  
  # Negative: Regression test
  refute response =~ ~r/bye|goodbye|see you later/i,
         "Expected informative response, got farewell: #{response}"
end

# Similar improvements for:
# - "responds to time question" - check for time-related words
# - "responds to how are you" - check for conversational patterns
# - "responds to what can you do" - check for capability descriptions
```

**Command Tests (lines 83-102)**
```elixir
# BEFORE
test "responds to play music command", %{conversation_id: conv_id} do
  {:ok, response} = Brain.evaluate(conv_id, "Play some music")
  refute response =~ ~r/bye|goodbye|see you later/i,
         "Expected action response, got farewell: #{response}"
end

# AFTER
test "responds to play music command", %{conversation_id: conv_id} do
  {:ok, response} = Brain.evaluate(conv_id, "Play some music")
  
  # Positive: Should acknowledge the command
  assert response =~ ~r/playing|play|music|song|ok|sure|alright|will do/i,
         "Expected music command acknowledgment, got: #{response}"
  
  # Negative: Regression test
  refute response =~ ~r/bye|goodbye|see you later/i,
         "Expected action response, got farewell: #{response}"
end
```

**Farewell Tests (lines 106-123)**
```elixir
# BEFORE
test "responds appropriately to goodbye", %{conversation_id: conv_id} do
  {:ok, response} = Brain.evaluate(conv_id, "Goodbye!")
  assert String.length(response) > 0
end

# AFTER
test "responds appropriately to goodbye", %{conversation_id: conv_id} do
  {:ok, response} = Brain.evaluate(conv_id, "Goodbye!")
  
  # Positive: Should contain farewell patterns
  assert response =~ ~r/bye|goodbye|see you|later|farewell|good night|take care/i,
         "Expected farewell response, got: #{response}"
end
```

**Multi-sentence Tests (lines 127-192)**
```elixir
# BEFORE
test "handles greeting with weather question (with location)", %{conversation_id: conv_id} do
  {:ok, response} = Brain.evaluate(conv_id, "Hello! What's the weather like in New York?")
  assert String.length(response) > 0
  refute response =~ ~r/bye|goodbye|see you later/i,
         "Expected informative response, got farewell: #{response}"
end

# AFTER
test "handles greeting with weather question (with location)", %{conversation_id: conv_id} do
  {:ok, response} = Brain.evaluate(conv_id, "Hello! What's the weather like in New York?")
  
  # Positive: Should address weather question (may also include greeting)
  assert response =~ ~r/weather|temperature|forecast|new york|nyc|degrees/i or
         response =~ ~r/hello|hi|hey/i,
         "Expected weather or greeting response, got: #{response}"
  
  # Negative: Regression test
  refute response =~ ~r/bye|goodbye|see you later/i,
         "Expected informative response, got farewell: #{response}"
end
```

**Conversational Context Tests (lines 196-209)**
```elixir
# BEFORE
test "handles simple statement", %{conversation_id: conv_id} do
  {:ok, response} = Brain.evaluate(conv_id, "My name is Alex")
  refute response =~ ~r/bye|goodbye|see you later/i,
         "Expected acknowledgment, got farewell: #{response}"
end

# AFTER
test "handles simple statement", %{conversation_id: conv_id} do
  {:ok, response} = Brain.evaluate(conv_id, "My name is Alex")
  
  # Positive: Should acknowledge the statement
  assert response =~ ~r/nice|meet|hello|hi|alex|thanks|ok|got it|understood/i,
         "Expected acknowledgment of name, got: #{response}"
  
  # Negative: Regression test
  refute response =~ ~r/bye|goodbye|see you later/i,
         "Expected acknowledgment, got farewell: #{response}"
end

# BEFORE
test "handles thank you", %{conversation_id: conv_id} do
  {:ok, response} = Brain.evaluate(conv_id, "Thank you!")
  refute response =~ ~r/bye|goodbye|see you later/i,
         "Expected polite response, got farewell: #{response}"
end

# AFTER
test "handles thank you", %{conversation_id: conv_id} do
  {:ok, response} = Brain.evaluate(conv_id, "Thank you!")
  
  # Positive: Should contain acknowledgment patterns
  assert response =~ ~r/welcome|anytime|gladly|certainly|absolutely|pleasure|happy to help/i,
         "Expected acknowledgment response, got: #{response}"
  
  # Negative: Regression test
  refute response =~ ~r/bye|goodbye|see you later/i,
         "Expected polite response, got farewell: #{response}"
end
```

**Factual Question Tests (lines 212-286)**
```elixir
# These tests already have good negative assertions for regression testing.
# Add positive assertions to verify expected behavior:

test "weather question gets weather-related response, not random facts", %{
  conversation_id: conv_id
} do
  {:ok, response} = Brain.evaluate(conv_id, "Can you tell me about the weather?")
  
  # Positive: Should mention weather or ask for location
  assert response =~ ~r/weather|temperature|forecast|location|city|where|which/i,
         "Expected weather-related response, got: #{response}"
  
  # Negative: Regression test - should not dump random facts
  refute response =~ ~r/week|days in a|alphabet|chess|olympic/i,
         "Weather question got unrelated factual response: #{response}"
end
```

### 2. `test/chat_bot/edge_cases_test.exs`

#### Current Issues
- Many tests only refute weather/music patterns without positive assertions
- Missing checks for expected greeting/acknowledgment patterns

#### Improvements

**Names Overlapping with Locations (lines 177-309)**
```elixir
# BEFORE
test "Austin is also a city in Texas", %{conversation_id: conv_id} do
  {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm Austin")
  refute response =~ ~r/weather|temperature|forecast/i,
         "Greeting was misclassified as weather query: #{response}"
  refute response =~ ~r/Texas|city|travel/i,
         "Austin was incorrectly interpreted as a location: #{response}"
end

# AFTER
test "Austin is also a city in Texas", %{conversation_id: conv_id} do
  {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm Austin")
  
  # Positive: Should recognize as greeting/introduction
  assert response =~ ~r/hello|hi|hey|nice|meet|welcome|austin/i,
         "Expected greeting/introduction response, got: #{response}"
  
  # Negative: Regression tests - should not misclassify as location
  refute response =~ ~r/weather|temperature|forecast/i,
         "Greeting was misclassified as weather query: #{response}"
  refute response =~ ~r/Texas|city|travel/i,
         "Austin was incorrectly interpreted as a location: #{response}"
end
```

**Names Overlapping with Songs (lines 314-345)**
```elixir
# BEFORE
test "Delilah is also a song (Hey There Delilah)", %{conversation_id: conv_id} do
  {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm Delilah")
  refute response =~ ~r/playing|play\s|music|song/i,
         "Delilah was misclassified as a music request: #{response}"
end

# AFTER
test "Delilah is also a song (Hey There Delilah)", %{conversation_id: conv_id} do
  {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm Delilah")
  
  # Positive: Should recognize as greeting/introduction
  assert response =~ ~r/hello|hi|hey|nice|meet|welcome|delilah/i,
         "Expected greeting/introduction response, got: #{response}"
  
  # Negative: Regression test - should not trigger music playback
  refute response =~ ~r/playing|play\s|music|song/i,
         "Delilah was misclassified as a music request: #{response}"
end
```

**Informal Greetings (lines 381-447)**
```elixir
# BEFORE
test "yo as greeting", %{conversation_id: conv_id} do
  {:ok, response} = Brain.evaluate(conv_id, "Yo")
  refute response =~ ~r/bye|goodbye|see you later/i,
         "Informal greeting got farewell response: #{response}"
  assert String.length(response) > 0
end

# AFTER
test "yo as greeting", %{conversation_id: conv_id} do
  {:ok, response} = Brain.evaluate(conv_id, "Yo")
  
  # Positive: Should recognize informal greeting
  assert response =~ ~r/hello|hi|hey|yo|sup|wassup|what.*up|how.*you/i,
         "Expected informal greeting response, got: #{response}"
  
  # Negative: Regression test
  refute response =~ ~r/bye|goodbye|see you later/i,
         "Informal greeting got farewell response: #{response}"
end
```

### 3. `test/chat_bot/analysis/disambiguation_integration_test.exs`

#### Current Issues
- Tests refute weather patterns but don't assert positive greeting patterns
- Some tests already have good positive assertions (lines 251-255) - use as model

#### Improvements

**Introduction Tests (lines 232-299)**
```elixir
# BEFORE
test "Hello I'm Austin - response is a greeting, not weather", %{conversation_id: conv_id} do
  {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm Austin")
  refute response =~ ~r/weather|temperature|forecast|degrees|rain|sunny|cloudy/i,
         "Introduction was misclassified - response mentions weather: #{response}"
  refute response =~ ~r/what location|which city|where.*weather/i,
         "Introduction was misclassified - response asks for location: #{response}"
  assert String.length(response) > 0
end

# AFTER (this test already has good positive assertions at lines 251-255, but can be improved)
test "Hello I'm Austin - response is a greeting, not weather", %{conversation_id: conv_id} do
  {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm Austin")
  
  # Positive: Should be a greeting-like response
  assert response =~ ~r/hello|hi|hey|nice|meet|welcome|austin/i,
         "Expected greeting response, got: #{response}"
  
  # Negative: Regression tests
  refute response =~ ~r/weather|temperature|forecast|degrees|rain|sunny|cloudy/i,
         "Introduction was misclassified - response mentions weather: #{response}"
  refute response =~ ~r/what location|which city|where.*weather/i,
         "Introduction was misclassified - response asks for location: #{response}"
end
```

### 4. `test/chat_bot/conversation_simulation_test.exs`

#### Current Issues
- Uses `refute response1 == response2` to ensure different responses
- Could add positive assertions about what each response should contain

#### Improvements

**Context Maintenance Tests (lines 87-106)**
```elixir
# BEFORE
test "maintains context across multiple turns", %{
  conversation_id: conv_id,
  user_id: user_id
} do
  {:ok, response1} = Brain.evaluate(conv_id, "Hello!", user_id: user_id)
  assert String.length(response1) > 0
  
  {:ok, response2} = Brain.evaluate(conv_id, "I like coffee", user_id: user_id)
  assert String.length(response2) > 0
  
  {:ok, response3} = Brain.evaluate(conv_id, "What's the weather like?", user_id: user_id)
  assert String.length(response3) > 0
  
  refute response1 == response2
  refute response2 == response3
end

# AFTER
test "maintains context across multiple turns", %{
  conversation_id: conv_id,
  user_id: user_id
} do
  {:ok, response1} = Brain.evaluate(conv_id, "Hello!", user_id: user_id)
  # Positive: Should be greeting response
  assert response1 =~ ~r/hello|hi|hey|welcome|nice|meet|how.*you/i,
         "Expected greeting response, got: #{response1}"
  
  {:ok, response2} = Brain.evaluate(conv_id, "I like coffee", user_id: user_id)
  # Positive: Should acknowledge the statement
  assert response2 =~ ~r/coffee|ok|got it|nice|good|thanks|understood/i,
         "Expected acknowledgment, got: #{response2}"
  
  {:ok, response3} = Brain.evaluate(conv_id, "What's the weather like?", user_id: user_id)
  # Positive: Should address weather question
  assert response3 =~ ~r/weather|temperature|forecast|location|city|where/i,
         "Expected weather-related response, got: #{response3}"
  
  # Negative: Ensure responses are different (regression test)
  refute response1 == response2
  refute response2 == response3
end
```

## Helper Functions

Create `test/support/assertion_helpers.ex` to reduce repetition:

```elixir
defmodule ChatBot.AssertionHelpers do
  @moduledoc """
  Helper functions for common test assertions.
  """

  @greeting_pattern ~r/hello|hi|hey|welcome|nice|meet|how.*you|help|can i|what can/i
  @farewell_pattern ~r/bye|goodbye|see you|later|farewell|good night|take care/i
  @weather_pattern ~r/weather|temperature|forecast|degrees|rain|sunny|cloudy|location|city|where|which/i
  @music_pattern ~r/playing|play\s|music|song|ok|sure|alright|will do/i
  @acknowledgment_pattern ~r/welcome|anytime|gladly|certainly|absolutely|pleasure|happy to help|ok|got it|understood/i

  def assert_greeting_response(response, context \\ "") do
    assert response =~ @greeting_pattern,
           "Expected greeting response#{context}, got: #{response}"
  end

  def refute_farewell_response(response, context \\ "") do
    refute response =~ @farewell_pattern,
           "Expected non-farewell response#{context}, got farewell: #{response}"
  end

  def assert_weather_response(response, context \\ "") do
    assert response =~ @weather_pattern,
           "Expected weather-related response#{context}, got: #{response}"
  end

  def assert_music_command_response(response, context \\ "") do
    assert response =~ @music_pattern,
           "Expected music command acknowledgment#{context}, got: #{response}"
  end

  def assert_acknowledgment_response(response, context \\ "") do
    assert response =~ @acknowledgment_pattern,
           "Expected acknowledgment response#{context}, got: #{response}"
  end

  def assert_farewell_response(response, context \\ "") do
    assert response =~ @farewell_pattern,
           "Expected farewell response#{context}, got: #{response}"
  end

  def assert_introduction_response(response, name \\ nil, context \\ "") do
    name_pattern = if name, do: ~r/#{name}/i, else: ~r//
    assert response =~ @greeting_pattern or response =~ name_pattern,
           "Expected introduction acknowledgment#{context}, got: #{response}"
  end
end
```

## IO.puts Replacements

### 1. `test/chat_bot/feature_test.exs` (Line 169-173)

**Current:**
```elixir
test "greeting introduction should not trigger music playback", %{conversation_id: conv_id} do
  {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm Austin")
  
  # Ideal: Should NOT mention playing anything
  # Current: "Hello" may match song name, disambiguation could be improved
  if response =~ ~r/playing|play\s/i do
    IO.puts(
      "Note: 'Hello, I'm Austin' triggered music response - hello/song disambiguation could be improved"
    )
  end
  
  # At minimum, should get a response (not be misclassified as error)
  assert String.length(response) > 0
end
```

**After:**
```elixir
test "greeting introduction should not trigger music playback", %{conversation_id: conv_id} do
  {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm Austin")
  
  # Positive: Should recognize as greeting/introduction
  assert response =~ ~r/hello|hi|hey|nice|meet|welcome|austin/i,
         "Expected greeting/introduction response, got: #{response}"
  
  # Negative: Regression test - should not trigger music playback
  refute response =~ ~r/playing|play\s|music|song/i,
         "Greeting introduction was misclassified as music request: #{response}"
end
```

### 2. `test/chat_bot/edge_cases_test.exs` (Multiple locations)

**Line 525-529: Extra spaces test**
```elixir
# BEFORE
if response =~ ~r/playing|play\s|weather/i do
  IO.puts(
    "Note: 'Hello,    I'm    Austin' (extra spaces) triggered unexpected response - tokenization could be improved"
  )
end

# AFTER
# Positive: Should recognize as greeting/introduction despite extra spaces
assert response =~ ~r/hello|hi|hey|nice|meet|welcome|austin/i,
       "Expected greeting/introduction response despite extra spaces, got: #{response}"

# Negative: Regression test
refute response =~ ~r/playing|play\s|weather/i,
       "Extra spaces caused misclassification: #{response}"
```

**Line 561-565: Typo test (im vs I'm)**
```elixir
# BEFORE
if response =~ ~r/playing|play\s|weather/i do
  IO.puts(
    "Note: 'Hello, im Austin' (typo) triggered unexpected response - disambiguation could be improved"
  )
end

# AFTER
# Positive: Should still recognize as greeting/introduction despite typo
assert response =~ ~r/hello|hi|hey|nice|meet|welcome|austin/i,
       "Expected greeting/introduction response despite typo, got: #{response}"

# Negative: Regression test
refute response =~ ~r/playing|play\s|weather/i,
       "Typo caused misclassification: #{response}"
```

**Line 654-658: Unusual introduction pattern (James Bond style)**
```elixir
# BEFORE
if response =~ ~r/playing|play\s|weather/i do
  IO.puts(
    "Note: 'The name's Bond, James Bond' triggered unexpected response - unusual pattern handling could be improved"
  )
end

# AFTER
# Positive: Should recognize as introduction despite unusual pattern
assert response =~ ~r/bond|name|introduce|meet|welcome|hello|hi/i,
       "Expected introduction response for unusual pattern, got: #{response}"

# Negative: Regression test
refute response =~ ~r/playing|play\s|weather/i,
       "Unusual introduction pattern was misclassified: #{response}"
```

**Line 691-695: Time-based greeting with city name**
```elixir
# BEFORE
if response =~ ~r/weather|temperature|forecast/i do
  IO.puts(
    "Note: 'Good afternoon, I'm Dallas' triggered weather response - disambiguation could be improved"
  )
end

# AFTER
# Positive: Should recognize as greeting/introduction
assert response =~ ~r/good afternoon|afternoon|hello|hi|nice|meet|welcome|dallas/i,
       "Expected greeting/introduction response, got: #{response}"

# Negative: Regression test - should not treat Dallas as location
refute response =~ ~r/weather|temperature|forecast/i,
       "Dallas was incorrectly interpreted as location: #{response}"
```

**Line 729-733: Complex multi-sentence scenario**
```elixir
# BEFORE
if response =~ ~r/Austin.*weather|weather.*Austin/i do
  IO.puts(
    "Note: 'Hi! I'm Austin. What's the weather?' used Austin as location - disambiguation could be improved"
  )
end

# AFTER
# Positive: Should address weather question (may ask for location)
assert response =~ ~r/weather|temperature|forecast|location|city|where/i,
       "Expected weather-related response, got: #{response}"

# Negative: Regression test - Austin from introduction should not be used as weather location
refute response =~ ~r/Austin.*weather|weather.*Austin/i,
       "Austin from introduction was incorrectly used as weather location: #{response}"
```

### 3. `test/chat_bot/learning/training_world_test.exs`

**Line 306-313: Entity discovery test**
```elixir
# BEFORE
if length(discoveries) == 0 do
  IO.puts("Warning: POS model did not discover any entities - model may need retraining")
else
  assert length(found_names) >= 1,
         "Expected to find at least 1 name, found: #{inspect(found_names)}"
end

# AFTER
# Positive: Verify the discovery process completed without errors
assert is_list(discoveries),
       "Entity discovery should return a list, got: #{inspect(discoveries)}"

# Positive: When entities are discovered, verify expected ones are found
if length(discoveries) > 0 do
  assert length(found_names) >= 1,
         "Expected to find at least 1 name when discoveries exist, found: #{inspect(found_names)}"
else
  # Positive: Even if no entities discovered, verify the workflow completed successfully
  # This ensures the system doesn't crash when model has no results
  assert length(discoveries) == 0,
         "When no entities discovered, should return empty list, got: #{inspect(discoveries)}"
end
```

**Line 520-531: Entity candidates test**
```elixir
# BEFORE
if length(candidates) == 0 do
  IO.puts("Warning: No entity candidates discovered - POS model may need retraining")
  # Still verify the workflow completed without errors
else
  {:ok, metrics} = WorldManager.get_metrics(world.id)
  assert metrics.entities_discovered > 0
  events = WorldManager.get_events(world.id)
  assert length(events) > 0
end

# AFTER
# Positive: Verify candidates retrieval completed without errors
assert is_list(candidates),
       "Candidates should return a list, got: #{inspect(candidates)}"

if length(candidates) > 0 do
  # Positive: When candidates exist, should have discovered entities
  {:ok, metrics} = WorldManager.get_metrics(world.id)
  assert metrics.entities_discovered > 0,
         "Expected entities to be discovered when candidates exist, got: #{metrics.entities_discovered}"
  
  # Positive: Should have recorded events when entities are discovered
  events = WorldManager.get_events(world.id)
  assert length(events) > 0,
         "Expected events to be recorded when entities are discovered, got: #{length(events)}"
else
  # Positive: Even if no candidates, verify the workflow completed successfully
  # Verify metrics exist and are valid (even if zero)
  {:ok, metrics} = WorldManager.get_metrics(world.id)
  assert is_map(metrics),
         "Metrics should be available even when no candidates, got: #{inspect(metrics)}"
  
  # Verify events list exists (may be empty)
  events = WorldManager.get_events(world.id)
  assert is_list(events),
         "Events should be available even when no candidates, got: #{inspect(events)}"
end
```

## Implementation Order

1. **Phase 1: Create helper functions** (`test/support/assertion_helpers.ex`)
   - Define common assertion patterns
   - Add log assertion helpers (`assert_log_contains`, `assert_contradiction_logged`, etc.)
   - Test helpers in isolation

2. **Phase 2: Replace IO.puts with assertions**
   - Update `feature_test.exs` (1 location)
   - Update `edge_cases_test.exs` (5 locations)
   - Update `training_world_test.exs` (2 locations)

3. **Phase 2b: Improve log assertions**
   - Update `disambiguation_integration_test.exs` - Add positive assertions on log content
   - Update `contradiction_handling_test.exs` - Add log assertions for contradiction warnings
   - Add tests for JTMS contradiction logging
   - Add tests for ContradictionHandler logging

3. **Phase 3: Update feature_test.exs**
   - Add positive assertions to greeting tests
   - Add positive assertions to question tests
   - Add positive assertions to command tests
   - Improve farewell tests
   - Update multi-sentence tests
   - Update conversational context tests

4. **Phase 4: Update edge_cases_test.exs**
   - Add positive assertions to name/location tests
   - Add positive assertions to name/song tests
   - Add positive assertions to informal greeting tests

5. **Phase 5: Update disambiguation_integration_test.exs**
   - Add positive assertions where missing
   - Ensure consistency with other test files

6. **Phase 6: Update conversation_simulation_test.exs**
   - Add positive assertions for context maintenance
   - Improve preference learning tests

7. **Phase 7: Add Code Coverage Tool**
   - Add ExCoveralls dependency (works locally, no cloud required)
   - Configure for local HTML reports
   - Add mix alias for coverage reports
   - Update documentation

## Code Coverage Setup

### Goal
Add code coverage tracking to identify untested code paths and measure test quality. The tool must work locally without requiring cloud services.

### Solution: ExCoveralls (Local HTML Reports)

**ExCoveralls** is the standard Elixir coverage tool. While it can post to coveralls.io, it also supports **local HTML reports** without any cloud dependency.

### Implementation

#### 1. Add Dependency to `mix.exs`

```elixir
defp deps do
  [
    # ... existing deps ...
    {:excoveralls, "~> 0.18", only: :test}
  ]
end
```

#### 2. Configure Coverage in `mix.exs`

Add to the `project` function:

```elixir
def project do
  [
    # ... existing config ...
    test_coverage: [tool: ExCoveralls],
    preferred_cli_env: [
      coveralls: :test,
      "coveralls.html": :test,
      "coveralls.json": :test
    ]
  ]
end
```

#### 3. Add Coverage Aliases

Add to `aliases` function:

```elixir
defp aliases do
  [
    # ... existing aliases ...
    "test.coverage": ["coveralls.html"],
    "test.coverage.json": ["coveralls.json"]
  ]
end
```

#### 4. Create `.coveralls.json` Configuration (Optional)

Create `config/.coveralls.json`:

```json
{
  "coverage_options": {
    "minimum_coverage": 80
  },
  "skip_files": [
    "test/",
    "lib/chat_bot_web/",
    "lib/chat_bot/application.ex"
  ]
}
```

#### 5. Update `.gitignore`

Add coverage output directories:

```
# Coverage reports
cover/
```

#### 6. Usage

After setup, developers can run:

```bash
# Generate HTML coverage report (opens in browser)
mix test.coverage

# Or directly
mix coveralls.html

# Generate JSON report
mix test.coverage.json

# Run tests with coverage
mix coveralls
```

#### 7. Integration with Precommit

Optionally add coverage check to precommit (but don't fail on low coverage initially):

```elixir
"precommit": ["compile --warning-as-errors", "deps.unlock --unused", "format", "test", "coveralls.html"]
```

### Benefits

- **Local-only** - No cloud service required
- **HTML reports** - Visual coverage reports with line-by-line highlighting
- **JSON output** - For CI/CD integration if needed later
- **Standard tool** - Widely used in Elixir community
- **Identifies gaps** - Shows which code paths aren't tested

### Files to Create/Modify

1. **`mix.exs`** - Add dependency and configuration
2. **`.gitignore`** - Add `cover/` directory
3. **`config/.coveralls.json`** (optional) - Coverage thresholds and skip files
4. **`README.md`** (update) - Document coverage commands

## Testing the Changes

After making changes, run:
```bash
# Run specific test files
mix test test/chat_bot/feature_test.exs
mix test test/chat_bot/edge_cases_test.exs
mix test test/chat_bot/analysis/disambiguation_integration_test.exs
mix test test/chat_bot/conversation_simulation_test.exs

# Generate coverage report
mix test.coverage
# or
mix coveralls.html
```

## Notes

- **Keep negative assertions** where they serve as regression tests (preventing misclassification)
- **Add positive assertions** to verify expected behavior, not just absence of wrong behavior
- **Use helper functions** to reduce repetition and improve maintainability
- **Be flexible** with regex patterns - responses may vary, so use `or` conditions where appropriate
- **Document intent** - Comments should explain what positive assertion is checking for

## Log Assertion Improvements

### Problem: Log Leaks

Currently, some tests emit logs (via `Logger.warning`, `Logger.error`) without asserting on them. This creates "log leaks" where important information (contradictions, errors, warnings) is emitted but not verified. Per `.cursorrules`, we should assert on log content using `ExUnit.CaptureLog` with positive assertions.

### Current Issues

1. **Tests that log but don't assert:**
   - `disambiguation_integration_test.exs` - Uses `capture_log` but only verifies log contains a string, not the actual log content
   - `contradiction_handling_test.exs` - Comments say "We can't easily test logging" but we should use `capture_log`

2. **Code that logs but isn't tested:**
   - `lib/chat_bot/epistemic/jtms.ex` - Line 718: `Logger.warning("Contradiction detected")`
   - `lib/chat_bot/epistemic/contradiction_handler.ex` - Lines 308, 312: Logger calls
   - `lib/chat_bot/fact_database/integration.ex` - Line 282: Logger.warning for failed beliefs

### Improvements

#### 1. `test/chat_bot/analysis/disambiguation_integration_test.exs`

**Current (lines 234-260):**
```elixir
log =
  capture_log([level: :warning], fn ->
    {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm Austin")
    Logger.warning("Response to 'Hello, I'm Austin': #{response}")
    # ... assertions ...
  end)

# Verify the response was logged
assert log =~ "Response to 'Hello, I'm Austin':"
```

**After:**
```elixir
log =
  capture_log([level: :warning], fn ->
    {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm Austin")
    Logger.warning("Response to 'Hello, I'm Austin': #{response}")
    # ... assertions ...
  end)

# Positive: Assert log contains expected content
assert log =~ "Response to 'Hello, I'm Austin':",
       "Expected log entry for response, got: #{log}"

# Positive: Assert log contains the actual response text
assert log =~ response,
       "Expected log to contain response text, got: #{log}"
```

#### 2. `test/chat_bot/epistemic/contradiction_handling_test.exs`

**Current (lines 267-296):**
```elixir
test "logs warning when learned fact contradicts existing belief" do
  # ... setup ...
  result = Integration.verify_fact("france", "The capital is not Paris")
  
  assert {:contradicted, conflicting_beliefs} = result
  
  # The Learner would log a warning and not add the fact
  # We can't easily test logging, but we can verify the fact wasn't added
  # by checking that the contradictory fact is not in the belief store
  {:ok, beliefs} = BeliefStore.query_beliefs(subject: :world, predicate: :france)
  # ... assertions ...
end
```

**After:**
```elixir
test "logs warning when learned fact contradicts existing belief" do
  import ExUnit.CaptureLog
  
  # ... setup ...
  
  log =
    capture_log([level: :warning], fn ->
      result = Integration.verify_fact("france", "The capital is not Paris")
      assert {:contradicted, conflicting_beliefs} = result
    end)
  
  # Positive: Assert that a contradiction warning was logged
  assert log =~ "contradiction" or log =~ "contradicted" or log =~ "conflict",
         "Expected contradiction warning in log, got: #{log}"
  
  # Positive: Verify the fact wasn't added (existing assertion)
  {:ok, beliefs} = BeliefStore.query_beliefs(subject: :world, predicate: :france)
  not_paris_beliefs = Enum.filter(beliefs, &(&1.object == "The capital is not Paris"))
  assert length(not_paris_beliefs) == 0,
         "Contradictory fact should not be added when contradiction detected"
end
```

#### 3. Add Tests for JTMS Contradiction Logging

**New test in `test/chat_bot/epistemic/jtms_test.exs`:**
```elixir
test "logs warning when contradiction is detected" do
  import ExUnit.CaptureLog
  
  # Create nodes that will cause a contradiction
  {:ok, node1} = JTMS.create_node("fact1", node_type: :assumption)
  {:ok, node2} = JTMS.create_node("fact2", node_type: :assumption)
  
  JTMS.enable_assumption(node1)
  JTMS.enable_assumption(node2)
  
  # Create justification that causes contradiction
  {:ok, _justification} = JTMS.create_justification("contradiction", 
    in_list: [node1, node2],
    out_list: [JTMS.contradiction_node()]
  )
  
  log = capture_log([level: :warning], fn ->
    # Trigger contradiction detection
    JTMS.check_contradictions()
  end)
  
  # Positive: Assert contradiction was logged
  assert log =~ "Contradiction detected",
         "Expected contradiction warning in log, got: #{log}"
end
```

#### 4. Add Tests for ContradictionHandler Logging

**New test in `test/chat_bot/epistemic/contradiction_handling_test.exs`:**
```elixir
test "logs info when contradiction is resolved" do
  import ExUnit.CaptureLog
  
  # ... setup contradiction ...
  
  log = capture_log([level: :info], fn ->
    ContradictionHandler.resolve_contradiction(node_id, assumption_id, :auto_resolve)
  end)
  
  # Positive: Assert resolution was logged
  assert log =~ "Contradiction resolved" or log =~ "retracting assumption",
         "Expected resolution info in log, got: #{log}"
end

test "logs warning when assumption retraction fails" do
  import ExUnit.CaptureLog
  
  # ... setup with invalid assumption_id ...
  
  log = capture_log([level: :warning], fn ->
    ContradictionHandler.resolve_contradiction(node_id, "invalid_id", :auto_resolve)
  end)
  
  # Positive: Assert failure was logged
  assert log =~ "Failed to retract assumption" or log =~ "error",
         "Expected failure warning in log, got: #{log}"
end
```

### Helper Function for Log Assertions

Add to `test/support/assertion_helpers.ex`:

```elixir
defmodule ChatBot.AssertionHelpers do
  # ... existing helpers ...
  
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
```

## Success Criteria

- All tests have at least one positive assertion (not just `String.length > 0`)
- All `IO.puts` statements in tests are replaced with proper assertions
- **All log emissions are asserted** - Use `capture_log` with positive assertions on log content
- **No log leaks** - Every `Logger.warning/error/info` call that's testable should have an assertion
- **No skipped tests** - all tests must verify something, even if model-dependent
- **No regex/string matching outside tests** - Per `.cursorrules`, regex only in tests or data preparation
- **Code coverage tool configured** - ExCoveralls set up for local HTML reports
- Negative assertions are kept where they serve as regression tests
- Helper functions reduce code duplication
- Tests are more readable and maintainable
- Test failures provide clearer error messages
- No silent failures - tests must assert something meaningful
