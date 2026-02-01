# Writing Tests for ChatBot

This guide covers best practices for writing tests in the ChatBot project, with particular focus on managing GenServer lifecycle and avoiding common pitfalls.

## Table of Contents

- [Test Infrastructure Overview](#test-infrastructure-overview)
- [Starting Services in Tests](#starting-services-in-tests)
- [Common Patterns](#common-patterns)
- [Anti-Patterns to Avoid](#anti-patterns-to-avoid)
- [Troubleshooting](#troubleshooting)

---

## Test Infrastructure Overview

### Key Files

| File | Purpose |
|------|---------|
| `test/test_helper.exs` | Global test setup (PubSub, ETS tables) |
| `test/support/test_helpers.ex` | Helper functions for starting services |
| `test/support/test_world_sandbox.ex` | Sandbox for training world tests |

### How ExUnit Manages Processes

ExUnit provides `start_supervised/1` which:
1. Starts a process under a test-specific supervisor
2. Automatically stops the process when the test ends
3. Ensures proper cleanup even if the test fails

This is **critical** for tests that use named GenServers (like `Embedder`, `IntentClassifierSimple`, etc.).

---

## Starting Services in Tests

### Use `ensure_started/1` for All GenServers

Always use the `ensure_started/1` helper from `ChatBot.TestHelpers`:

```elixir
defmodule MyTest do
  use ExUnit.Case, async: false
  import ChatBot.TestHelpers

  setup do
    # Start services under ExUnit supervision
    ensure_started(ChatBot.Memory.Embedder)
    ensure_started(ChatBot.ML.IntentClassifierSimple)
    
    :ok
  end
end
```

### For Services with Options

Pass a tuple with the module and options:

```elixir
setup do
  ensure_started({ChatBot.Memory.Store, persistence_path: "/tmp/test_store.term"})
  ensure_started({ChatBot.Analysis.HeuristicStore, seeded_path: "data/heuristics/seeded_heuristics.json"})
  
  :ok
end
```

### Using `start_test_services/0`

For integration tests that need multiple services:

```elixir
setup do
  start_test_services()  # Starts PubSub, Gazetteer, IntentClassifierSimple, etc.
  :ok
end
```

For tests that also need the Brain:

```elixir
setup do
  start_brain_services()  # Includes start_test_services() + Brain
  :ok
end
```

---

## Common Patterns

### Pattern 1: Basic Unit Test with GenServer

```elixir
defmodule ChatBot.Memory.EmbedderTest do
  use ExUnit.Case, async: false
  import ChatBot.TestHelpers

  alias ChatBot.Memory.Embedder

  setup do
    ensure_pubsub_started()
    ensure_started(Embedder)
    :ok
  end

  test "builds vocabulary from texts" do
    texts = ["hello world", "hello there"]
    {:ok, vocab_size} = Embedder.build_vocabulary(texts)
    
    assert vocab_size > 0
    assert Embedder.ready?()
  end
end
```

### Pattern 2: Test with Data Cleanup

```elixir
defmodule ChatBot.Epistemic.BeliefStoreTest do
  use ExUnit.Case, async: false
  import ChatBot.TestHelpers

  alias ChatBot.Epistemic.BeliefStore

  setup do
    ensure_pubsub_started()
    ensure_started(BeliefStore)
    
    # Clear data before each test
    BeliefStore.clear()
    
    :ok
  end
end
```

### Pattern 3: Integration Test with Multiple Services

```elixir
defmodule ChatBot.Analysis.PipelineTest do
  use ExUnit.Case, async: false
  import ChatBot.TestHelpers

  setup do
    start_test_services()
    ChatBot.ML.EntityExtractor.load_entity_maps()
    :ok
  end
end
```

### Pattern 4: Test with Context Data

```elixir
setup do
  ensure_started(ChatBot.Memory.Embedder)
  
  texts = ["training text 1", "training text 2"]
  ChatBot.Memory.Embedder.build_vocabulary(texts)
  
  # Return context for tests
  {:ok, vocabulary_size: length(texts)}
end

test "uses vocabulary", %{vocabulary_size: size} do
  assert size == 2
end
```

---

## Anti-Patterns to Avoid

### Do NOT Use Raw `start_link`

```elixir
# BAD - Process not managed by ExUnit
setup do
  {:ok, _pid} = Embedder.start_link()
  :ok
end
```

```elixir
# GOOD - Process managed by ExUnit
setup do
  ensure_started(Embedder)
  :ok
end
```

### Do NOT Use `GenServer.stop/1`

```elixir
# BAD - Can interfere with other tests
test "some test" do
  GenServer.stop(Embedder)
  {:ok, _} = Embedder.start_link()
end
```

```elixir
# GOOD - Let ExUnit manage lifecycle
# If you need fresh state, use a clear/reset function instead
test "some test" do
  Embedder.clear()  # or reset state via API
end
```

### Do NOT Use `setup_all` for Named GenServers

```elixir
# BAD - Process lifetime extends beyond individual tests
setup_all do
  ensure_started(IntentClassifierSimple)
  :ok
end
```

```elixir
# GOOD - Each test gets its own process instance
setup do
  ensure_started(IntentClassifierSimple)
  IntentClassifierSimple.load_models()
  :ok
end
```

### Do NOT Mix Manual and Supervised Process Management

```elixir
# BAD - Conflicting lifecycle management
setup do
  case Process.whereis(Store) do
    nil -> Store.start_link()
    pid -> GenServer.stop(pid) && Store.start_link()
  end
end
```

```elixir
# GOOD - Consistent supervised management
setup do
  ensure_started(Store)
  Store.clear()
  :ok
end
```

---

## Troubleshooting

### Error: `(EXIT) no process`

**Symptom:**
```
** (exit) exited in: GenServer.call(ChatBot.ML.IntentClassifierSimple, ...)
    ** (EXIT) no process: the process is not alive
```

**Cause:** The GenServer was never started or was killed by another test.

**Fix:** Ensure the service is started in your test's `setup`:
```elixir
setup do
  ensure_started(ChatBot.ML.IntentClassifierSimple)
  :ok
end
```

### Error: `(EXIT) shutdown`

**Symptom:**
```
** (exit) exited in: GenServer.call(ChatBot.Memory.Embedder, ...)
    ** (EXIT) shutdown
```

**Cause:** Another test's teardown is killing the process while your test is using it.

**Fix:** 
1. Remove any `GenServer.stop()` calls
2. Use `ensure_started()` consistently
3. Ensure tests use `async: false` if they share state

### Error: `{:already_started, pid}`

**Symptom:**
```
** (MatchError) no match of right hand side value: {:error, {:already_started, #PID<0.123.0>}}
```

**Cause:** Trying to start a process that's already running (possibly from the application or another test).

**Fix:** Use `ensure_started()` which handles this case:
```elixir
# This handles already_started gracefully
ensure_started(Embedder)
```

### Tests Pass Individually but Fail Together

**Cause:** Tests are not properly isolated - they share global state.

**Fixes:**
1. Use `async: false` for tests that share named GenServers
2. Clear/reset state in `setup`
3. Use `ensure_started()` instead of `start_link()`
4. Avoid `setup_all` for named processes

---

## Test Configuration

### When to Use `async: true` vs `async: false`

| Scenario | Setting |
|----------|---------|
| Tests use named GenServers | `async: false` |
| Tests use shared ETS tables | `async: false` |
| Tests modify global state | `async: false` |
| Pure function tests | `async: true` |
| Tests with isolated state | `async: true` |

### Example Test Module Structure

```elixir
defmodule ChatBot.MyModuleTest do
  use ExUnit.Case, async: false  # Use false for GenServer tests
  import ChatBot.TestHelpers

  alias ChatBot.MyModule

  # Setup runs before each test
  setup do
    ensure_pubsub_started()
    ensure_started(MyModule)
    
    # Clear any existing state
    MyModule.clear()
    
    :ok
  end

  describe "feature_a" do
    test "does something" do
      # Test code here
    end
  end

  describe "feature_b" do
    # Additional setup for this describe block
    setup do
      MyModule.configure(option: :value)
      :ok
    end

    test "does something else" do
      # Test code here
    end
  end
end
```

---

## Training World Tests

For tests involving training worlds, use the sandbox:

```elixir
defmodule ChatBot.Learning.MyWorldTest do
  use ExUnit.Case, async: false
  import ChatBot.TestHelpers

  setup do
    start_world_test_services()
    setup_world_sandbox()  # Enables automatic cleanup
    :ok
  end

  test "creates a test world" do
    {:ok, world} = create_test_world("my_test")
    
    # World is automatically cleaned up after test
    assert world.metadata.test == true
  end
end
```

---

## Quick Reference

| Task | Helper Function |
|------|-----------------|
| Start a GenServer | `ensure_started(Module)` |
| Start GenServer with options | `ensure_started({Module, opts})` |
| Start common services | `start_test_services()` |
| Start Brain + services | `start_brain_services()` |
| Start world services | `start_world_test_services()` |
| Ensure PubSub running | `ensure_pubsub_started()` |
| Create test world | `create_test_world(name)` |
| Setup world cleanup | `setup_world_sandbox()` |

---

## See Also

- [CONTRIBUTING.md](CONTRIBUTING.md) - Main contributor guide
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture overview
- [PIPELINE_ORDER.md](PIPELINE_ORDER.md) - Pipeline execution order
