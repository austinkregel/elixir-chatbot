# Implementing a module

A decision guide for adding code to this umbrella, aimed at people whose
instincts come from class-based OOP. See [Architecture](ARCHITECTURE.md) for the
diagrams, and `.cursorrules` in the repo root for the project's strict rules
(which this guide defers to wherever they overlap).

---

## Step 0: check whether it already exists

437 modules, heavily documented, and duplicate systems are a recurring failure
mode in this repo. Before writing anything:

```
codemap search {query: "<the concept>"}      # by concept, across names + docs
codemap module {name: "Brain.Soul"}          # one module's public interface
codemap callers {target: "Brain.Soul.system_prompt"}   # blast radius
```

`mix docs && open doc/index.html` gives the same corpus as a browsable site
grouped by subsystem, which is often faster for "what's in
`Brain.Response`?"-shaped questions.

---

## Step 1: is it a process, or just functions?

This is the fork that OOP intuition gets wrong, because in OOP *everything* is
a class. Here, ~370 of 437 modules are namespaces of pure functions and only 64
are processes.

Write a **pure module** — the default — unless you can point at one of these:

| You need | Then it's a process |
|---|---|
| State that outlives a single call | GenServer |
| A loaded model, ETS table, or parsed corpus held in memory | GenServer |
| Serialized access to a resource (one writer at a time) | GenServer |
| Something to supervise, restart, or monitor | GenServer under the tree |
| A long-running connection or subscription | GenServer |

If none of those apply, it's a pure module. "It feels like a noun" is not a
reason. `Brain.Analysis.ChunkPriority` selects the primary chunk from a list —
that's a *verb* wearing a noun's name, and it is correctly a pure module.

> **Default hard, toward pure.** A GenServer is a process you must supervise,
> boot in the right order, and defend against being called before it's ready
> (rule 3). Every one you add is permanent cost. Pure functions cost nothing.

---

## Step 2a: writing a pure module

The shape, following [`Brain.Analysis.ChunkPriority`](../apps/brain/lib/brain/analysis/chunk_priority.ex):

```elixir
defmodule Brain.Analysis.MyThing do
  @moduledoc """
  One sentence on what this does.

  Then the part that actually earns its keep: *why* it works this way, what
  upstream signals it consumes, and what the ordering/precedence rules are.
  Codemap and ExDoc both index this text, so it is how the next person finds
  this module at all.
  """

  alias Brain.Analysis.ChunkAnalysis

  @type analysis :: ChunkAnalysis.t() | map()

  @doc """
  Returns X for `input`.

  Document edge cases here — empty list, missing key, tie-breaking.
  """
  @spec my_function([analysis()]) :: analysis()
  def my_function(analyses) when is_list(analyses) do
    # ...
  end

  # -- Private ----------------------------------------------------------------

  defp helper(_), do: :ok
end
```

Rules that are actually enforced:

- **Every public module gets a `@moduledoc`.** Brain is at 100% coverage; don't
  be the regression. `@moduledoc false` is correct for `Application` callbacks
  and similar plumbing.
- **Every public function gets `@doc` and `@spec`.**
- **Guard your public functions** (`when is_list(...)`, `when is_binary(...)`).
  The codebase does this consistently and it turns bad callers into clear
  `FunctionClauseError`s at the boundary.
- **No regex or `String.contains?` for NLP** (rule 1). Use
  `Brain.ML.Tokenizer`. For classification use `Brain.ML.MicroClassifiers` with
  training data in `data/classifiers/*.json`, not string matching (rules 2, 8).
- **No silent fallbacks** (rules 4–6). If a model or corpus is missing, raise
  with a message that says what to run to fix it. Look at
  `Brain.Test.ModelFactory` for the house style:

  ```
  ** (RuntimeError) ModelFactory: no speech act training data. Expected examples in
  .../gold_standard.json with "text" and "speech_act".
  Run `mix generate_gold_standard --speech-act` (or the full pipeline) to create it.
  ```

### Watch the module attributes

A module attribute that evaluates a remote call creates a **compile-time
dependency**, which is the one dependency kind that hurts:

```elixir
@domains Brain.Lexicon.domain_atoms()   # compile dep on Brain.Lexicon
```

Brain currently has exactly one compile edge and zero compile-connected cycles.
If you need a constant from elsewhere at compile time, either compute it inside
the function body (runtime dep, free) or extract it into a dependency-free leaf
module — see [`Brain.Lexicon.Supersenses`](../apps/brain/lib/brain/lexicon/supersenses.ex)
for the pattern. Then verify:

```bash
cd apps/brain && mix xref graph --format stats --label compile-connected  # want Cycles: 0
```

---

## Step 2b: writing a GenServer

The shape, following [`Brain.Analysis.HeuristicStore`](../apps/brain/lib/brain/analysis/heuristic_store.ex).
Note the three-part layout — this ordering is consistent across the codebase
and worth matching:

```elixir
defmodule Brain.Analysis.MyStore do
  @moduledoc """
  What state this owns, and its lifecycle.

  Document where data is persisted and what is in ETS vs. GenServer state —
  that split is the first thing anyone debugging this will need to know.
  """

  use GenServer
  require Logger

  alias Brain.ML.Tokenizer

  @table :my_store

  # -- Client API -------------------------------------------------------------
  # Public functions that other modules call. These run in the CALLER's
  # process and do nothing but package a message.

  @doc "Starts the store. Registered under `__MODULE__`."
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Returns true once init has finished loading. See rule 3."
  @spec ready?(GenServer.server()) :: boolean()
  def ready?(name \\ __MODULE__) do
    GenServer.call(name, :ready?, 100)
  catch
    :exit, _ -> false
  end

  @doc "Reads an entry. Goes straight to ETS — no message to the server."
  @spec lookup(String.t()) :: {:ok, term()} | :error
  def lookup(key) when is_binary(key) do
    case :ets.lookup(@table, key) do
      [{^key, value}] -> {:ok, value}
      [] -> :error
    end
  end

  @doc "Records an outcome. Fire-and-forget."
  @spec record(String.t()) :: :ok
  def record(key) when is_binary(key), do: GenServer.cast(__MODULE__, {:record, key})

  # -- Server callbacks -------------------------------------------------------
  # These run in the SERVER's process, one message at a time.

  @impl true
  def init(opts) do
    :ets.new(@table, [:named_table, :public, :set, read_concurrency: true])
    {:ok, %{ready?: true, opts: opts}}
  end

  @impl true
  def handle_call(:ready?, _from, state), do: {:reply, state.ready?, state}

  @impl true
  def handle_cast({:record, key}, state) do
    # ...
    {:noreply, state}
  end

  # -- Private ----------------------------------------------------------------
end
```

The parts that matter:

- **Client API vs. server callbacks is the whole idea.** `lookup/1` runs in
  *your* process. `handle_cast/2` runs in the *store's* process. The GenServer
  handles one message at a time, so anything you put in a callback is a
  bottleneck for every caller.
- **Read through ETS, write through the server.** `HeuristicStore` puts its
  data in `:public, read_concurrency: true` ETS tables so reads bypass the
  server entirely and only mutations serialize. Do this for anything read on
  the hot path.
- **`call` vs. `cast`:** `call` blocks and gets a reply; `cast` is
  fire-and-forget. Use `cast` for "record this outcome", `call` when you need
  the answer. Always pass an explicit timeout on `call`.
- **Provide `ready?/0`** and check it before calling a store that may still be
  initializing (rule 3). 100ms timeout, catch the exit.
- **`persist/0` must actually persist.** [`docs/BRAIN.md`](BRAIN.md) documents
  at least four stores whose `persist/0` docs claim a disk write but whose
  bodies are a bare `{:reply, :ok, state}`. Don't add a fifth.

### Registering it in the supervision tree

A GenServer nobody starts is dead code. Add it to
[`apps/brain/lib/brain/application.ex`](../apps/brain/lib/brain/application.ex)
**in dependency order** — the list is sequential, and your child may assume
everything above it is alive:

```elixir
children = [
  # ...
  Brain.Analysis.HeuristicStore,
  Brain.Analysis.MyStore,        # <- after what it depends on
  # ...
]
```

With `strategy: :one_for_one`, a crash restarts only your process. If your
module needs config, pass it as a tuple: `{Brain.Analysis.MyStore, [path: "..."]}`.

Confirm it booted: `/dashboard` -> Applications tab, or `Process.whereis(Brain.Analysis.MyStore)` in IEx.

---

## Step 3: which app does it go in?

Follow the dependency arrows in [Architecture](ARCHITECTURE.md#app-dependency-graph).
A module can only reference apps *below* it.

| Put it in | If it is |
|---|---|
| `atlas` | An Ecto schema, graph query, or anything touching Postgres/AGE |
| `brain` | Cognition: analysis, response, ML, memory, epistemic, souls, lattice |
| `world` | Training worlds, rosters, ingestion, per-world model registry |
| `fleet` | Multi-agent crew orchestration |
| `chat_web` | LiveView, channels, controllers, components |
| `fourth_wall` | Introspection and Credo tooling |

If you find yourself wanting `brain` to depend on `world`, stop — that's the
inverted edge described in the architecture doc, and it needs the
`@compile {:no_warn_undefined, ...}` + `Code.ensure_loaded?/1` treatment rather
than a new dependency.

---

## Step 4: before committing

```bash
mix format
mix precommit                                             # format check + credo --strict + test
cd apps/brain && mix xref graph --format stats --label compile-connected
```

Tests run from the **umbrella root**, not from `apps/brain` — `config/runtime.exs`
imports `Dotenvy`, which is a root-only dependency:

```bash
mix test apps/brain/test/brain/analysis/       # correct
```

The brain test suite trains real models on real gold-standard corpora at
startup (`test/test_helper.exs` -> `Brain.Test.ModelFactory`). If it raises
about missing training data, that is rule 4 working as designed — run the
generator it names rather than working around it.
