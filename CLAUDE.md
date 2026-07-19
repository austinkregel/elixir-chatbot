# chat_bot_llm — Elixir umbrella

An Elixir umbrella of ~430 modules across 7 apps:

- **brain** (~300 modules) — cognition: `Brain.Analysis`, `Brain.ML`, `Brain.Response`, `Brain.Knowledge`, `Brain.Epistemic`, souls, lattice, memory
- **atlas** — knowledge graph + Ecto schemas (Postgres/AGE)
- **world** — training worlds, rosters, ingestion
- **chat_web** — Phoenix/LiveView UI
- **fleet** — multi-agent crew orchestration
- **fourth_wall**, **tasks** — introspection and task tooling

## Navigating the codebase — use codemap FIRST

This repo ships a `codemap` MCP server (registered in `.mcp.json`, implemented
in `scripts/codemap_mcp.exs`) that indexes every module's docs, functions,
specs, and structs. **Before grepping, globbing, or writing anything new,
check whether the system already exists:**

1. `codemap overview` — map of all apps; `overview {app: "brain"}` for the
   namespace tree of one app with per-module doc summaries; add
   `namespace: "Brain.Analysis"` to drill into one subsystem of a large app.
2. `codemap search {query: "..."}` — find existing systems by concept
   (searches module names, function names, and @moduledoc/@doc text).
3. `codemap module {name: "Brain.Soul"}` — full public interface of one
   module (moduledoc, specs, struct, file:line). Prefer this over reading a
   whole file when you only need the interface.
4. `codemap callers {target: "Brain.Soul.system_prompt"}` — reference sites
   across lib and test; check blast radius before changing any interface.

Raw `Grep`/`Read` are still right for implementation details inside a file
you've already located, private functions, and non-`.ex` assets. But
discovery ("does X exist?", "where does Y live?", "who uses Z?") should go
through codemap — this corpus is large and heavily documented, and duplicate
systems are a recurring failure mode.

## Conventions

- Every public module gets a `@moduledoc`; public functions get `@doc`/`@spec`.
  Codemap's usefulness depends on this — keep it up.
- Run `mix format` before committing; `mix precommit` runs in the test env.
