defmodule Fleet.Tool do
  @moduledoc """
  A registered capability the *harness* can execute on an agent's behalf — and the
  code-owned registry of them.

  A tool is **not** something the model can invoke. It is an authority-gated action
  the harness dispatches only when an approved `Fleet.Proposal` names it. Three
  invariants make this structural, not behavioural:

    * **The registry is code-owned.** A human registers tools here; the model can
      never add one, and a name not in the registry is **unrunnable** (default-deny).
    * **Effect tier lives in the data.** Each tool declares `:read | :mutate |
      :irreversible`, so the gate reasons about blast radius from the spec, not from
      anything the model says.
    * **Firing requires an authority.** `required_authority` must be in the caller's
      order-conferred grant (`Fleet.Authority.tool/1`) — checked by the harness
      against the agent's real grants, never against the proposal payload.

  ## Argument schemas

  A tool declares `:args_schema`, and `Fleet.Dispatcher.decide/2` validates the
  proposal's `args` against it *before* any handler runs. Without this a
  proposing model must guess argument shapes, and a bad guess reaches the handler
  and fails as an opaque `{:error, reason}` — indistinguishable in the audit from
  a tool fault. The Python original validated here and recorded a distinct
  `malformed_call` (`harness.py:306-317`); this is that step.

  The schema deliberately reuses the **shape** of
  `Brain.Services.Service.slot_schema/0` so one vocabulary spans both selection
  paths:

      %{
        "required" => ["symbol"],
        "optional" => ["limit"],
        "types"    => %{"symbol" => :string, "limit" => :integer},
        "describe" => %{"symbol" => "Fully qualified or bare symbol name"}
      }

  It omits that schema's `entity_mappings` and `clarification_templates`: those
  bind NER output to slots on the analytical path, and a proposing model emits
  JSON directly, so carrying them here would be dead weight.

  `nil` means "takes no arguments" and is enforced as such — an unrecognised key
  is refused rather than ignored, because silently dropping an argument the model
  believed it was passing produces a confidently wrong answer.
  """

  alias Brain.Code.QueryHandler
  alias Brain.Epistemic.{BeliefStore, JTMS}
  alias Brain.FactDatabase
  alias Brain.Knowledge.Academic.{Arxiv, OpenAlex, SemanticScholar}
  alias Brain.Memory.Store, as: MemoryStore
  alias Brain.Services.HomeAssistant
  alias World.CodeContext

  # Narrowing descriptors the gate itself reads off every proposal
  # (`Fleet.Dispatcher.gate_read/3`). They only ever *reduce* what a read
  # returns, so they are accepted on any tool without being declared.
  @universal_args ~w(ship_id agent_id)

  @enforce_keys [:name, :effect, :required_authority, :handler]
  defstruct [
    :name,
    :effect,
    :required_authority,
    :handler,
    :description,
    :info_class,
    :args_schema,
    egress: []
  ]

  @type effect :: :read | :mutate | :irreversible
  @type ctx :: %{optional(any) => any}
  @type args_schema :: %{optional(String.t()) => term()}
  @type arg_error ::
          {:missing, String.t()} | {:unknown, String.t()} | {:type, String.t(), atom()}
  @type t :: %__MODULE__{
          name: String.t(),
          effect: effect(),
          required_authority: term(),
          handler: (map(), ctx() -> {:ok, term()} | {:error, term()}),
          description: String.t() | nil,
          info_class: atom() | nil,
          args_schema: args_schema() | nil,
          egress: [String.t()]
        }

  @doc """
  The code-owned registry (name => spec). Human-registered; never model-registered.

  A `:read`-effect tool that declares an `:info_class` is gated by `Fleet.Clearance`
  in the Dispatcher (in addition to the action-grant): `beliefs.read` reads the
  caller's own mind (`:agent_mind`, self-read); `systems.read` reads ship health
  (`:system_status`, ship-commissioned).
  """
  @spec registry() :: %{optional(String.t()) => t()}
  def registry do
    %{
      "beliefs.read" => %__MODULE__{
        name: "beliefs.read",
        effect: :read,
        info_class: :agent_mind,
        required_authority: Fleet.Authority.tool("beliefs.read"),
        description: "Read the calling agent's own beliefs (its mind-world belief store). Read-only.",
        handler: &beliefs_read/2
      },
      "systems.read" => %__MODULE__{
        name: "systems.read",
        effect: :read,
        info_class: :system_status,
        required_authority: Fleet.Authority.tool("systems.read"),
        description:
          "Read the ship's systems health — the layered inventory of services, " <>
            "processes, and subsystems. Read-only; ground truth independent of any LLM.",
        handler: &systems_read/2
      },
      # Clearance-gated to :trust_ledger (XO and above), which is what keeps a
      # reviewer from reading an officer's reputation before judging its work.
      "trust.read" => %__MODULE__{
        name: "trust.read",
        effect: :read,
        info_class: :trust_ledger,
        required_authority: Fleet.Authority.tool("trust.read"),
        description:
          "Read an officer's derived trust ledger — accuracy, dissent record, and " <>
            "anomaly handling, computed from the append-only audit and service history. " <>
            "Read-only; never self-reported.",
        args_schema: %{
          "required" => [],
          "optional" => ["soul_id"],
          "types" => %{"soul_id" => :string},
          "describe" => %{
            "soul_id" => "Whose ledger to read. Defaults to the calling officer's own."
          }
        },
        handler: &trust_read/2
      },

      # ── Working capability ────────────────────────────────────────────────
      # Every one of these wraps a subsystem that already worked and had no
      # route to a commissioned officer. They are all `:read`: an officer can
      # study, cite, and report, and changes nothing.

      "beliefs.why" => %__MODULE__{
        name: "beliefs.why",
        effect: :read,
        info_class: :agent_mind,
        required_authority: Fleet.Authority.tool("beliefs.why"),
        description:
          "Trace WHY a belief is held — the JTMS justification chain behind it, back to " <>
            "its premises. This is the difference between an officer that asserts and one " <>
            "that can show its work.",
        args_schema: %{
          "required" => ["node_id"],
          "optional" => [],
          "types" => %{"node_id" => :string},
          "describe" => %{"node_id" => "The belief/JTMS node to justify."}
        },
        handler: &beliefs_why/2
      },
      "memory.search" => %__MODULE__{
        name: "memory.search",
        effect: :read,
        info_class: :agent_mind,
        required_authority: Fleet.Authority.tool("memory.search"),
        description: "Search the calling agent's own episodic memory by similarity to a query.",
        args_schema: %{
          "required" => ["query"],
          "optional" => ["limit"],
          "types" => %{"query" => :string, "limit" => :integer},
          "describe" => %{
            "query" => "What to recall.",
            "limit" => "How many episodes to return (default 5)."
          }
        },
        handler: &memory_search/2
      },
      "facts.query" => %__MODULE__{
        name: "facts.query",
        effect: :read,
        info_class: :world_knowledge,
        required_authority: Fleet.Authority.tool("facts.query"),
        description:
          "Query grounded facts. Spans the curated layer — immutable by construction, so " <>
            "nothing an agent does can corrupt it — and the learned layer.",
        args_schema: %{
          "required" => [],
          "optional" => ["entity", "category", "limit"],
          "types" => %{"entity" => :string, "category" => :string, "limit" => :integer},
          "describe" => %{
            "entity" => "Restrict to facts about this entity.",
            "category" => "Restrict to a fact category, e.g. science.",
            "limit" => "How many facts to return (default 20)."
          }
        },
        handler: &facts_query/2
      },
      "code.search" => %__MODULE__{
        name: "code.search",
        effect: :read,
        info_class: :corpus,
        required_authority: Fleet.Authority.tool("code.search"),
        description:
          "Search the indexed code corpus of the ORDER's world for symbols matching a query. " <>
            "Returns name, kind, file and line — the map, not the source.",
        args_schema: %{
          "required" => ["query"],
          "optional" => ["kind", "limit"],
          "types" => %{"query" => :string, "kind" => :string, "limit" => :integer},
          "describe" => %{
            "query" => "Symbol name or fragment to search for.",
            "kind" => "Restrict to one symbol type, e.g. code.function or code.class.",
            "limit" => "How many symbols to return (default 25)."
          }
        },
        handler: &code_search/2
      },
      "code.explain" => %__MODULE__{
        name: "code.explain",
        effect: :read,
        info_class: :corpus,
        required_authority: Fleet.Authority.tool("code.explain"),
        description: "Explain a symbol in the indexed corpus — what it is, where, and what it does.",
        args_schema: %{
          "required" => ["symbol"],
          "optional" => [],
          "types" => %{"symbol" => :string},
          "describe" => %{"symbol" => "Bare or qualified symbol name, e.g. Brain.Soul.get."}
        },
        handler: &code_explain/2
      },
      "code.relations" => %__MODULE__{
        name: "code.relations",
        effect: :read,
        info_class: :corpus,
        required_authority: Fleet.Authority.tool("code.relations"),
        description:
          "Who calls a symbol, what it calls, and what a module depends on — the blast " <>
            "radius of a change, read off the indexed corpus rather than guessed.",
        args_schema: %{
          "required" => ["symbol"],
          "optional" => ["direction"],
          "types" => %{"symbol" => :string, "direction" => :string},
          "describe" => %{
            "symbol" => "Qualified symbol or module name.",
            "direction" => "callers (default), callees, or dependencies."
          }
        },
        handler: &code_relations/2
      },

      # ── Capability that leaves the ship ───────────────────────────────────
      # Semantically reads, so they stay `effect: :read` rather than being
      # tiered up to :irreversible — which would put a literature search behind
      # the two-officer rule. They pay for their reach with `{:egress, host}`,
      # and the allowlist is here in the code-owned spec, so the handler cannot
      # reach a host this struct does not name.

      "papers.search" => %__MODULE__{
        name: "papers.search",
        effect: :read,
        info_class: :external_research,
        required_authority: Fleet.Authority.tool("papers.search"),
        egress: ["export.arxiv.org", "api.openalex.org", "api.semanticscholar.org"],
        description:
          "Search the academic literature (arXiv, OpenAlex, Semantic Scholar) and return " <>
            "papers with titles, abstracts and citation counts. Read-only; no publication path.",
        args_schema: %{
          "required" => ["query"],
          "optional" => ["source", "limit"],
          "types" => %{"query" => :string, "source" => :string, "limit" => :integer},
          "describe" => %{
            "query" => "The search query.",
            "source" => "arxiv (default), openalex, or semantic_scholar.",
            "limit" => "How many papers to return (default 10)."
          }
        },
        handler: &papers_search/2
      },
      "homeassistant.read_state" => %__MODULE__{
        name: "homeassistant.read_state",
        effect: :read,
        info_class: :home_sensors,
        required_authority: Fleet.Authority.tool("homeassistant.read_state"),
        egress: ["home_assistant"],
        description:
          "Read Home Assistant sensor and device state. Observation only — this reaches a " <>
            "read-only API and has no path to actuation.",
        args_schema: %{
          "required" => [],
          "optional" => ["entity_id", "domain"],
          "types" => %{"entity_id" => :string, "domain" => :string},
          "describe" => %{
            "entity_id" => "One entity, e.g. sensor.office_temperature.",
            "domain" => "Read every entity in a domain instead, e.g. sensor (default)."
          }
        },
        handler: &home_assistant_read/2
      }
    }
  end

  @doc "Look up a tool by name. `:error` (default-deny) for anything unregistered."
  @spec lookup(term()) :: {:ok, t()} | :error
  def lookup(name) when is_binary(name), do: Map.fetch(registry(), name)
  def lookup(_), do: :error

  @doc """
  Validate a proposal's `args` against a tool's `:args_schema`. **Pure** — this is
  called from `Fleet.Dispatcher.decide/2`, which must stay side-effect free.

  Returns `:ok` or `{:error, errors}` with every problem found, not just the
  first: one corrective turn should be able to fix the whole call.
  """
  @spec validate_args(t(), term()) :: :ok | {:error, [arg_error()]}
  # `%Fleet.Proposal{}` defaults `:args` to nil and `parse/1` fills in `%{}`, so
  # nil means "no arguments supplied" — which is only a problem if the tool
  # requires some. Treating it as malformed would refuse every no-arg call built
  # from the struct directly.
  def validate_args(%__MODULE__{} = tool, nil), do: validate_args(tool, %{})

  def validate_args(%__MODULE__{} = tool, args) when is_map(args) do
    schema = tool.args_schema || %{}
    required = Map.get(schema, "required", [])
    optional = Map.get(schema, "optional", [])
    types = Map.get(schema, "types", %{})
    declared = required ++ optional ++ @universal_args

    errors =
      Enum.map(required, fn key ->
        if blank?(Map.get(args, key)), do: {:missing, key}
      end) ++
        Enum.map(Map.keys(args), fn key ->
          cond do
            key not in declared -> {:unknown, key}
            true -> type_error(key, Map.get(args, key), Map.get(types, key))
          end
        end)

    case Enum.reject(errors, &is_nil/1) do
      [] -> :ok
      errs -> {:error, errs}
    end
  end

  # A non-map `args` is a malformed call in its own right; `Fleet.Proposal`
  # defaults the field to `%{}`, so this only fires on an explicit non-object.
  def validate_args(%__MODULE__{}, _args), do: {:error, [{:type, "args", :object}]}

  @doc """
  Render a tool's interface as the appendix an officer sees — name, what it does,
  and the arguments it takes. `Fleet.Officer` used to append bare names, which
  left a proposing model guessing both purpose and shape.
  """
  @spec describe(t()) :: String.t()
  def describe(%__MODULE__{} = tool) do
    schema = tool.args_schema || %{}
    describe_map = Map.get(schema, "describe", %{})
    types = Map.get(schema, "types", %{})

    args =
      Enum.map(Map.get(schema, "required", []), &{&1, :required}) ++
        Enum.map(Map.get(schema, "optional", []), &{&1, :optional})

    arg_text =
      case args do
        [] ->
          "    args: none"

        list ->
          Enum.map_join(list, "\n", fn {key, req} ->
            type = Map.get(types, key, :string)
            note = Map.get(describe_map, key)
            "    #{key} (#{type}, #{req})" <> if(note, do: " — #{note}", else: "")
          end)
      end

    "  #{tool.name} — #{tool.description}\n#{arg_text}"
  end

  defp blank?(nil), do: true
  defp blank?(""), do: true
  defp blank?(_), do: false

  defp type_error(_key, _value, nil), do: nil
  defp type_error(_key, value, :string) when is_binary(value), do: nil
  defp type_error(_key, value, :integer) when is_integer(value), do: nil
  defp type_error(_key, value, :number) when is_number(value), do: nil
  defp type_error(_key, value, :boolean) when is_boolean(value), do: nil
  defp type_error(_key, value, :object) when is_map(value), do: nil
  defp type_error(_key, value, :list) when is_list(value), do: nil
  defp type_error(key, _value, expected), do: {:type, key, expected}

  # ── Handlers: (args, ctx) -> {:ok, data} | {:error, reason} ────────────────
  # A handler NEVER decides whether it may run — that is settled by the gate before
  # it is ever called. It only produces data, which the dispatcher frames as DATA.

  defp beliefs_read(_args, %{world_id: world_id}) when is_binary(world_id) do
    cond do
      is_nil(Process.whereis(BeliefStore)) ->
        {:error, :belief_store_unavailable}

      true ->
        case BeliefStore.query_beliefs(world_id: world_id) do
          {:ok, beliefs} ->
            {:ok,
             Enum.map(beliefs, fn b ->
               b |> Map.from_struct() |> Map.take([:subject, :predicate, :object, :confidence, :world_id])
             end)}

          {:error, reason} ->
            {:error, reason}
        end
    end
  end

  defp beliefs_read(_args, _ctx), do: {:error, :no_world}

  # The justification chain — the artifact a generic tool-using model cannot
  # produce, and until now the only JTMS read with no caller outside its test.
  defp beliefs_why(%{"node_id" => node_id}, ctx) when is_binary(node_id) do
    case JTMS.why_node(mind_world(ctx), node_id) do
      nil -> {:error, :no_such_node}
      {:error, reason} -> {:error, reason}
      explanation -> {:ok, explanation}
    end
  end

  defp beliefs_why(_args, _ctx), do: {:error, :no_node_id}

  defp memory_search(%{"query" => query} = args, ctx) when is_binary(query) do
    limit = Map.get(args, "limit", 5)

    case MemoryStore.query_similar(query, limit, world_id: mind_world(ctx)) do
      {:ok, episodes} -> {:ok, episodes}
      episodes when is_list(episodes) -> {:ok, episodes}
      {:error, reason} -> {:error, reason}
    end
  end

  defp memory_search(_args, _ctx), do: {:error, :no_query}

  defp facts_query(args, _ctx) do
    opts =
      []
      |> put_opt(:entity, Map.get(args, "entity"))
      |> put_opt(:category, Map.get(args, "category"))
      |> put_opt(:limit, Map.get(args, "limit", 20))

    case FactDatabase.query(opts) do
      {:ok, facts} -> {:ok, Enum.map(facts, &summarize_fact/1)}
      facts when is_list(facts) -> {:ok, Enum.map(facts, &summarize_fact/1)}
      {:error, reason} -> {:error, reason}
    end
  end

  # Code reads target the ORDER's world, not the officer's mind: a corpus is
  # shared working material, and an officer's private mind is the wrong place
  # for it. `corpus_world/1` refuses rather than silently falling back to a
  # default world, which would read someone else's index and look like success.
  defp code_search(%{"query" => query} = args, ctx) when is_binary(query) do
    with {:ok, world_id} <- corpus_world(ctx),
         :ok <- corpus_indexed(world_id) do
      opts =
        []
        |> put_opt(:entity_type, Map.get(args, "kind"))
        |> put_opt(:limit, Map.get(args, "limit", 25))

      {:ok, world_id |> CodeContext.search_symbols(query, opts) |> Enum.map(&summarize_symbol/1)}
    end
  end

  defp code_search(_args, _ctx), do: {:error, :no_query}

  # `QueryHandler.explain/2` answers a *not found* with `{:ok, "I couldn't find
  # X — make sure the codebase has been analyzed"}`. That is right for a chat
  # turn and wrong here: framed as DATA it reads to an officer as a finding, and
  # an officer that reports "no such symbol" when the truth is "nothing is
  # indexed" has inflated a tooling gap into a result. So the two cases are
  # separated before the handler is consulted, and both come back as errors —
  # which `Fleet.ToolRound` tells the agent is a tool fault, not an answer.
  defp code_explain(%{"symbol" => symbol}, ctx) when is_binary(symbol) do
    with {:ok, world_id} <- corpus_world(ctx),
         :ok <- corpus_indexed(world_id),
         {:ok, _} <- CodeContext.get_symbol(world_id, symbol) do
      case QueryHandler.explain(symbol, world_id: world_id) do
        {:ok, text} -> {:ok, text}
        other -> {:error, other}
      end
    else
      :not_found -> {:error, {:symbol_not_found, symbol}}
      {:error, _} = error -> error
    end
  end

  defp code_explain(_args, _ctx), do: {:error, :no_symbol}

  defp code_relations(%{"symbol" => symbol} = args, ctx) when is_binary(symbol) do
    with {:ok, world_id} <- corpus_world(ctx),
         :ok <- corpus_indexed(world_id) do
      case Map.get(args, "direction", "callers") do
        "callers" -> {:ok, %{symbol: symbol, callers: CodeContext.get_callers(world_id, symbol)}}
        "callees" -> {:ok, %{symbol: symbol, callees: CodeContext.get_callees(world_id, symbol)}}
        "dependencies" -> {:ok, %{symbol: symbol, dependencies: CodeContext.get_dependencies(world_id, symbol)}}
        other -> {:error, {:unknown_direction, other}}
      end
    end
  end

  defp code_relations(_args, _ctx), do: {:error, :no_symbol}

  defp papers_search(%{"query" => query} = args, _ctx) when is_binary(query) do
    limit = Map.get(args, "limit", 10)

    result =
      case Map.get(args, "source", "arxiv") do
        "arxiv" -> Arxiv.search(query, max_results: limit)
        "openalex" -> OpenAlex.search(query, per_page: limit)
        "semantic_scholar" -> SemanticScholar.search(query, limit: limit)
        other -> {:error, {:unknown_source, other}}
      end

    case result do
      {:ok, papers} -> {:ok, Enum.map(papers, &summarize_paper/1)}
      {:error, reason} -> {:error, reason}
    end
  end

  defp papers_search(_args, _ctx), do: {:error, :no_query}

  defp home_assistant_read(args, _ctx) do
    HomeAssistant.read_state(Map.get(args, "entity_id"),
      domain: Map.get(args, "domain", "sensor")
    )
  end

  # ── Handler helpers ───────────────────────────────────────────────────────

  defp mind_world(ctx), do: ctx[:mind_world_id] || ctx[:world_id]

  # An order that named no world confers no corpus to read. Refusing is the
  # honest outcome: reading whatever happens to be in the default world would
  # answer from someone else's index and be indistinguishable from working.
  defp corpus_world(ctx) do
    case ctx[:order_world_id] do
      world_id when is_binary(world_id) and world_id != "" -> {:ok, world_id}
      _ -> {:error, :no_corpus_world}
    end
  end

  # `Brain.Code.CodeGazetteer` is in-memory ETS with no load-from-disk, so a
  # corpus lives only as long as the node that ingested it — a `mix` task cannot
  # populate a running server. Until that is persisted, "empty" is a normal
  # state and must be reported as one rather than read as "the code contains
  # nothing matching".
  defp corpus_indexed(world_id) do
    case CodeContext.stats(world_id) do
      %{symbols: n} when is_integer(n) and n > 0 -> :ok
      _ -> {:error, {:corpus_not_indexed, world_id}}
    end
  end

  defp put_opt(opts, _key, nil), do: opts
  defp put_opt(opts, key, value), do: Keyword.put(opts, key, value)

  defp summarize_fact(%{} = fact),
    do: Map.take(fact, [:id, :entity, :fact, :category, :confidence, :verification_source])

  defp summarize_fact(other), do: other

  defp summarize_symbol(%{} = symbol),
    do: Map.take(symbol, [:name, :qualified_name, :entity_type, :file_path, :line, :language])

  defp summarize_symbol(other), do: other

  # Abstracts are long enough to crowd out everything else in one framed block;
  # the officer can ask for a specific paper if a summary looks relevant.
  defp summarize_paper(%{} = paper) do
    paper
    |> Map.take([:id, :title, :abstract, :authors, :year, :citation_count, :url, :source])
    |> Map.update(:abstract, nil, fn
      text when is_binary(text) -> String.slice(text, 0, 600)
      other -> other
    end)
  end

  defp summarize_paper(other), do: other

  # systems.read — the ship's health, as ground truth. Returns a compact summary
  # (counts + rolled-up health + any non-nominal live systems) so the framed <data>
  # stays legible; the full inventory is the human board's job.
  defp systems_read(_args, _ctx) do
    snap = Fleet.Systems.snapshot()

    not_nominal =
      (snap.services ++ snap.processes)
      |> Enum.reject(&(&1.status == :up))
      |> Enum.map(&%{system: &1.name, layer: &1.layer, deck: &1.deck, status: &1.status})

    {:ok,
     %{
       ship_id: snap.ship_id,
       health: Map.take(snap.health, [:health_score, :health_status, :services_up, :services_total, :genservers_running, :genservers_total]),
       counts: snap.counts,
       not_nominal: not_nominal
     }}
  end

  # Reads the ledger for a named officer, defaulting to the caller's own. The
  # clearance gate (:trust_ledger, XO and above) has already run by the time
  # this executes — an ensign cannot reach its own file this way.
  defp trust_read(args, ctx) do
    soul_id = Map.get(args, "soul_id") || Map.get(args, "agent_id") || ctx[:agent_id]

    if is_binary(soul_id) do
      ledger = Fleet.TrustLedger.compute(soul_id)

      {:ok,
       ledger
       |> Map.take([:soul_id, :orders, :verified_accuracy, :dissent_quality, :anomaly_record,
                    :calibration, :evidence_quality])
       |> Map.put(:score, Fleet.TrustLedger.score(ledger))}
    else
      {:error, :no_subject}
    end
  end
end
