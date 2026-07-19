defmodule Fleet.Systems.System do
  @moduledoc "One system in the ship's layered inventory (see `Fleet.Systems`)."
  @enforce_keys [:id, :name, :layer, :deck, :section, :status, :ship_id]
  defstruct [:id, :name, :layer, :deck, :section, :status, :key_metric, :ship_id]

  @type layer :: :service | :process | :subsystem
  @type status :: :up | :degraded | :down | :idle | :static | :unknown
  @type t :: %__MODULE__{
          id: String.t(),
          name: String.t(),
          layer: layer(),
          deck: atom(),
          section: String.t(),
          status: status(),
          key_metric: String.t() | nil,
          ship_id: String.t()
        }
end

defmodule Fleet.Systems do
  @moduledoc """
  The ship's **layered systems inventory** — a runtime observer that gives one
  view of everything running on this instance, for the black-box board and the
  crew's `systems.read` tool.

  Three layers (LCARS "the ship's computer that sees the whole ship"):
    * **Layer 0 — services** (~5): Postgres/AGE, the generation backend, Ouro,
      PubSub. Live health.
    * **Layer 1 — processes** (~60): the supervised GenServers. Reuses
      `Brain.SystemStatus.get_all_genservers_status/0` for the ~51 brain systems and
      walks the atlas/world/chat_web/fleet root supervisors for the rest.
    * **Layer 2 — subsystems** (200+): every module by namespace
      (`:application.get_key(app, :modules)`). A pure-function module is `:static`
      (compiled/loaded, no runtime health); health that can be *down* comes from
      Layers 0–1.

  A runtime observer, not a compile dependency: it uses the rich APIs where a real
  dep exists (`Brain.SystemStatus`, `Atlas.Stats`, `Brain.ML.Generation`) and
  generic introspection elsewhere. Every system is stamped with `Fleet.Ship.id/0`.

  **Reads flow through `Fleet.Clearance`** at the call site (the `systems.read` tool
  and the human query console) — this module produces the data; it does not decide
  who may see it.
  """

  alias Fleet.Systems.System
  require Logger

  @apps [:brain, :atlas, :world, :chat_web, :fleet, :fourth_wall, :tasks]
  @root_supervisors [
    {Atlas.Supervisor, :atlas},
    {World.Supervisor, :world},
    {ChatWeb.Supervisor, :chat_web},
    {Fleet.Supervisor, :fleet}
  ]

  @doc "The rolled-up ship health (reuses `Brain.SystemStatus.get_health_indicators/0`)."
  def health do
    base = safe(fn -> Brain.SystemStatus.get_health_indicators() end) || %{}
    svcs = services()

    Map.merge(base, %{
      ship_id: Fleet.Ship.id(),
      services_up: Enum.count(svcs, &(&1.status == :up)),
      services_total: length(svcs),
      generated_at: DateTime.utc_now()
    })
  end

  @doc "A full layered snapshot: services + processes + subsystems + rolled-up counts."
  def snapshot do
    svcs = services()
    procs = processes()
    subs = subsystems()
    live = svcs ++ procs

    %{
      ship_id: Fleet.Ship.id(),
      generated_at: DateTime.utc_now(),
      health: health(),
      services: svcs,
      processes: procs,
      subsystems: subs,
      counts: %{
        services: length(svcs),
        processes: length(procs),
        subsystems: length(subs),
        total: length(svcs) + length(procs) + length(subs),
        up: Enum.count(live, &(&1.status == :up)),
        degraded: Enum.count(live, &(&1.status == :degraded)),
        down: Enum.count(live, &(&1.status == :down))
      }
    }
  end

  @doc "One system by id, or nil."
  def get(id), do: Enum.find(services() ++ processes() ++ subsystems(), &(&1.id == id))

  # ── Layer 0: external services ─────────────────────────────────────────────

  def services do
    gen = safe(fn -> Brain.ML.Generation.name() end) || :unknown

    base = [
      sys("svc:postgres-age", "Postgres / Apache AGE", :service, :atlas, "storage", db_status()),
      sys("svc:generation", "Generation (#{gen})", :service, :brain, "ml",
        bool_status(safe(fn -> Brain.ML.Generation.ready?() end)), to_string(gen)),
      sys("svc:pubsub", "Brain.PubSub", :service, :brain, "infra", whereis_status(Brain.PubSub))
    ]

    base ++ ouro_service(gen)
  end

  defp ouro_service(:ouro_sidecar) do
    info = safe(fn -> Brain.ML.Ouro.Model.info() end) || %{}
    [sys("svc:ouro", "Ouro sidecar", :service, :brain, "ml", bool_status(info[:ready]))]
  end

  defp ouro_service(_), do: []

  # ── Layer 1: supervised processes ──────────────────────────────────────────

  def processes, do: brain_processes() ++ supervised_processes()

  defp brain_processes do
    case safe(fn -> Brain.SystemStatus.get_all_genservers_status() end) do
      %{categories: cats} when is_map(cats) ->
        for {category, mods} <- cats, {module, st} <- mods do
          sys(
            "proc:#{inspect(module)}",
            (is_map(st) && st[:label]) || inspect(module),
            :process,
            :brain,
            to_string(category),
            process_status(st),
            process_metric(st)
          )
        end

      _ ->
        []
    end
  end

  # The non-brain apps: walk each root supervisor's direct children.
  defp supervised_processes do
    for {sup, deck} <- @root_supervisors,
        is_pid(Process.whereis(sup)),
        {id, child, _type, _mods} <- safe(fn -> Supervisor.which_children(sup) end) || [] do
      status = if is_pid(child) and Process.alive?(child), do: :up, else: :down
      sys("proc:#{deck}:#{inspect(id)}", child_name(id), :process, deck, "supervised", status)
    end
  end

  defp child_name({:via, _, _} = id), do: inspect(id)
  defp child_name(id) when is_atom(id), do: inspect(id)
  defp child_name(id), do: inspect(id)

  # ── Layer 2: logical subsystems (modules by namespace) ─────────────────────

  def subsystems do
    for app <- @apps, module <- app_modules(app) do
      sys("mod:#{inspect(module)}", inspect(module), :subsystem, app, namespace(module), :static)
    end
  end

  defp app_modules(app) do
    case :application.get_key(app, :modules) do
      {:ok, mods} -> mods
      _ -> []
    end
  end

  # First two namespace segments (e.g. Brain.Analysis.Foo -> "Brain.Analysis").
  defp namespace(module) do
    inspect(module) |> String.split(".") |> Enum.take(2) |> Enum.join(".")
  end

  # ── status derivation ──────────────────────────────────────────────────────

  defp process_status(st) when is_map(st) do
    cond do
      st[:running] == false -> :down
      st[:ready] == false -> :degraded
      st[:running] == true -> :up
      true -> :unknown
    end
  end

  defp process_status(_), do: :unknown

  defp process_metric(st) when is_map(st) do
    case st[:message_queue_len] do
      n when is_integer(n) and n > 0 -> "q=#{n}"
      _ -> nil
    end
  end

  defp process_metric(_), do: nil

  defp db_status do
    if safe(fn -> Atlas.Stats.connected?() end) == true, do: :up, else: :down
  end

  defp bool_status(true), do: :up
  defp bool_status(_), do: :down

  defp whereis_status(name), do: if(is_pid(Process.whereis(name)), do: :up, else: :down)

  defp sys(id, name, layer, deck, section, status, key_metric \\ nil) do
    %System{
      id: id,
      name: name,
      layer: layer,
      deck: deck,
      section: section,
      status: status,
      key_metric: key_metric,
      ship_id: Fleet.Ship.id()
    }
  end

  defp safe(fun) do
    fun.()
  rescue
    e -> Logger.debug("Fleet.Systems: probe failed: #{inspect(e)}"); nil
  catch
    :exit, _ -> nil
  end
end
