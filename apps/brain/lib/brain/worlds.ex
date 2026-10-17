defmodule Brain.Worlds do
  @moduledoc """
  The worlds brain runs: one process tree per world, under
  `Brain.Worlds.Supervisor` (a `DynamicSupervisor`), registered by the world's
  id in `Brain.Worlds.Registry`. A world brain runs in is one whose tree is
  registered (`fetch/1`); a conversation, an analysis or a learning session
  names it.

  Brain does not decide which worlds exist. Whatever manages worlds (creating,
  loading, destroying and checkpointing them) drives these trees:

  - `create!/2` starts a new world's tree and seeds it by copy with its own
    copy of each trained world-scoped model: the entity model
    (`Brain.ML.EntityModel`), the embedder vocabulary (`Brain.Worlds.Embedder`),
    the Poincaré embeddings (`Brain.ML.Poincare.Embeddings`) and the triple
    scorer (`Brain.ML.KnowledgeGraph.TripleScorer`). A persistent world's
    copies are files in its directory (`world_path/1`); an ephemeral world has
    no directory, so its copies, and its knowledge
    (`Brain.KnowledgeStore.keep_in_memory/1`), are held in memory.
  - `start!/2` starts the tree of a world that already holds its copies, one
    loaded from its directory. Every world whose tree starts has its own Atlas
    graphs of every per-world kind (`Atlas.Graph.ensure_world_graphs!/1`),
    created before the tree is.
  - `stop!/1` stops the tree and drops what is loaded for the world: its
    models, its embedding caches (`Brain.EmbeddingCaches`) and the knowledge
    held in memory for it. A persistent world's files stay.

  Nothing is read from another world or from the trained models in place of a
  world's own copy.

  ## Where worlds live

  Every world's directory is under `base_path/0`: `config :brain,
  :training_worlds_path`, or, with `config :brain, :test_world_sandbox` on, a
  directory under the system temp directory, so a test world never lands
  among real ones.

  ## Reports

  What a turn learns that the world's manager records, an entity candidate
  (`report_candidate!/2`) or a world event (`report_event/3`), is broadcast on
  `Brain.PubSub` to the processes that called `subscribe_reports/0`. A
  candidate is validated into a `Brain.Worlds.EntityCandidate` in the
  reporting process first, so an incomplete one raises there.
  """

  alias Brain.EmbeddingCaches
  alias Brain.KnowledgeStore
  alias Brain.ML.EntityModel
  alias Brain.ML.KnowledgeGraph.TripleScorer
  alias Brain.ML.Poincare.Embeddings, as: PoincareEmbeddings
  alias Brain.Worlds.{Embedder, EntityCandidate}

  @registry Brain.Worlds.Registry
  @supervisor Brain.Worlds.Supervisor
  @modes [:persistent, :ephemeral]
  @reports_topic "worlds:reports"
  @test_sandbox_dir "chat_bot_test_worlds"

  @type mode :: :persistent | :ephemeral

  @doc "The registry and supervisor the worlds' trees run under, for `Brain.Application`."
  @spec child_specs() :: [Supervisor.child_spec() | {module(), keyword()}]
  def child_specs do
    [
      {Registry, keys: :unique, name: @registry},
      {DynamicSupervisor, name: @supervisor, strategy: :one_for_one}
    ]
  end

  @doc """
  Starts a new world's tree and seeds the world by copy (see the moduledoc).
  A world whose tree is already running is stopped first (`stop!/1`), so it
  starts again from fresh copies. Raises when a trained model to copy is
  missing.
  """
  @spec create!(String.t(), mode()) :: :ok
  def create!(world_id, mode) do
    world_id = world_id!(world_id)
    mode = mode!(mode)

    if running?(world_id), do: :ok = stop!(world_id)

    :ok = start!(world_id, mode)
    seed!(world_id, mode)
  end

  @doc """
  Starts the tree of a world that holds its copies already, after creating the
  world's graphs (see the moduledoc). Starting a world that runs in the same
  mode is `:ok`; one that runs in the other mode raises, as does a tree that
  does not start, and graphs that cannot be created
  (`Atlas.Graph.ensure_world_graphs!/1` refuses while unscoped graphs exist).
  """
  @spec start!(String.t(), mode()) :: :ok
  def start!(world_id, mode) do
    world_id = world_id!(world_id)
    mode = mode!(mode)

    Atlas.Graph.ensure_world_graphs!(world_id)

    case DynamicSupervisor.start_child(@supervisor, world_spec(world_id, mode)) do
      {:ok, _pid} ->
        :ok

      {:error, {:already_started, _pid}} ->
        case fetch(world_id) do
          {:ok, ^mode} ->
            :ok

          {:ok, running_mode} ->
            raise ArgumentError,
                  "Brain.Worlds: world #{inspect(world_id)} runs as #{inspect(running_mode)}, " <>
                    "not #{inspect(mode)}; stop it before starting it in another mode"
        end

      {:error, reason} ->
        raise "Brain.Worlds: world #{inspect(world_id)}'s tree did not start: #{inspect(reason)}"
    end
  end

  @doc """
  Stops the world's tree and drops what is loaded for the world (see the
  moduledoc). A world whose tree is not running has nothing loaded to keep, so
  stopping it again is `:ok`.
  """
  @spec stop!(String.t()) :: :ok
  def stop!(world_id) do
    world_id = world_id!(world_id)

    case Registry.lookup(@registry, world_id) do
      [{pid, _mode}] -> stop_tree!(world_id, pid)
      [] -> :ok
    end

    EntityModel.drop(world_id)
    Embedder.clear(world_id)
    :ok = EmbeddingCaches.forget_world(world_id)
    PoincareEmbeddings.drop(world_id)
    TripleScorer.drop(world_id)
    :ok = KnowledgeStore.drop_world(world_id)
  end

  @doc "The mode of a world brain runs, or `{:error, :not_found}` when its tree is not running."
  @spec fetch(String.t()) :: {:ok, mode()} | {:error, :not_found}
  def fetch(world_id) when is_binary(world_id) do
    case Registry.lookup(@registry, world_id) do
      [{_pid, mode}] -> {:ok, mode}
      [] -> {:error, :not_found}
    end
  end

  @doc "Whether the world's tree is running."
  @spec running?(String.t()) :: boolean()
  def running?(world_id) when is_binary(world_id), do: match?({:ok, _mode}, fetch(world_id))

  @doc "The ids of every world whose tree is running."
  @spec list() :: [String.t()]
  def list do
    Registry.select(@registry, [{{:"$1", :_, :_}, [], [:"$1"]}])
  end

  @doc """
  The directory every world's directory is under: `config :brain,
  :training_worlds_path`, or, with `config :brain, :test_world_sandbox` on, the
  test worlds' directory under the system temp directory. Raises when neither
  is configured.
  """
  @spec base_path() :: Path.t()
  def base_path do
    if Application.get_env(:brain, :test_world_sandbox) do
      Path.join(System.tmp_dir!(), @test_sandbox_dir)
    else
      case Application.get_env(:brain, :training_worlds_path) do
        path when is_binary(path) and path != "" ->
          path

        other ->
          raise "Brain.Worlds: config :brain, :training_worlds_path must name the directory worlds " <>
                  "are kept under, got #{inspect(other)}"
      end
    end
  end

  @doc "A world's directory: where it keeps its models, knowledge and checkpoints."
  @spec world_path(String.t()) :: Path.t()
  def world_path(world_id) when is_binary(world_id), do: Path.join(base_path(), world_id)

  @doc "Subscribes the caller to the worlds' reports (`report_candidate!/2`, `report_event/3`)."
  @spec subscribe_reports() :: :ok
  def subscribe_reports do
    :ok = Phoenix.PubSub.subscribe(Brain.PubSub, @reports_topic)
  end

  @doc """
  Reports an entity a world observed and has not promoted, validated into a
  `Brain.Worlds.EntityCandidate` here, in the caller (an incomplete one raises
  `ArgumentError`). Subscribers receive
  `{:world_entity_candidate, world_id, candidate}`.
  """
  @spec report_candidate!(String.t(), map() | keyword()) :: :ok
  def report_candidate!(world_id, attrs) when is_binary(world_id) do
    candidate = EntityCandidate.new!(attrs)
    :ok = Phoenix.PubSub.broadcast(Brain.PubSub, @reports_topic, {:world_entity_candidate, world_id, candidate})
  end

  @doc """
  Reports an event in a world. Subscribers receive
  `{:world_event, world_id, event_type, data}`.
  """
  @spec report_event(String.t(), atom(), map()) :: :ok
  def report_event(world_id, event_type, data) when is_binary(world_id) and is_atom(event_type) and is_map(data) do
    :ok = Phoenix.PubSub.broadcast(Brain.PubSub, @reports_topic, {:world_event, world_id, event_type, data})
  end

  defp seed!(world_id, :persistent) do
    EntityModel.seed!(world_id)
    Embedder.seed!(world_id)
    PoincareEmbeddings.seed!(world_id)
    TripleScorer.seed!(world_id)
    :ok
  end

  defp seed!(world_id, :ephemeral) do
    EntityModel.seed_in_memory!(world_id)
    Embedder.seed_in_memory!(world_id)
    PoincareEmbeddings.seed_in_memory!(world_id)
    TripleScorer.seed_in_memory!(world_id)
    :ok = KnowledgeStore.keep_in_memory(world_id)
  end

  # A world's tree is the process registered under its id, holding its mode.
  defp world_spec(world_id, mode) do
    %{
      id: world_id,
      start: {Agent, :start_link, [fn -> mode end, [name: {:via, Registry, {@registry, world_id, mode}}]]}
    }
  end

  defp stop_tree!(world_id, pid) do
    case DynamicSupervisor.terminate_child(@supervisor, pid) do
      :ok ->
        :ok

      {:error, :not_found} ->
        raise "Brain.Worlds: world #{inspect(world_id)}'s tree #{inspect(pid)} is registered but is not " <>
                "a child of #{inspect(@supervisor)}"
    end
  end

  defp world_id!(world_id) when is_binary(world_id) and world_id != "", do: world_id

  defp world_id!(other),
    do: raise(ArgumentError, "Brain.Worlds needs a world id, got: #{inspect(other)}")

  defp mode!(mode) when mode in @modes, do: mode

  defp mode!(other),
    do: raise(ArgumentError, "Brain.Worlds: a world's mode is one of #{inspect(@modes)}, got: #{inspect(other)}")
end
