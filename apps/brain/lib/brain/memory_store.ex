defmodule Brain.MemoryStore do
  @moduledoc """
  Memory store for persistent storage of conversation history and thoughts.
  Provides functions to store and retrieve conversation memory.
  """

  use GenServer
  require Logger

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Starts the MemoryStore GenServer.

  ## Options
    - `:name` - The name to register under (default: `#{__MODULE__}`)
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Loads all memory for a persona.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
  """
  def load_all(persona_name, opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, {:load_all, persona_name})
  end

  @doc """
  Appends a thought to memory.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
  """
  def append_thought(persona_name, role, text, tags \\ [], opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, {:append_thought, persona_name, role, text, tags})
  end

  @doc """
  Gets a window of recent memory entries.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
  """
  def get_memory_window(persona_name, max_entries \\ 100, opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, {:get_memory_window, persona_name, max_entries})
  end

  @doc """
  Clears all memory for a persona.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
  """
  def clear_memory(persona_name, opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, {:clear_memory, persona_name})
  end

  @doc """
  Checks if the memory store is ready.
  """
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end

  # Server Callbacks

  @impl true
  def init(_opts) do
    # Create memory directory if it doesn't exist
    memory_dir = get_memory_dir()
    File.mkdir_p!(memory_dir)

    Logger.info("Memory store started", %{memory_dir: memory_dir})
    {:ok, %{}}
  end

  @impl true
  def handle_call({:load_all, persona_name}, _from, state) do
    file_path = get_memory_file_path(persona_name)

    memory =
      case File.read(file_path) do
        {:ok, content} ->
          case Jason.decode(content) do
            {:ok, json} -> json
            {:error, _} -> []
          end

        {:error, _} ->
          []
      end

    Logger.debug("Loaded memory", %{persona_name: persona_name, entries: length(memory)})
    {:reply, memory, state}
  end

  @impl true
  def handle_call({:append_thought, persona_name, role, text, tags}, _from, state) do
    entry = %{
      "role" => role,
      "text" => String.trim(text),
      "tags" => tags,
      "timestamp" => System.system_time(:millisecond),
      "id" => generate_id()
    }

    memory = load_memory_data(persona_name)
    updated_memory = memory ++ [entry]
    save_memory_data(persona_name, updated_memory)

    Logger.debug("Appended thought", %{
      persona_name: persona_name,
      role: role,
      text_length: String.length(text),
      tags: tags
    })

    {:reply, entry, state}
  end

  @impl true
  def handle_call({:get_memory_window, persona_name, max_entries}, _from, state) do
    memory = load_memory_data(persona_name)

    # Get the most recent entries
    window =
      memory
      |> Enum.take(-max_entries)

    Logger.debug("Retrieved memory window", %{
      persona_name: persona_name,
      total_entries: length(memory),
      window_size: length(window)
    })

    {:reply, window, state}
  end

  @impl true
  def handle_call({:clear_memory, persona_name}, _from, state) do
    file_path = get_memory_file_path(persona_name)

    case File.write(file_path, "[]") do
      :ok ->
        Logger.info("Cleared memory", %{persona_name: persona_name})
        {:reply, :ok, state}

      {:error, reason} ->
        Logger.error("Failed to clear memory", %{persona_name: persona_name, reason: reason})
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  # Private Functions

  defp get_memory_dir do
    case Application.get_env(:brain, :memory_dir) do
      nil -> Brain.priv_path("memory")
      dir -> dir
    end
  end

  defp get_memory_file_path(persona_name) do
    Path.join(get_memory_dir(), "#{persona_name}.json")
  end

  defp load_memory_data(persona_name) do
    file_path = get_memory_file_path(persona_name)

    case File.read(file_path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, json} -> json
          {:error, _} -> []
        end

      {:error, _} ->
        []
    end
  end

  defp save_memory_data(persona_name, memory) do
    file_path = get_memory_file_path(persona_name)
    File.write(file_path, Jason.encode!(memory, pretty: true))
  end

  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end
end
