defmodule Brain.Response.PhraseInventory do
  @moduledoc """
  GenServer that loads and indexes the phrase fragment inventory for
  the lattice-based surface realizer.

  Fragments are dual-indexed by chunk_type (greeting, body, etc.) and
  by primitive_type:variant (acknowledgment:social, content:enriched, etc.).
  """

  use GenServer

  require Logger

  @inventory_path "priv/ml_models/lattice/phrase_inventory.json"
  @runtime_path "priv/ml_models/lattice/phrase_inventory_runtime.json"
  @config_path "priv/response/system_config.json"

  defmodule Fragment do
    @moduledoc false
    defstruct [
      :id,
      :text,
      :chunk_type,
      :primitive_type,
      :primitive_variant,
      :tone,
      :tone_vector,
      :prototype_vector,
      :source_intent,
      slots: [],
      enrichment_fields: [],
      conditions: %{},
      origin: "unknown"
    ]
  end

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  def ready?(name \\ __MODULE__) do
    try do
      GenServer.call(name, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  def lookup_by_chunk_type(chunk_type, name \\ __MODULE__) do
    GenServer.call(name, {:lookup_chunk, chunk_type}, 5_000)
  end

  def lookup_by_primitive(primitive_type, variant, name \\ __MODULE__) do
    key = primitive_key(primitive_type, variant)
    GenServer.call(name, {:lookup_primitive, key}, 5_000)
  end

  def lookup_by_primitive_with_tone(primitive_type, variant, desired_tone_vector, name \\ __MODULE__) do
    key = primitive_key(primitive_type, variant)
    GenServer.call(name, {:lookup_primitive_tone, key, desired_tone_vector}, 5_000)
  end

  def stats(name \\ __MODULE__) do
    try do
      GenServer.call(name, :stats, 5_000)
    catch
      :exit, _ -> %{status: :unavailable}
    end
  end

  def reload(name \\ __MODULE__) do
    GenServer.call(name, :reload, 30_000)
  end

  def add_fragments(fragments, name \\ __MODULE__) when is_list(fragments) do
    GenServer.cast(name, {:add_fragments, fragments})
  end

  def tone_vectors(name \\ __MODULE__) do
    GenServer.call(name, :tone_vectors, 5_000)
  end

  # Server

  @impl true
  def init(_opts) do
    state = load_inventory()
    {:ok, state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, state.loaded, state}
  end

  def handle_call({:lookup_chunk, chunk_type}, _from, state) do
    key = to_string(chunk_type)
    fragments = Map.get(state.by_chunk_type, key, [])
    {:reply, fragments, state}
  end

  def handle_call({:lookup_primitive, key}, _from, state) do
    fragments = Map.get(state.by_primitive, key, [])
    {:reply, fragments, state}
  end

  def handle_call({:lookup_primitive_tone, key, desired_tone_vector}, _from, state) do
    fragments = Map.get(state.by_primitive, key, [])

    sorted =
      fragments
      |> Enum.map(fn frag ->
        compat = tone_compatibility(desired_tone_vector, frag.tone_vector)
        {frag, compat}
      end)
      |> Enum.sort_by(fn {_f, compat} -> compat end, :desc)

    {:reply, sorted, state}
  end

  def handle_call(:stats, _from, state) do
    chunk_counts =
      state.by_chunk_type
      |> Enum.map(fn {k, v} -> {k, length(v)} end)
      |> Map.new()

    primitive_counts =
      state.by_primitive
      |> Enum.map(fn {k, v} -> {k, length(v)} end)
      |> Map.new()

    tone_dist =
      state.all_fragments
      |> Enum.frequencies_by(& &1.tone)

    stats = %{
      status: :ok,
      loaded: state.loaded,
      total_fragments: length(state.all_fragments),
      by_chunk_type: chunk_counts,
      by_primitive: primitive_counts,
      tone_distribution: tone_dist
    }

    {:reply, stats, state}
  end

  def handle_call(:reload, _from, _state) do
    new_state = load_inventory()
    {:reply, :ok, new_state}
  end

  def handle_call(:tone_vectors, _from, state) do
    {:reply, state.tone_config, state}
  end

  @impl true
  def handle_cast({:add_fragments, new_fragments}, state) do
    parsed = Enum.map(new_fragments, &parse_fragment/1)
    all = state.all_fragments ++ parsed

    new_state = %{state |
      all_fragments: all,
      by_chunk_type: index_by_chunk_type(all),
      by_primitive: index_by_primitive(all)
    }

    {:noreply, new_state}
  end

  # Internal

  defp load_inventory do
    inventory_path = brain_priv(@inventory_path)
    runtime_path = brain_priv(@runtime_path)
    config_path = brain_priv(@config_path)

    tone_config = load_tone_config(config_path)
    fragments = load_fragments_file(inventory_path) ++ load_fragments_file(runtime_path)
    loaded = fragments != []

    if loaded do
      Logger.info("PhraseInventory: loaded #{length(fragments)} fragments")
    else
      Logger.debug("PhraseInventory: no inventory files found, starting empty")
    end

    %{
      loaded: loaded,
      all_fragments: fragments,
      by_chunk_type: index_by_chunk_type(fragments),
      by_primitive: index_by_primitive(fragments),
      tone_config: tone_config
    }
  end

  defp load_fragments_file(path) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, %{"fragments" => fragments_map}} when is_map(fragments_map) ->
            fragments_map
            |> Enum.flat_map(fn {_chunk_type, frags} ->
              Enum.map(frags, &parse_fragment/1)
            end)

          _ ->
            []
        end

      {:error, _} ->
        []
    end
  end

  defp load_tone_config(path) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, %{"tone" => tone}} -> tone
          _ -> %{}
        end

      {:error, _} ->
        %{}
    end
  end

  defp parse_fragment(map) when is_map(map) do
    %Fragment{
      id: Map.get(map, "id"),
      text: Map.get(map, "text", ""),
      chunk_type: Map.get(map, "chunk_type", "body"),
      primitive_type: Map.get(map, "primitive_type", "content"),
      primitive_variant: Map.get(map, "primitive_variant"),
      tone: Map.get(map, "tone", "neutral"),
      tone_vector: Map.get(map, "tone_vector", List.duplicate(0.0, 10)),
      prototype_vector: Map.get(map, "prototype_vector", []),
      source_intent: Map.get(map, "source_intent"),
      slots: Map.get(map, "slots", []),
      enrichment_fields: Map.get(map, "enrichment_fields", []),
      conditions: Map.get(map, "conditions", %{}),
      origin: Map.get(map, "origin", "unknown")
    }
  end

  defp index_by_chunk_type(fragments) do
    Enum.group_by(fragments, & &1.chunk_type)
  end

  defp index_by_primitive(fragments) do
    Enum.group_by(fragments, fn f ->
      primitive_key(f.primitive_type, f.primitive_variant)
    end)
  end

  defp primitive_key(type, nil), do: "#{type}:_"
  defp primitive_key(type, variant), do: "#{type}:#{variant}"

  defp tone_compatibility(desired, fragment_tone) when is_list(desired) and is_list(fragment_tone) do
    dist = cosine_distance(desired, fragment_tone)
    max(0.0, 1.0 - dist)
  end

  defp tone_compatibility(_, _), do: 0.5

  defp cosine_distance(a, b) when length(a) == length(b) do
    1.0 - cosine_similarity(a, b)
  end

  defp cosine_distance(_, _), do: 1.0

  defp cosine_similarity(a, b) do
    dot = Enum.zip(a, b) |> Enum.reduce(0.0, fn {x, y}, sum -> sum + x * y end)
    mag_a = :math.sqrt(Enum.reduce(a, 0.0, fn x, sum -> sum + x * x end))
    mag_b = :math.sqrt(Enum.reduce(b, 0.0, fn x, sum -> sum + x * x end))

    if mag_a == 0.0 or mag_b == 0.0, do: 0.0, else: dot / (mag_a * mag_b)
  end

  defp brain_priv(relative) do
    case :code.priv_dir(:brain) do
      {:error, _} -> Path.join("apps/brain", relative)
      priv_dir -> Path.join(priv_dir, Path.relative_to(relative, "priv"))
    end
  end
end
