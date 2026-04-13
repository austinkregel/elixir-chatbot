defmodule Brain.KnowledgeStore do
  @moduledoc "Knowledge store for persistent storage of learned facts about users, pets, rooms, devices, etc.\nProvides functions to store and retrieve structured knowledge.\n\n## World Scoping\n\nSupports both legacy persona-based storage and new world-scoped storage.\nWorld-scoped knowledge is stored in `priv/training_worlds/{world_id}/knowledge.json`.\nPersona-based knowledge remains in `priv/knowledge/{persona}.json`.\n"

  # World.Persistence is in a sibling umbrella app that depends on :brain.
  # It's available at runtime but not at compile time.
  @compile {:no_warn_undefined, World.Persistence}

  alias World.Persistence
  use GenServer
  require Logger

  @doc "Gets knowledge for a specific world.\n\n## Parameters\n  - world_id: The world to get knowledge from\n  - category: Optional category filter (e.g., \"people\", \"facts\")\n"
  def get_world_knowledge(world_id, category \\ nil) do
    GenServer.call(__MODULE__, {:get_world_knowledge, world_id, category})
  end

  @doc "Saves knowledge to a specific world.\n"
  def save_world_knowledge(world_id, knowledge) do
    GenServer.call(__MODULE__, {:save_world_knowledge, world_id, knowledge})
  end

  @doc "Adds an entry to a world's knowledge in a specific category.\n"
  def add_to_world(world_id, category, key, value) do
    GenServer.call(__MODULE__, {:add_to_world, world_id, category, key, value})
  end

  @doc "Removes an entry from a world's knowledge.\n"
  def remove_from_world(world_id, category, key) do
    GenServer.call(__MODULE__, {:remove_from_world, world_id, category, key})
  end

  @doc "Clears all knowledge for a world.\n"
  def clear_world(world_id) do
    GenServer.call(__MODULE__, {:clear_world, world_id})
  end

  @doc "Lists all worlds that have knowledge stored.\n"
  def list_knowledge_worlds do
    GenServer.call(__MODULE__, :list_knowledge_worlds)
  end

  @doc """
  Starts the KnowledgeStore GenServer.

  ## Options
    - `:name` - The name to register under (default: `#{__MODULE__}`)
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Loads knowledge for a persona.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
  """
  def load_knowledge(persona_name, opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, {:load_knowledge, persona_name})
  end

  def save_knowledge(persona_name, knowledge) do
    GenServer.call(__MODULE__, {:save_knowledge, persona_name, knowledge})
  end

  def add_pet(persona_name, pet_name, pet_info) do
    GenServer.call(__MODULE__, {:add_pet, persona_name, pet_name, pet_info})
  end

  def add_person(persona_name, person_name, person_info) do
    GenServer.call(__MODULE__, {:add_person, persona_name, person_name, person_info})
  end

  def add_room(persona_name, room_name, room_info) do
    GenServer.call(__MODULE__, {:add_room, persona_name, room_name, room_info})
  end

  def add_device(persona_name, device_name, device_info) do
    GenServer.call(__MODULE__, {:add_device, persona_name, device_name, device_info})
  end

  def add_place(persona_name, place_name, place_info) do
    GenServer.call(__MODULE__, {:add_place, persona_name, place_name, place_info})
  end

  def add_task(persona_name, task_name, task_info) do
    GenServer.call(__MODULE__, {:add_task, persona_name, task_name, task_info})
  end

  def add_event(persona_name, event_name, event_info) do
    GenServer.call(__MODULE__, {:add_event, persona_name, event_name, event_info})
  end

  def add_preference(persona_name, preference_name, preference_info) do
    GenServer.call(__MODULE__, {:add_preference, persona_name, preference_name, preference_info})
  end

  def add_relationship(persona_name, subject, relation, object, confidence \\ 1.0) do
    GenServer.call(
      __MODULE__,
      {:add_relationship, persona_name, subject, relation, object, confidence}
    )
  end

  def add_fact(persona_name, entity, fact, confidence \\ 1.0) do
    GenServer.call(__MODULE__, {:add_fact, persona_name, entity, fact, confidence})
  end

  def set_birthdate(persona_name, person_name, birthdate) do
    GenServer.call(__MODULE__, {:set_birthdate, persona_name, person_name, birthdate})
  end

  def add_favorite_holiday(persona_name, person_name, holiday) do
    GenServer.call(__MODULE__, {:add_favorite_holiday, persona_name, person_name, holiday})
  end

  def get_knowledge(persona_name, category \\ nil) do
    GenServer.call(__MODULE__, {:get_knowledge, persona_name, category})
  end

  @doc "Clears all learned knowledge for a persona.\n"
  def clear(persona_name) do
    GenServer.call(__MODULE__, {:clear, persona_name})
  end

  @doc "Checks if the knowledge store is ready.\n"
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end

  @impl true
  def init(_opts) do
    knowledge_dir = get_knowledge_dir()
    File.mkdir_p!(knowledge_dir)

    Logger.info("Knowledge store started", %{knowledge_dir: knowledge_dir})
    {:ok, %{}}
  end

  @impl true
  def handle_call({:load_knowledge, persona_name}, _from, state) do
    file_path = get_knowledge_file_path(persona_name)

    knowledge =
      case File.read(file_path) do
        {:ok, content} ->
          case Jason.decode(content) do
            {:ok, json} -> json
            {:error, _} -> %{}
          end

        {:error, _} ->
          %{}
      end

    Logger.debug("Loaded knowledge", %{persona_name: persona_name, size: map_size(knowledge)})
    {:reply, knowledge, state}
  end

  @impl true
  def handle_call({:save_knowledge, persona_name, knowledge}, _from, state) do
    file_path = get_knowledge_file_path(persona_name)

    case File.write(file_path, Jason.encode!(knowledge, pretty: true)) do
      :ok ->
        Logger.debug("Saved knowledge", %{persona_name: persona_name, size: map_size(knowledge)})
        {:reply, :ok, state}

      {:error, reason} ->
        Logger.error("Failed to save knowledge", %{persona_name: persona_name, reason: reason})
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:add_pet, persona_name, pet_name, pet_info}, _from, state) do
    knowledge = load_knowledge_data(persona_name)
    pets = Map.get(knowledge, "pets", %{})

    updated_pets =
      Map.put(pets, pet_name, Map.merge(pet_info, %{"type" => "pet", "name" => pet_name}))

    updated_knowledge = Map.put(knowledge, "pets", updated_pets)

    save_knowledge_data(persona_name, updated_knowledge)
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:add_person, persona_name, person_name, person_info}, _from, state) do
    knowledge = load_knowledge_data(persona_name)
    people = Map.get(knowledge, "people", %{})

    updated_people =
      Map.put(
        people,
        person_name,
        Map.merge(person_info, %{"type" => "person", "name" => person_name})
      )

    updated_knowledge = Map.put(knowledge, "people", updated_people)

    save_knowledge_data(persona_name, updated_knowledge)
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:add_room, persona_name, room_name, room_info}, _from, state) do
    knowledge = load_knowledge_data(persona_name)
    rooms = Map.get(knowledge, "rooms", %{})

    updated_rooms =
      Map.put(rooms, room_name, Map.merge(room_info, %{"type" => "room", "name" => room_name}))

    updated_knowledge = Map.put(knowledge, "rooms", updated_rooms)

    save_knowledge_data(persona_name, updated_knowledge)
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:add_device, persona_name, device_name, device_info}, _from, state) do
    knowledge = load_knowledge_data(persona_name)
    devices = Map.get(knowledge, "devices", %{})

    updated_devices =
      Map.put(
        devices,
        device_name,
        Map.merge(device_info, %{"type" => "device", "name" => device_name})
      )

    updated_knowledge = Map.put(knowledge, "devices", updated_devices)

    save_knowledge_data(persona_name, updated_knowledge)
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:set_birthdate, persona_name, person_name, birthdate}, _from, state) do
    knowledge = load_knowledge_data(persona_name)
    people = Map.get(knowledge, "people", %{})

    case Map.get(people, person_name) do
      nil ->
        new_person = %{"type" => "person", "name" => person_name, "birthdate" => birthdate}
        updated_people = Map.put(people, person_name, new_person)
        updated_knowledge = Map.put(knowledge, "people", updated_people)
        save_knowledge_data(persona_name, updated_knowledge)

      existing_person ->
        updated_person = Map.put(existing_person, "birthdate", birthdate)
        updated_people = Map.put(people, person_name, updated_person)
        updated_knowledge = Map.put(knowledge, "people", updated_people)
        save_knowledge_data(persona_name, updated_knowledge)
    end

    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:add_favorite_holiday, persona_name, person_name, holiday}, _from, state) do
    knowledge = load_knowledge_data(persona_name)
    people = Map.get(knowledge, "people", %{})

    case Map.get(people, person_name) do
      nil ->
        new_person = %{
          "type" => "person",
          "name" => person_name,
          "favorite_holidays" => [holiday]
        }

        updated_people = Map.put(people, person_name, new_person)
        updated_knowledge = Map.put(knowledge, "people", updated_people)
        save_knowledge_data(persona_name, updated_knowledge)

      existing_person ->
        existing_holidays = Map.get(existing_person, "favorite_holidays", [])

        updated_holidays =
          if holiday in existing_holidays do
            existing_holidays
          else
            existing_holidays ++ [holiday]
          end

        updated_person = Map.put(existing_person, "favorite_holidays", updated_holidays)
        updated_people = Map.put(people, person_name, updated_person)
        updated_knowledge = Map.put(knowledge, "people", updated_people)
        save_knowledge_data(persona_name, updated_knowledge)
    end

    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:add_place, persona_name, place_name, place_info}, _from, state) do
    knowledge = load_knowledge_data(persona_name)
    places = Map.get(knowledge, "places", %{})

    updated_places =
      Map.put(
        places,
        place_name,
        Map.merge(place_info, %{"type" => "place", "name" => place_name})
      )

    updated_knowledge = Map.put(knowledge, "places", updated_places)

    save_knowledge_data(persona_name, updated_knowledge)
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:add_task, persona_name, task_name, task_info}, _from, state) do
    knowledge = load_knowledge_data(persona_name)
    tasks = Map.get(knowledge, "tasks", %{})

    updated_tasks =
      Map.put(tasks, task_name, Map.merge(task_info, %{"type" => "task", "name" => task_name}))

    updated_knowledge = Map.put(knowledge, "tasks", updated_tasks)

    save_knowledge_data(persona_name, updated_knowledge)
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:add_event, persona_name, event_name, event_info}, _from, state) do
    knowledge = load_knowledge_data(persona_name)
    events = Map.get(knowledge, "events", %{})

    updated_events =
      Map.put(
        events,
        event_name,
        Map.merge(event_info, %{"type" => "event", "name" => event_name})
      )

    updated_knowledge = Map.put(knowledge, "events", updated_events)

    save_knowledge_data(persona_name, updated_knowledge)
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:add_preference, persona_name, preference_name, preference_info}, _from, state) do
    knowledge = load_knowledge_data(persona_name)
    preferences = Map.get(knowledge, "preferences", %{})

    updated_preferences =
      Map.put(
        preferences,
        preference_name,
        Map.merge(preference_info, %{"type" => "preference", "name" => preference_name})
      )

    updated_knowledge = Map.put(knowledge, "preferences", updated_preferences)

    save_knowledge_data(persona_name, updated_knowledge)
    {:reply, :ok, state}
  end

  @impl true
  def handle_call(
        {:add_relationship, persona_name, subject, relation, object, confidence},
        _from,
        state
      ) do
    knowledge = load_knowledge_data(persona_name)
    relationships = Map.get(knowledge, "relationships", [])

    new_relationship = %{
      "subject" => subject,
      "relation" => relation,
      "object" => object,
      "confidence" => confidence,
      "timestamp" => System.system_time(:millisecond)
    }

    updated_relationships = upsert_relationship(relationships, new_relationship)
    updated_knowledge = Map.put(knowledge, "relationships", updated_relationships)

    save_knowledge_data(persona_name, updated_knowledge)
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:add_fact, persona_name, entity, fact, confidence}, _from, state) do
    knowledge = load_knowledge_data(persona_name)
    facts = Map.get(knowledge, "facts", [])

    new_fact = %{
      "entity" => entity,
      "fact" => fact,
      "confidence" => confidence,
      "timestamp" => System.system_time(:millisecond)
    }

    updated_facts = upsert_fact(facts, new_fact)
    updated_knowledge = Map.put(knowledge, "facts", updated_facts)

    save_knowledge_data(persona_name, updated_knowledge)
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:get_knowledge, persona_name, category}, _from, state) do
    knowledge = load_knowledge_data(persona_name)

    result =
      case category do
        nil -> knowledge
        category -> Map.get(knowledge, category, %{})
      end

    {:reply, result, state}
  end

  @impl true
  def handle_call({:clear, persona_name}, _from, state) do
    empty_knowledge = %{
      "people" => %{},
      "pets" => %{},
      "rooms" => %{},
      "devices" => %{},
      "places" => %{},
      "tasks" => %{},
      "events" => %{},
      "preferences" => %{},
      "relationships" => [],
      "facts" => []
    }

    save_knowledge_data(persona_name, empty_knowledge)
    Logger.info("Cleared knowledge for persona", %{persona: persona_name})
    {:reply, :ok, state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  @impl true
  def handle_call({:get_world_knowledge, world_id, nil}, _from, state) do
    knowledge = load_world_knowledge_data(world_id)
    {:reply, knowledge, state}
  end

  @impl true
  def handle_call({:get_world_knowledge, world_id, category}, _from, state) do
    knowledge = load_world_knowledge_data(world_id)
    category_data = Map.get(knowledge, category, %{})
    {:reply, category_data, state}
  end

  @impl true
  def handle_call({:save_world_knowledge, world_id, knowledge}, _from, state) do
    result = save_world_knowledge_data(world_id, knowledge)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:add_to_world, world_id, category, key, value}, _from, state) do
    knowledge = load_world_knowledge_data(world_id)

    category_data =
      case Map.get(knowledge, category, %{}) do
        data when is_map(data) -> data
        _ -> %{}
      end

    updated_category = Map.put(category_data, key, value)
    updated_knowledge = Map.put(knowledge, category, updated_category)

    result = save_world_knowledge_data(world_id, updated_knowledge)

    case result do
      :ok ->
        Logger.debug("Added to world knowledge", %{
          world_id: world_id,
          category: category,
          key: key
        })

      _ ->
        :ok
    end

    {:reply, result, state}
  end

  @impl true
  def handle_call({:remove_from_world, world_id, category, key}, _from, state) do
    knowledge = load_world_knowledge_data(world_id)

    category_data = Map.get(knowledge, category, %{})
    updated_category = Map.delete(category_data, key)
    updated_knowledge = Map.put(knowledge, category, updated_category)

    result = save_world_knowledge_data(world_id, updated_knowledge)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:clear_world, world_id}, _from, state) do
    empty_knowledge = %{
      "people" => %{},
      "pets" => %{},
      "rooms" => %{},
      "devices" => %{},
      "places" => %{},
      "tasks" => %{},
      "events" => %{},
      "preferences" => %{},
      "relationships" => [],
      "facts" => []
    }

    result = save_world_knowledge_data(world_id, empty_knowledge)
    Logger.info("Cleared knowledge for world", %{world_id: world_id})
    {:reply, result, state}
  end

  @impl true
  def handle_call(:list_knowledge_worlds, _from, state) do
    worlds_dir = Persistence.base_path()

    worlds =
      if File.dir?(worlds_dir) do
        worlds_dir
        |> File.ls!()
        |> Enum.filter(fn name ->
          knowledge_path = Path.join([worlds_dir, name, "knowledge.json"])
          File.exists?(knowledge_path)
        end)
      else
        []
      end

    {:reply, {:ok, worlds}, state}
  end

  defp get_knowledge_dir do
    case Application.get_env(:brain, :knowledge_dir) do
      nil -> Brain.priv_path("knowledge")
      dir -> dir
    end
  end

  defp get_knowledge_file_path(persona_name) do
    Path.join(get_knowledge_dir(), "#{persona_name}.json")
  end

  defp load_knowledge_data(persona_name) do
    file_path = get_knowledge_file_path(persona_name)

    case File.read(file_path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, json} -> json
          {:error, _} -> %{}
        end

      {:error, _} ->
        %{}
    end
  end

  defp save_knowledge_data(persona_name, knowledge) do
    file_path = get_knowledge_file_path(persona_name)
    File.write(file_path, Jason.encode!(knowledge, pretty: true))
  end

  defp get_world_knowledge_path(world_id) do
    Path.join(Persistence.world_path(world_id), "knowledge.json")
  end

  defp load_world_knowledge_data(world_id) do
    file_path = get_world_knowledge_path(world_id)

    case File.read(file_path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, json} when is_map(json) -> json
          {:ok, _non_map} -> %{}
          {:error, _} -> %{}
        end

      {:error, _} ->
        %{}
    end
  end

  defp save_world_knowledge_data(world_id, knowledge) do
    file_path = get_world_knowledge_path(world_id)
    file_path |> Path.dirname() |> File.mkdir_p!()

    case File.write(file_path, Jason.encode!(knowledge, pretty: true)) do
      :ok ->
        :ok

      {:error, reason} ->
        Logger.error("Failed to save world knowledge", %{world_id: world_id, reason: reason})
        {:error, reason}
    end
  end

  defp upsert_relationship(existing, new_rel) do
    key = fn rel ->
      {
        normalize(Map.get(rel, "subject")),
        normalize(Map.get(rel, "relation")),
        normalize(Map.get(rel, "object"))
      }
    end

    {s, r, o} = key.(new_rel)

    case Enum.find_index(existing, fn rel -> key.(rel) == {s, r, o} end) do
      nil ->
        existing ++ [new_rel]

      idx ->
        old = Enum.at(existing, idx)

        merged =
          old
          |> Map.put(
            "confidence",
            max(Map.get(old, "confidence", 0.0), Map.get(new_rel, "confidence", 0.0))
          )
          |> Map.put(
            "timestamp",
            max(Map.get(old, "timestamp", 0), Map.get(new_rel, "timestamp", 0))
          )

        List.replace_at(existing, idx, merged)
    end
  end

  defp upsert_fact(existing, new_fact) do
    key = fn f -> {normalize(Map.get(f, "entity")), normalize(Map.get(f, "fact"))} end
    {e, ft} = key.(new_fact)

    case Enum.find_index(existing, fn f -> key.(f) == {e, ft} end) do
      nil ->
        existing ++ [new_fact]

      idx ->
        old = Enum.at(existing, idx)

        merged =
          old
          |> Map.put(
            "confidence",
            max(Map.get(old, "confidence", 0.0), Map.get(new_fact, "confidence", 0.0))
          )
          |> Map.put(
            "timestamp",
            max(Map.get(old, "timestamp", 0), Map.get(new_fact, "timestamp", 0))
          )

        List.replace_at(existing, idx, merged)
    end
  end

  defp normalize(nil) do
    ""
  end

  defp normalize(val) when is_binary(val) do
    val |> String.downcase() |> String.trim()
  end

  defp normalize(val) do
    to_string(val) |> String.downcase() |> String.trim()
  end
end