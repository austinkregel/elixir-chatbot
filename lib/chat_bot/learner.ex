defmodule ChatBot.Learner do
  @moduledoc """
  NLP-based learner module for extracting facts from user inputs.
  Uses classical NLP entity recognition and relationship extraction to build
  dynamic, adaptable knowledge about people, pets, rooms, devices, places, tasks, etc.
  """

  require Logger
  alias ChatBot.KnowledgeStore
  alias ChatBot.MemoryStore
  alias ChatBot.FactDatabase.Integration

  # Client API

  @doc """
  Learn from user input using classical NLP entity extraction.
  Extracts entities and stores them in the knowledge store.
  """
  def learn_from_input(persona_name, input) do
    Logger.debug("Learner.learn_from_input called", %{persona_name: persona_name, input: input})

    # Use classical NLP pipeline to extract entities
    entities = ChatBot.ML.EntityExtractor.extract_entities(input)

    if length(entities) > 0 do
      # Convert to extracted_data format and process
      extracted_data = %{
        "entities" =>
          Enum.map(entities, fn entity ->
            %{
              "name" => Map.get(entity, :value, ""),
              "type" => Map.get(entity, :entity, "unknown"),
              "properties" => %{},
              "confidence" => Map.get(entity, :confidence, 0.9)
            }
          end),
        "relationships" => [],
        "facts" => [],
        "context" => input
      }

      process_extracted_data(persona_name, extracted_data, input)
      {:ok, extracted_data}
    else
      # Store as general memory if no specific entities found
      store_general_memory(persona_name, input)
      {:ok, %{"type" => "general_memory", "input" => input}}
    end
  end

  @doc """
  Learn from pre-extracted classical NLP entities.
  Accepts entities from the classical NLP pipeline and stores them directly.
  """
  def learn_from_classical_extraction(persona_name, entities, input) do
    Logger.debug("Learner.learn_from_classical_extraction called", %{
      persona_name: persona_name,
      entities_count: length(entities),
      input: input
    })

    # Convert classical NLP entities to extracted_data format
    # Handle both map and struct entity formats safely
    extracted_data = %{
      "entities" =>
        Enum.map(entities, fn entity ->
          %{
            "name" => get_entity_field(entity, [:value, "value"]),
            "type" => get_entity_field(entity, [:entity, "entity", :type, "type"]),
            "properties" => %{},
            "confidence" => get_entity_field(entity, [:confidence, "confidence"]) || 0.9
          }
        end),
      "relationships" => [],
      "facts" => [],
      "context" => input
    }

    process_extracted_data(persona_name, extracted_data, input)
    {:ok, extracted_data}
  end

  # Helper to safely get a field from an entity (map or struct)
  defp get_entity_field(entity, keys) when is_list(keys) do
    Enum.find_value(keys, fn key ->
      case entity do
        %{^key => value} when not is_nil(value) -> value
        _ -> nil
      end
    end)
  end

  # Process extracted NLP data

  defp process_extracted_data(persona_name, data, original_input) do
    # Process all entities
    entities = Map.get(data, "entities", [])
    relationships = Map.get(data, "relationships", [])
    facts = Map.get(data, "facts", [])

    # Group entities by type for efficient processing
    entities_by_type = Enum.group_by(entities, & &1["type"])

    # Process each entity type
    Enum.each(entities_by_type, fn {type, entities_of_type} ->
      process_entities_of_type(persona_name, type, entities_of_type)
    end)

    # Process relationships
    process_relationships(persona_name, relationships)

    # Process facts
    process_facts(persona_name, facts)

    # Store the original input as context
    store_context_memory(persona_name, original_input, data)

    Logger.info("Processed NLP extraction", %{
      persona_name: persona_name,
      entities_count: length(entities),
      relationships_count: length(relationships),
      facts_count: length(facts)
    })
  end

  defp process_entities_of_type(persona_name, type, entities) do
    case type do
      "person" ->
        Enum.each(entities, &process_person_entity(persona_name, &1))

      "pet" ->
        Enum.each(entities, &process_pet_entity(persona_name, &1))

      "room" ->
        Enum.each(entities, &process_room_entity(persona_name, &1))

      "device" ->
        Enum.each(entities, &process_device_entity(persona_name, &1))

      "place" ->
        Enum.each(entities, &process_place_entity(persona_name, &1))

      "task" ->
        Enum.each(entities, &process_task_entity(persona_name, &1))

      "event" ->
        Enum.each(entities, &process_event_entity(persona_name, &1))

      "preference" ->
        Enum.each(entities, &process_preference_entity(persona_name, &1))

      _ ->
        Logger.debug("Unknown entity type", %{type: type})
    end
  end

  defp process_person_entity(persona_name, entity) do
    name = entity["name"]
    properties = Map.get(entity, "properties", %{})
    confidence = Map.get(entity, "confidence", 0.5)

    # Only process high-confidence entities
    if confidence >= 0.7 do
      person_info =
        %{
          "name" => name,
          "type" => "person"
        }
        |> maybe_add_property(properties, "age")
        |> maybe_add_property(properties, "occupation")
        |> maybe_add_property(properties, "location")
        |> maybe_add_property(properties, "description")
        |> Map.put("confidence", confidence)

      KnowledgeStore.add_person(persona_name, name, person_info)
      Logger.info("Learned person entity", %{name: name, confidence: confidence})
    end
  end

  defp process_pet_entity(persona_name, entity) do
    name = entity["name"]
    properties = Map.get(entity, "properties", %{})
    confidence = Map.get(entity, "confidence", 0.5)

    if confidence >= 0.7 do
      pet_info =
        %{
          "name" => name,
          "type" => "pet"
        }
        |> maybe_add_property(properties, "species")
        |> maybe_add_property(properties, "breed")
        |> maybe_add_property(properties, "age")
        |> maybe_add_property(properties, "color")
        |> maybe_add_property(properties, "size")
        |> maybe_add_property(properties, "description")
        |> Map.put("confidence", confidence)

      KnowledgeStore.add_pet(persona_name, name, pet_info)
      Logger.info("Learned pet entity", %{name: name, confidence: confidence})
    end
  end

  defp process_room_entity(persona_name, entity) do
    name = entity["name"]
    properties = Map.get(entity, "properties", %{})
    confidence = Map.get(entity, "confidence", 0.5)

    if confidence >= 0.7 do
      room_info =
        %{
          "name" => name,
          "type" => "room"
        }
        |> maybe_add_property(properties, "type")
        |> maybe_add_property(properties, "size")
        |> maybe_add_property(properties, "color")
        |> maybe_add_property(properties, "purpose")
        |> maybe_add_property(properties, "location")
        |> maybe_add_property(properties, "description")
        |> Map.put("confidence", confidence)

      KnowledgeStore.add_room(persona_name, name, room_info)
      Logger.info("Learned room entity", %{name: name, confidence: confidence})
    end
  end

  defp process_device_entity(persona_name, entity) do
    name = entity["name"]
    properties = Map.get(entity, "properties", %{})
    confidence = Map.get(entity, "confidence", 0.5)

    if confidence >= 0.7 do
      device_info =
        %{
          "name" => name,
          "type" => "device"
        }
        |> maybe_add_property(properties, "type")
        |> maybe_add_property(properties, "brand")
        |> maybe_add_property(properties, "model")
        |> maybe_add_property(properties, "location")
        |> maybe_add_property(properties, "status")
        |> maybe_add_property(properties, "description")
        |> Map.put("confidence", confidence)

      KnowledgeStore.add_device(persona_name, name, device_info)
      Logger.info("Learned device entity", %{name: name, confidence: confidence})
    end
  end

  defp process_place_entity(persona_name, entity) do
    name = entity["name"]
    properties = Map.get(entity, "properties", %{})
    confidence = Map.get(entity, "confidence", 0.5)

    if confidence >= 0.7 do
      place_info =
        %{
          "name" => name,
          "type" => "place"
        }
        |> maybe_add_property(properties, "type")
        |> maybe_add_property(properties, "location")
        |> maybe_add_property(properties, "description")
        |> Map.put("confidence", confidence)

      # Store places in a new knowledge category
      KnowledgeStore.add_place(persona_name, name, place_info)
      Logger.info("Learned place entity", %{name: name, confidence: confidence})
    end
  end

  defp process_task_entity(persona_name, entity) do
    name = entity["name"]
    properties = Map.get(entity, "properties", %{})
    confidence = Map.get(entity, "confidence", 0.5)

    if confidence >= 0.7 do
      task_info =
        %{
          "name" => name,
          "type" => "task"
        }
        |> maybe_add_property(properties, "status")
        |> maybe_add_property(properties, "priority")
        |> maybe_add_property(properties, "due_date")
        |> maybe_add_property(properties, "description")
        |> Map.put("confidence", confidence)

      # Store tasks in a new knowledge category
      KnowledgeStore.add_task(persona_name, name, task_info)
      Logger.info("Learned task entity", %{name: name, confidence: confidence})
    end
  end

  defp process_event_entity(persona_name, entity) do
    name = entity["name"]
    properties = Map.get(entity, "properties", %{})
    confidence = Map.get(entity, "confidence", 0.5)

    if confidence >= 0.7 do
      event_info =
        %{
          "name" => name,
          "type" => "event"
        }
        |> maybe_add_property(properties, "date")
        |> maybe_add_property(properties, "time")
        |> maybe_add_property(properties, "location")
        |> maybe_add_property(properties, "description")
        |> Map.put("confidence", confidence)

      # Store events in a new knowledge category
      KnowledgeStore.add_event(persona_name, name, event_info)
      Logger.info("Learned event entity", %{name: name, confidence: confidence})
    end
  end

  defp process_preference_entity(persona_name, entity) do
    name = entity["name"]
    properties = Map.get(entity, "properties", %{})
    confidence = Map.get(entity, "confidence", 0.5)

    if confidence >= 0.7 do
      preference_info =
        %{
          "name" => name,
          "type" => "preference"
        }
        |> maybe_add_property(properties, "category")
        |> maybe_add_property(properties, "value")
        |> maybe_add_property(properties, "description")
        |> Map.put("confidence", confidence)

      # Store preferences in a new knowledge category
      KnowledgeStore.add_preference(persona_name, name, preference_info)
      Logger.info("Learned preference entity", %{name: name, confidence: confidence})
    end
  end

  defp process_relationships(persona_name, relationships) do
    Enum.each(relationships, fn rel ->
      confidence = Map.get(rel, "confidence", 0.5)

      if confidence >= 0.7 do
        subject = rel["subject"]
        relation = rel["relation"]
        object = rel["object"]

        # Store relationship in knowledge store
        KnowledgeStore.add_relationship(persona_name, subject, relation, object, confidence)

        Logger.info("Learned relationship", %{
          subject: subject,
          relation: relation,
          object: object,
          confidence: confidence
        })
      end
    end)
  end

  defp process_facts(persona_name, facts) do
    Enum.each(facts, fn fact ->
      normalized = normalize_fact(fact)
      confidence = Map.get(normalized, "confidence", 0.5)

      if confidence >= 0.7 do
        fact_text = Map.get(normalized, "fact", "")
        entity = Map.get(normalized, "entity", "general")

        # Store fact in knowledge store (persona-specific)
        KnowledgeStore.add_fact(persona_name, entity, fact_text, confidence)

        # Also add to fact database if it's a general knowledge fact (not user-specific)
        if is_general_knowledge_fact?(entity, fact_text) do
          # Verify fact before adding
          case Integration.verify_fact(entity, fact_text) do
            {:verified, verified_confidence} ->
              # Use the higher confidence
              final_confidence = max(confidence, verified_confidence)

              case Integration.add_fact(entity, fact_text,
                     category: "learned",
                     verification_source: "conversation_learning",
                     confidence: final_confidence,
                     register_with_jtms: true,
                     create_belief: true
                   ) do
                {:ok, fact_id, _fact} ->
                  Logger.info("Added learned fact to database", %{
                    fact_id: fact_id,
                    entity: entity,
                    confidence: final_confidence
                  })

                error ->
                  Logger.debug("Failed to add fact to database", %{error: error})
              end

            {:contradicted, conflicting_beliefs} ->
              Logger.warning("Learned fact contradicts existing beliefs", %{
                entity: entity,
                fact: fact_text,
                conflicts: length(conflicting_beliefs)
              })

            {:uncertain, reason} ->
              Logger.debug("Cannot verify learned fact", %{
                entity: entity,
                fact: fact_text,
                reason: reason
              })
          end
        end

        Logger.info("Learned fact", %{
          entity: entity,
          fact: fact_text,
          confidence: confidence
        })
      end
    end)
  end

  defp is_general_knowledge_fact?(entity, fact_text) do
    # Determine if this is general knowledge vs user-specific
    # User-specific entities: person names, pets, rooms, devices, preferences
    user_specific_patterns = [
      "my ",
      "i ",
      "me ",
      "mine ",
      "our ",
      "we "
    ]

    entity_lower = String.downcase(entity)
    fact_lower = String.downcase(fact_text)

    # Check if it's clearly user-specific
    is_user_specific =
      Enum.any?(user_specific_patterns, &String.contains?(fact_lower, &1)) or
        String.contains?(entity_lower, "person") or
        String.contains?(entity_lower, "pet") or
        String.contains?(entity_lower, "room") or
        String.contains?(entity_lower, "device") or
        String.contains?(entity_lower, "preference")

    not is_user_specific
  end

  defp normalize_fact(fact) do
    cond do
      is_map(fact) ->
        fact_text = Map.get(fact, "fact") || Map.get(fact, "text") || to_string(fact)

        %{
          "fact" => fact_text,
          "entity" => Map.get(fact, "entity") || infer_entity_from_text(fact_text),
          "confidence" => Map.get(fact, "confidence", 0.8)
        }

      is_binary(fact) ->
        %{
          "fact" => fact,
          "entity" => infer_entity_from_text(fact),
          "confidence" => 0.8
        }

      true ->
        %{"fact" => to_string(fact), "entity" => "general", "confidence" => 0.5}
    end
  end

  defp infer_entity_from_text(text) when is_binary(text) do
    # Simple heuristic: take the phrase before " is " or the first token
    entity_candidate =
      case String.split(text, " is ", parts: 2) do
        [lhs, _rhs] -> String.trim(lhs)
        _ -> text
      end

    candidate =
      entity_candidate
      |> String.split()
      |> List.first()
      |> case do
        nil -> "general"
        "" -> "general"
        word -> word
      end

    candidate
  end

  defp store_context_memory(persona_name, original_input, extracted_data) do
    # Store the learning context in memory
    entities_count = length(Map.get(extracted_data, "entities", []))
    relationships_count = length(Map.get(extracted_data, "relationships", []))
    facts_count = length(Map.get(extracted_data, "facts", []))

    memory_text =
      "Learned #{entities_count} entities, #{relationships_count} relationships, #{facts_count} facts from: #{original_input}"

    # Avoid duplicate context entries by checking last memory entry for same text
    last_entries = ChatBot.MemoryStore.load_all(persona_name) |> Enum.reverse() |> Enum.take(1)

    case last_entries do
      [%{"content" => ^memory_text}] ->
        :ok

      _ ->
        MemoryStore.append_thought(persona_name, "system", memory_text, ["learning", "context"])
    end
  end

  defp store_general_memory(persona_name, input) do
    # Store as general memory when no specific entities are found
    MemoryStore.append_thought(persona_name, "user", input, ["general", "conversation"])
    Logger.info("Stored general memory", %{persona_name: persona_name, input: input})
  end

  # Helper functions

  defp maybe_add_property(map, properties, key) do
    case Map.get(properties, key) do
      nil -> map
      value -> Map.put(map, key, value)
    end
  end
end
