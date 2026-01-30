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

  Options:
  - `:discourse` - Discourse analysis result for entity disambiguation
  - `:speech_act` - Speech act classification result for entity disambiguation
  - `:world_id` - World ID for world-scoped type inference (required for disambiguation)
  """
  def learn_from_input(persona_name, input, opts \\ []) do
    Logger.debug("Learner.learn_from_input called", %{persona_name: persona_name, input: input})

    # Extract discourse and speech_act context if not provided
    # This ensures entities are disambiguated correctly for learning
    discourse = Keyword.get(opts, :discourse)
    speech_act = Keyword.get(opts, :speech_act)
    world_id = Keyword.get(opts, :world_id)

    # If no world_id provided, skip disambiguation to avoid errors
    # Entity extraction will still work, just without type inference
    skip_disambiguation = is_nil(world_id)

    entity_opts =
      if discourse || speech_act do
        opts ++ [skip_disambiguation: skip_disambiguation]
      else
        # Try to extract context for better disambiguation
        discourse_result =
          try do
            ChatBot.Analysis.DiscourseAnalyzer.analyze(input, [])
          rescue
            _ -> nil
          catch
            _ -> nil
          end

        speech_act_result =
          try do
            ChatBot.Analysis.SpeechActClassifier.classify(input)
          rescue
            _ -> nil
          catch
            _ -> nil
          end

        opts ++ [discourse: discourse_result, speech_act: speech_act_result, skip_disambiguation: skip_disambiguation]
      end

    # Use classical NLP pipeline to extract entities with disambiguation context
    entities = ChatBot.ML.EntityExtractor.extract_entities(input, entity_opts)

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

  @doc """
  Learn from a conversation turn with full analysis context.

  This is the primary entry point for conversational learning. When a user makes
  an assertive statement (e.g., "Paris is the capital of France"), we extract
  the claim as a learnable fact.

  ## Parameters
    - persona_name: The bot persona name
    - input: The user's input text
    - analysis: The full analysis result from the Pipeline
    - opts: Options including :user_id for user-specific learning

  ## Returns
    - {:ok, %{entities: [...], facts: [...], learned: boolean}}
  """
  def learn_from_conversation(persona_name, input, analysis, opts \\ []) do
    Logger.debug("Learner.learn_from_conversation called", %{
      persona_name: persona_name,
      input: input,
      has_analysis: analysis != nil
    })

    entities = Map.get(analysis, :entities, [])
    speech_act = Map.get(analysis, :speech_act, %{})

    # Convert entities to extracted format
    extracted_entities =
      Enum.map(entities, fn entity ->
        %{
          "name" => get_entity_field(entity, [:value, "value"]),
          "type" => get_entity_field(entity, [:entity, "entity", :entity_type, "entity_type", :type, "type"]),
          "properties" => %{},
          "confidence" => get_entity_field(entity, [:confidence, "confidence"]) || 0.9
        }
      end)

    # Extract facts from assertive statements
    extracted_facts = extract_facts_from_statement(input, entities, speech_act, opts)

    extracted_data = %{
      "entities" => extracted_entities,
      "relationships" => [],
      "facts" => extracted_facts,
      "context" => input
    }

    # Process the extracted data (entities, relationships, facts)
    process_extracted_data(persona_name, extracted_data, input)

    {:ok, %{
      entities: extracted_entities,
      facts: extracted_facts,
      learned: length(extracted_facts) > 0
    }}
  end

  # Extract factual claims from assertive statements
  defp extract_facts_from_statement(input, entities, speech_act, _opts) do
    category = Map.get(speech_act, :category)
    sub_type = Map.get(speech_act, :sub_type)

    # Only learn from assertive statements (not questions, commands, etc.)
    is_assertive = category == :assertive and sub_type == :statement

    # Skip self-referential statements (handled separately by Brain)
    is_self_referential = is_self_referential_statement?(input)

    # Skip very short or very long statements
    word_count = input |> String.split(~r/\s+/) |> length()
    valid_length = word_count >= 3 and word_count <= 30

    if is_assertive and not is_self_referential and valid_length and length(entities) > 0 do
      # Get the primary entity (highest confidence or first)
      primary_entity =
        entities
        |> Enum.max_by(fn e ->
          get_entity_field(e, [:confidence, "confidence"]) || 0.5
        end, fn -> nil end)

      if primary_entity do
        entity_value = get_entity_field(primary_entity, [:value, "value"]) || ""
        entity_type = get_entity_field(primary_entity, [:entity, "entity", :entity_type, "entity_type", :type, "type"]) || "unknown"

        # Clean the input as the fact text
        fact_text = String.trim(input)

        # Calculate confidence based on entity confidence and statement clarity
        entity_confidence = get_entity_field(primary_entity, [:confidence, "confidence"]) || 0.5
        fact_confidence = min(entity_confidence * 0.9, 0.85)  # Cap at 0.85 for learned facts

        Logger.debug("Extracted potential fact from conversation", %{
          entity: entity_value,
          entity_type: entity_type,
          fact: String.slice(fact_text, 0, 50),
          confidence: fact_confidence
        })

        [
          %{
            "entity" => entity_value,
            "entity_type" => entity_type,
            "fact" => fact_text,
            "confidence" => fact_confidence,
            "source" => "conversation"
          }
        ]
      else
        []
      end
    else
      []
    end
  end

  # Check if a statement is self-referential (about the user)
  defp is_self_referential_statement?(input) do
    lower = String.downcase(input)

    # Patterns that indicate the user is talking about themselves
    self_patterns = [
      ~r/^i\s+(am|was|have|had|will|would|like|love|hate|work|live|think|feel|want|need)\b/,
      ~r/^my\s+(name|age|job|work|location|home|favorite|preference)\b/,
      ~r/^i'm\b/,
      ~r/^i've\b/,
      ~r/\bmy\s+name\s+is\b/,
      ~r/\bi\s+am\s+from\b/,
      ~r/\bi\s+live\s+in\b/,
      ~r/\bcall\s+me\b/
    ]

    Enum.any?(self_patterns, &Regex.match?(&1, lower))
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
    alias ChatBot.FactDatabase.Fact

    cond do
      is_map(fact) ->
        fact_text = Map.get(fact, "fact") || Map.get(fact, "text") || to_string(fact)
        entity = Map.get(fact, "entity") || infer_entity_from_text(fact_text)
        category = Map.get(fact, "category", "learned")
        entity_type = Map.get(fact, "entity_type") || Fact.infer_entity_type(entity, category)

        %{
          "fact" => fact_text,
          "entity" => entity,
          "entity_type" => entity_type,
          "category" => category,
          "confidence" => Map.get(fact, "confidence", 0.8)
        }

      is_binary(fact) ->
        entity = infer_entity_from_text(fact)
        entity_type = Fact.infer_entity_type(entity, "learned")

        %{
          "fact" => fact,
          "entity" => entity,
          "entity_type" => entity_type,
          "category" => "learned",
          "confidence" => 0.8
        }

      true ->
        %{
          "fact" => to_string(fact),
          "entity" => "general",
          "entity_type" => "general",
          "category" => "learned",
          "confidence" => 0.5
        }
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
