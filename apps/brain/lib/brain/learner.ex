defmodule Brain.Learner do
  @moduledoc "NLP-based learner module for extracting facts from user inputs.\nUses classical NLP entity recognition and relationship extraction to build\ndynamic, adaptable knowledge about people, pets, rooms, devices, places, tasks, etc.\n"

  alias Brain.ML.EntityExtractor
  alias Brain.ML.Tokenizer
  alias Brain.Analysis.SpeechActClassifier
  alias Brain.Analysis.DiscourseAnalyzer
  require Logger
  alias Brain.KnowledgeStore
  alias Brain.MemoryStore
  alias Brain.FactDatabase.Integration
  alias Brain.Knowledge.ReviewQueue

  @doc "Learn from user input using classical NLP entity extraction.\nExtracts entities and stores them in the knowledge store.\n\nOptions:\n- `:discourse` - Discourse analysis result for entity disambiguation\n- `:speech_act` - Speech act classification result for entity disambiguation\n- `:world_id` - World ID for world-scoped type inference (required for disambiguation)\n"
  def learn_from_input(persona_name, input, opts \\ []) do
    Logger.debug("Learner.learn_from_input called", %{persona_name: persona_name, input: input})
    discourse = Keyword.get(opts, :discourse)
    speech_act = Keyword.get(opts, :speech_act)
    world_id = Keyword.get(opts, :world_id)
    skip_disambiguation = is_nil(world_id)

    entity_opts =
      if discourse || speech_act do
        opts ++ [skip_disambiguation: skip_disambiguation]
      else
        discourse_result =
          try do
            DiscourseAnalyzer.analyze(input, [])
          rescue
            _ -> nil
          catch
            _ -> nil
          end

        speech_act_result =
          try do
            SpeechActClassifier.classify(input)
          rescue
            _ -> nil
          catch
            _ -> nil
          end

        opts ++
          [
            discourse: discourse_result,
            speech_act: speech_act_result,
            skip_disambiguation: skip_disambiguation
          ]
      end

    entities = EntityExtractor.extract_entities(input, entity_opts)

    if entities != [] do
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
      store_general_memory(persona_name, input)
      {:ok, %{"type" => "general_memory", "input" => input}}
    end
  end

  @doc "Learn from pre-extracted classical NLP entities.\nAccepts entities from the classical NLP pipeline and stores them directly.\n"
  def learn_from_classical_extraction(persona_name, entities, input) do
    Logger.debug("Learner.learn_from_classical_extraction called", %{
      persona_name: persona_name,
      entities_count: length(entities),
      input: input
    })

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

  @doc "Learn from a conversation turn with full analysis context.\n\nThis is the primary entry point for conversational learning. When a user makes\nan assertive statement (e.g., \"Paris is the capital of France\"), we extract\nthe claim as a learnable fact.\n\n## Parameters\n  - persona_name: The bot persona name\n  - input: The user's input text\n  - analysis: The full analysis result from the Pipeline\n  - opts: Options including :user_id for user-specific learning\n\n## Returns\n  - {:ok, %{entities: [...], facts: [...], learned: boolean}}\n"
  def learn_from_conversation(persona_name, input, analysis, opts \\ []) do
    Logger.debug("Learner.learn_from_conversation called", %{
      persona_name: persona_name,
      input: input,
      has_analysis: analysis != nil
    })

    entities = Map.get(analysis, :entities, [])
    speech_act = Map.get(analysis, :speech_act, %{})

    extracted_entities =
      Enum.map(entities, fn entity ->
        %{
          "name" => get_entity_field(entity, [:value, "value"]),
          "type" =>
            get_entity_field(entity, [
              :entity,
              "entity",
              :entity_type,
              "entity_type",
              :type,
              "type"
            ]),
          "properties" => %{},
          "confidence" => get_entity_field(entity, [:confidence, "confidence"]) || 0.9
        }
      end)

    extracted_facts = extract_facts_from_statement(input, entities, speech_act, opts)

    extracted_data = %{
      "entities" => extracted_entities,
      "relationships" => [],
      "facts" => extracted_facts,
      "context" => input
    }

    process_extracted_data(persona_name, extracted_data, input)

    {:ok,
     %{
       entities: extracted_entities,
       facts: extracted_facts,
       learned: extracted_facts != []
     }}
  end

  defp extract_facts_from_statement(input, entities, speech_act, _opts) do
    category = Map.get(speech_act, :category)
    sub_type = Map.get(speech_act, :sub_type)
    is_assertive = category == :assertive and sub_type == :statement
    is_self_referential = is_self_referential_statement?(input)
    word_count = input |> Tokenizer.split_words() |> length()
    valid_length = word_count >= 3 and word_count <= 30

    if is_assertive and not is_self_referential and valid_length and entities != [] do
      primary_entity =
        entities
        |> Enum.max_by(
          fn e ->
            get_entity_field(e, [:confidence, "confidence"]) || 0.5
          end,
          fn -> nil end
        )

      if primary_entity do
        entity_value = get_entity_field(primary_entity, [:value, "value"]) || ""

        entity_type =
          get_entity_field(primary_entity, [
            :entity,
            "entity",
            :entity_type,
            "entity_type",
            :type,
            "type"
          ]) || "unknown"

        fact_text = String.trim(input)
        entity_confidence = get_entity_field(primary_entity, [:confidence, "confidence"]) || 0.5
        fact_confidence = min(entity_confidence * 0.9, 0.85)

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

  @self_ref_verbs MapSet.new(~w(am was have had will would like love hate work live think feel want need))
  @self_ref_nouns MapSet.new(~w(name age job work location home favorite preference))

  defp is_self_referential_statement?(input) do
    tokens = Brain.ML.Tokenizer.tokenize_normalized(input)
    first = Enum.at(tokens, 0)
    second = Enum.at(tokens, 1)
    third = Enum.at(tokens, 2)

    cond do
      # "i am/was/have/..." or "i'm/i've" (tokenizer expands contractions)
      first == "i" and MapSet.member?(@self_ref_verbs, second || "") -> true
      # "my name/age/job/..."
      first == "my" and MapSet.member?(@self_ref_nouns, second || "") -> true
      # "my name is ..."
      first == "my" and second == "name" and third == "is" -> true
      # "call me ..."
      first == "call" and second == "me" -> true
      # "i am from ...", "i live in ..."
      first == "i" and second == "am" and third == "from" -> true
      first == "i" and second == "live" and third == "in" -> true
      true -> false
    end
  end

  defp get_entity_field(entity, keys) when is_list(keys) do
    Enum.find_value(keys, fn key ->
      case entity do
        %{^key => value} when not is_nil(value) -> value
        _ -> nil
      end
    end)
  end

  defp process_extracted_data(persona_name, data, original_input) do
    entities = Map.get(data, "entities", [])
    relationships = Map.get(data, "relationships", [])
    facts = Map.get(data, "facts", [])
    entities_by_type = Enum.group_by(entities, & &1["type"])

    Enum.each(entities_by_type, fn {type, entities_of_type} ->
      process_entities_of_type(persona_name, type, entities_of_type)
    end)

    process_relationships(persona_name, relationships)
    process_facts(persona_name, facts)
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

    if confidence >= 0.9 and not gazetteer_conflicts_with_type?(name, "room") do
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
    else
      if confidence >= 0.7 and confidence < 0.9 do
        Logger.debug("Room entity below auto-store threshold",
          name: name, confidence: confidence
        )
      end

      if gazetteer_conflicts_with_type?(name, "room") do
        Logger.debug("Room entity conflicts with existing gazetteer entry",
          name: name
        )
      end
    end
  end

  @location_types ~w(location city gpe geo place address country state)

  defp gazetteer_conflicts_with_type?(name, proposed_type) when is_binary(name) do
    conflicting_types =
      case proposed_type do
        "room" -> @location_types
        "device" -> @location_types
        _ -> []
      end

    if conflicting_types == [] do
      false
    else
      case Brain.ML.Gazetteer.lookup_all_types(name) do
        types when is_list(types) and types != [] ->
          Enum.any?(types, fn entry ->
            existing_type = Map.get(entry, :entity_type, "")
            existing_type in conflicting_types
          end)

        _ ->
          false
      end
    end
  rescue
    _ -> false
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
        KnowledgeStore.add_fact(persona_name, entity, fact_text, confidence)

        if is_general_knowledge_fact?(entity, fact_text) do
          case Integration.verify_fact(entity, fact_text) do
            {:verified, verified_confidence} ->
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

              new_fact = %{
                entity: entity,
                entity_type: Map.get(normalized, "entity_type"),
                fact: fact_text,
                confidence: confidence,
                source: "conversation_learning"
              }

              Enum.each(conflicting_beliefs, fn belief ->
                ReviewQueue.add_contradiction(new_fact, belief)
              end)

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
    first_person_pronouns = MapSet.new(~w(my i me mine our we))
    fact_words = fact_text |> Tokenizer.tokenize_normalized(min_length: 1) |> MapSet.new()

    has_first_person = not MapSet.disjoint?(fact_words, first_person_pronouns)

    classifier_input = "#{String.downcase(entity)} #{String.downcase(fact_text)}"

    classifier_says_user_specific =
      case Brain.ML.MicroClassifiers.classify(:user_fact_type, classifier_input) do
        {:ok, "user_specific", score} when score > 0.3 -> true
        _ -> false
      end

    not (has_first_person or classifier_says_user_specific)
  end

  defp normalize_fact(fact) do
    alias Brain.FactDatabase.Fact

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
    entities_count = length(Map.get(extracted_data, "entities", []))
    relationships_count = length(Map.get(extracted_data, "relationships", []))
    facts_count = length(Map.get(extracted_data, "facts", []))

    memory_text =
      "Learned #{entities_count} entities, #{relationships_count} relationships, #{facts_count} facts from: #{original_input}"

    last_entries = Brain.MemoryStore.load_all(persona_name) |> Enum.reverse() |> Enum.take(1)

    case last_entries do
      [%{"content" => ^memory_text}] ->
        :ok

      _ ->
        MemoryStore.append_thought(persona_name, "system", memory_text, ["learning", "context"])
    end
  end

  defp store_general_memory(persona_name, input) do
    MemoryStore.append_thought(persona_name, "user", input, ["general", "conversation"])
    Logger.info("Stored general memory", %{persona_name: persona_name, input: input})
  end

  defp maybe_add_property(map, properties, key) do
    case Map.get(properties, key) do
      nil -> map
      value -> Map.put(map, key, value)
    end
  end
end