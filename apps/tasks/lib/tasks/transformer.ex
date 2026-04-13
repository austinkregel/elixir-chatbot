defmodule Tasks.Transformer do
  @moduledoc "Transforms domain-specific NLP task data into chatbot training format.\n\nProvides category-specific converters that transform task instances into:\n- Intent training samples (text + tokens + POS tags + entities + intent)\n- Knowledge facts (for semantic memory)\n- Entity candidates (for gazetteer)\n\n## Supported Categories\n\n- Question Answering: Extracts question-answer pairs for factual responses\n- Commonsense: Extracts reasoning facts and rules\n- Sentiment Analysis: Training for emotion detection\n- Paraphrasing: Multiple phrasings for intent augmentation\n- Text Categorization: Classification training data\n"

  alias Tasks.Analyzer
  alias Brain.ML
  require Logger

  alias ML.{Tokenizer, POSTagger, EntityExtractor}

  @type training_sample :: %{
          text: String.t(),
          tokens: [String.t()],
          pos_tags: [String.t()],
          entities: [map()],
          id: String.t(),
          intent: String.t(),
          metadata: map()
        }

  @type knowledge_fact :: %{
          subject: String.t(),
          predicate: String.t(),
          object: String.t(),
          source: String.t(),
          confidence: float()
        }

  @type transform_result :: %{
          training_samples: [training_sample()],
          knowledge_facts: [knowledge_fact()],
          entity_candidates: [map()]
        }

  @doc "Transforms a task file into training data.\n\nAutomatically selects the appropriate converter based on task category.\n\n## Options\n  - `:max_instances` - Maximum instances to process (default: 1000)\n  - `:include_examples` - Include positive/negative examples (default: true)\n  - `:extract_entities` - Run entity extraction on text (default: true)\n"
  @spec transform_task(String.t(), keyword()) :: {:ok, transform_result()} | {:error, term()}
  def transform_task(file_path, opts \\ []) do
    max_instances = Keyword.get(opts, :max_instances, 1000)
    include_examples = Keyword.get(opts, :include_examples, true)

    case Analyzer.parse_task_file(file_path) do
      nil ->
        {:error, :parse_failed}

      metadata ->
        case Analyzer.load_instances(file_path,
               max_instances: max_instances,
               include_examples: include_examples
             ) do
          {:ok, instances} ->
            {:ok, definition} = Analyzer.get_definition(file_path)
            category = List.first(metadata.categories) || "Unknown"
            converter = select_converter(category)
            result = apply_converter(converter, instances, metadata, definition, opts)
            {:ok, result}

          {:error, reason} ->
            {:error, reason}
        end
    end
  end

  @doc "Transforms multiple task files.\n\n## Options\n  - All options from `transform_task/2`\n  - `:progress_callback` - Function called with progress updates\n"
  @spec transform_tasks([String.t()], keyword()) :: {:ok, transform_result()} | {:error, term()}
  def transform_tasks(file_paths, opts \\ []) do
    progress_callback = Keyword.get(opts, :progress_callback)
    total = length(file_paths)

    results =
      file_paths
      |> Enum.with_index(1)
      |> Enum.map(fn {file_path, idx} ->
        if progress_callback do
          progress_callback.(%{current: idx, total: total, file: file_path})
        end

        case transform_task(file_path, opts) do
          {:ok, result} -> result
          {:error, _} -> empty_result()
        end
      end)

    merged = merge_results(results)
    {:ok, merged}
  end

  @doc "Converts a text to a training sample with tokenization, POS tagging, and entity extraction.\n"
  @spec text_to_training_sample(String.t(), String.t(), String.t(), keyword()) ::
          training_sample()
  def text_to_training_sample(text, intent, id, opts \\ []) do
    extract_entities = Keyword.get(opts, :extract_entities, true)
    metadata = Keyword.get(opts, :metadata, %{})
    tokens = Tokenizer.tokenize_words(text)

    pos_tags =
      case POSTagger.load_model() do
        {:ok, model} -> POSTagger.predict_tags(tokens, model)
        {:error, _} -> Enum.map(tokens, fn _ -> "UNKNOWN" end)
      end

    entities =
      if extract_entities do
        case EntityExtractor.extract_entities(text, skip_disambiguation: true) do
          entities when is_list(entities) ->
            Enum.map(entities, fn entity ->
              %{
                "text" => Map.get(entity, :value, ""),
                "type" => Map.get(entity, :entity, "unknown"),
                "start" => Map.get(entity, :start, 0),
                "end" => Map.get(entity, :end, 0)
              }
            end)

          _ ->
            []
        end
      else
        []
      end

    %{
      text: text,
      tokens: tokens,
      pos_tags: pos_tags,
      entities: entities,
      id: id,
      intent: intent,
      metadata: metadata
    }
  end

  @doc "Converts Question Answering task instances.\n\nInput questions become training utterances for a \"factual_question\" intent.\nAnswers are stored as knowledge facts for retrieval.\n"
  def convert_qa(instances, metadata, _definition, opts) do
    task_id = metadata.task_id

    {samples, facts, entities} =
      instances
      |> Enum.with_index()
      |> Enum.reduce({[], [], []}, fn {instance, idx}, {s_acc, f_acc, e_acc} ->
        input = get_input(instance)
        outputs = get_outputs(instance)
        instance_id = get_instance_id(instance, "#{task_id}-#{idx}")

        sample =
          text_to_training_sample(
            input,
            "factual_question",
            instance_id,
            opts
          )

        facts =
          Enum.map(outputs, fn output ->
            %{
              subject: normalize_text(input),
              predicate: "has_answer",
              object: normalize_text(output),
              source: task_id,
              confidence: 0.9
            }
          end)

        answer_entities =
          outputs
          |> Enum.flat_map(fn output ->
            extract_entity_candidates(output, task_id)
          end)

        {[sample | s_acc], facts ++ f_acc, answer_entities ++ e_acc}
      end)

    %{
      training_samples: Enum.reverse(samples),
      knowledge_facts: facts,
      entity_candidates: entities
    }
  end

  @doc "Converts Commonsense Reasoning task instances.\n\nExtracts factual statements and reasoning patterns.\n"
  def convert_commonsense(instances, metadata, _definition, opts) do
    task_id = metadata.task_id

    {samples, facts, entities} =
      instances
      |> Enum.with_index()
      |> Enum.reduce({[], [], []}, fn {instance, idx}, {s_acc, f_acc, e_acc} ->
        input = get_input(instance)
        outputs = get_outputs(instance)
        explanation = Map.get(instance, "explanation", "")
        instance_id = get_instance_id(instance, "#{task_id}-#{idx}")

        new_facts =
          if explanation != "" do
            [
              %{
                subject: normalize_text(input),
                predicate: "implies",
                object: normalize_text(explanation),
                source: task_id,
                confidence: 0.85
              }
            ]
          else
            outputs
            |> Enum.map(fn output ->
              %{
                subject: normalize_text(input),
                predicate: "commonsense_answer",
                object: normalize_text(output),
                source: task_id,
                confidence: 0.8
              }
            end)
          end

        sample =
          text_to_training_sample(
            input,
            "commonsense_query",
            instance_id,
            opts
          )

        {[sample | s_acc], new_facts ++ f_acc, e_acc}
      end)

    %{
      training_samples: Enum.reverse(samples),
      knowledge_facts: facts,
      entity_candidates: entities
    }
  end

  @doc "Converts Sentiment Analysis task instances.\n\nCreates training data for sentiment/emotion detection.\n"
  def convert_sentiment(instances, metadata, _definition, opts) do
    task_id = metadata.task_id

    samples =
      instances
      |> Enum.with_index()
      |> Enum.map(fn {instance, idx} ->
        input = get_input(instance)
        outputs = get_outputs(instance)
        instance_id = get_instance_id(instance, "#{task_id}-#{idx}")
        sentiment = List.first(outputs) || "neutral"
        intent = "sentiment.#{String.downcase(sentiment)}"

        text_to_training_sample(
          input,
          intent,
          instance_id,
          Keyword.put(opts, :metadata, %{sentiment: sentiment})
        )
      end)

    %{
      training_samples: samples,
      knowledge_facts: [],
      entity_candidates: []
    }
  end

  @doc "Converts Paraphrasing task instances.\n\nCreates multiple phrasings that can augment existing intents.\n"
  def convert_paraphrase(instances, metadata, _definition, opts) do
    task_id = metadata.task_id

    samples =
      instances
      |> Enum.with_index()
      |> Enum.flat_map(fn {instance, idx} ->
        input = get_input(instance)
        outputs = get_outputs(instance)
        instance_id = get_instance_id(instance, "#{task_id}-#{idx}")

        original =
          text_to_training_sample(
            input,
            "paraphrase.original",
            "#{instance_id}-orig",
            opts
          )

        paraphrases =
          outputs
          |> Enum.with_index()
          |> Enum.map(fn {output, p_idx} ->
            text_to_training_sample(
              output,
              "paraphrase.variant",
              "#{instance_id}-p#{p_idx}",
              Keyword.put(opts, :metadata, %{original: input})
            )
          end)

        [original | paraphrases]
      end)

    %{
      training_samples: samples,
      knowledge_facts: [],
      entity_candidates: []
    }
  end

  @doc "Converts Text Categorization task instances.\n\nCreates intent classification training data.\n"
  def convert_categorization(instances, metadata, _definition, opts) do
    task_id = metadata.task_id

    samples =
      instances
      |> Enum.with_index()
      |> Enum.map(fn {instance, idx} ->
        input = get_input(instance)
        outputs = get_outputs(instance)
        instance_id = get_instance_id(instance, "#{task_id}-#{idx}")
        category = List.first(outputs) || "unknown"
        intent = "category.#{normalize_intent(category)}"

        text_to_training_sample(input, intent, instance_id, opts)
      end)

    %{
      training_samples: samples,
      knowledge_facts: [],
      entity_candidates: []
    }
  end

  @doc "Converts Story/Text Composition task instances.\n\nExtracts narrative patterns and story elements.\n"
  def convert_composition(instances, metadata, _definition, opts) do
    task_id = metadata.task_id

    {samples, facts, _} =
      instances
      |> Enum.with_index()
      |> Enum.reduce({[], [], []}, fn {instance, idx}, {s_acc, f_acc, e_acc} ->
        input = get_input(instance)
        outputs = get_outputs(instance)
        instance_id = get_instance_id(instance, "#{task_id}-#{idx}")

        story_facts =
          outputs
          |> Enum.map(fn output ->
            %{
              subject: "story_prompt",
              predicate: "generates",
              object: normalize_text(output),
              source: task_id,
              confidence: 0.7
            }
          end)

        sample =
          text_to_training_sample(
            input,
            "story_request",
            instance_id,
            opts
          )

        {[sample | s_acc], story_facts ++ f_acc, e_acc}
      end)

    %{
      training_samples: Enum.reverse(samples),
      knowledge_facts: facts,
      entity_candidates: []
    }
  end

  @doc "Generic converter for unsupported categories.\n\nCreates basic training samples without specialized processing.\n"
  def convert_generic(instances, metadata, _definition, opts) do
    task_id = metadata.task_id
    category = List.first(metadata.categories) || "unknown"
    intent = "domain.#{normalize_intent(category)}"

    samples =
      instances
      |> Enum.with_index()
      |> Enum.map(fn {instance, idx} ->
        input = get_input(instance)
        instance_id = get_instance_id(instance, "#{task_id}-#{idx}")

        text_to_training_sample(input, intent, instance_id, opts)
      end)

    %{
      training_samples: samples,
      knowledge_facts: [],
      entity_candidates: []
    }
  end

  defp select_converter(category) do
    case category do
      "Question Answering" -> :qa
      "Commonsense Classification" -> :commonsense
      "Sentiment Analysis" -> :sentiment
      "Paraphrasing" -> :paraphrase
      "Text Categorization" -> :categorization
      "Story Composition" -> :composition
      "Sentence Composition" -> :composition
      "Data to Text" -> :composition
      "Text Completion" -> :composition
      "Coherence Classification" -> :commonsense
      "Word Semantics" -> :commonsense
      "Explanation" -> :qa
      "Coreference Resolution" -> :commonsense
      _ -> :generic
    end
  end

  defp apply_converter(converter, instances, metadata, definition, opts) do
    case converter do
      :qa -> convert_qa(instances, metadata, definition, opts)
      :commonsense -> convert_commonsense(instances, metadata, definition, opts)
      :sentiment -> convert_sentiment(instances, metadata, definition, opts)
      :paraphrase -> convert_paraphrase(instances, metadata, definition, opts)
      :categorization -> convert_categorization(instances, metadata, definition, opts)
      :composition -> convert_composition(instances, metadata, definition, opts)
      :generic -> convert_generic(instances, metadata, definition, opts)
    end
  end

  defp get_input(instance) do
    Map.get(instance, "input", "")
  end

  defp get_outputs(instance) do
    case Map.get(instance, "output") do
      nil -> []
      list when is_list(list) -> list
      single -> [single]
    end
  end

  defp get_instance_id(instance, default) do
    Map.get(instance, "id", default)
  end

  defp normalize_text(text) when is_binary(text) do
    text
    |> String.trim()
    |> String.slice(0, 500)
  end

  defp normalize_text(_) do
    ""
  end

  defp normalize_intent(category) when is_binary(category) do
    category
    |> String.downcase()
    |> String.replace(~r/[^a-z0-9]+/, "_")
    |> String.trim("_")
  end

  defp normalize_intent(_) do
    "unknown"
  end

  defp extract_entity_candidates(text, source) when is_binary(text) do
    case EntityExtractor.extract_entities(text, skip_disambiguation: true) do
      entities when is_list(entities) ->
        Enum.map(entities, fn entity ->
          %{
            value: Map.get(entity, :value, ""),
            inferred_type: Map.get(entity, :entity, "unknown"),
            confidence: Map.get(entity, :confidence, 0.8),
            source: source
          }
        end)

      _ ->
        []
    end
  end

  defp extract_entity_candidates(_, _) do
    []
  end

  defp empty_result do
    %{
      training_samples: [],
      knowledge_facts: [],
      entity_candidates: []
    }
  end

  defp merge_results(results) do
    Enum.reduce(results, empty_result(), fn result, acc ->
      %{
        training_samples: acc.training_samples ++ result.training_samples,
        knowledge_facts: acc.knowledge_facts ++ result.knowledge_facts,
        entity_candidates: acc.entity_candidates ++ result.entity_candidates
      }
    end)
  end
end