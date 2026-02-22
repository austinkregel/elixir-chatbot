defmodule Mix.Tasks.AugmentTrainingData do
  @shortdoc "Augment sparse intents in gold standard to meet minimum example threshold"
  @moduledoc """
  Generates additional training examples for intents that have fewer
  than the minimum threshold of examples in the gold standard.

  Uses token-level transformations (word dropout, token swap, slot
  substitution) to create realistic variations of existing examples.

  ## Usage

      mix augment_training_data                  # Augment to 10 per intent
      mix augment_training_data --min 15         # Custom threshold
      mix augment_training_data --preview        # Show what would be generated
      mix augment_training_data --task intent    # Only augment intents (default)
      mix augment_training_data --task ner       # Only augment NER

  ## Techniques

  1. **Word dropout**: Remove a non-essential token (articles, prepositions)
  2. **Token swap**: Swap two adjacent content tokens
  3. **Prefix variation**: Add/remove politeness prefixes ("please", "can you")
  4. **Slot substitution**: Replace entity slots with alternatives from the same example set
  """

  use Mix.Task
  require Logger

  alias Brain.ML.{EvaluationStore, Tokenizer}

  @droppable_pos ~w(DET ADP PART SCONJ CCONJ PUNCT)

  @add_prefixes [
    {"please ", ""},
    {"can you ", ""},
    {"could you ", ""},
    {"hey ", ""},
    {"I want to ", ""},
    {"I'd like to ", ""}
  ]

  @removable_prefixes [
    "please ",
    "can you ",
    "could you ",
    "hey ",
    "i want to ",
    "i'd like to ",
    "i would like to "
  ]

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    {opts, _, _} =
      OptionParser.parse(args,
        strict: [min: :integer, preview: :boolean, task: :string]
      )

    min_threshold = Keyword.get(opts, :min, 10)
    preview? = Keyword.get(opts, :preview, false)
    task = Keyword.get(opts, :task, "intent")

    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("  TRAINING DATA AUGMENTATION")
    IO.puts("  Task: #{task}  |  Min threshold: #{min_threshold}")
    IO.puts(String.duplicate("=", 60))

    case task do
      "intent" -> augment_intents(min_threshold, preview?)
      "ner" -> augment_ner(min_threshold, preview?)
      _ -> IO.puts("\n  Unsupported task: #{task}. Use 'intent' or 'ner'.")
    end
  end

  defp augment_intents(min_threshold, preview?) do
    gold = EvaluationStore.load_gold_standard("intent")
    counts = Enum.frequencies_by(gold, fn ex -> ex["intent"] end)

    sparse_intents =
      counts
      |> Enum.filter(fn {_intent, count} -> count < min_threshold end)
      |> Enum.sort_by(fn {_intent, count} -> count end)

    if sparse_intents == [] do
      IO.puts("\n  All intents have >= #{min_threshold} examples. Nothing to augment.")
      return_early()
    end

    IO.puts("\n  #{length(sparse_intents)} intents below threshold")

    total_generated = 0
    all_new_examples = []

    {all_new_examples, total_generated} =
      Enum.reduce(sparse_intents, {all_new_examples, total_generated}, fn {intent, count}, {acc_examples, acc_count} ->
        needed = min_threshold - count
        existing = Enum.filter(gold, fn ex -> ex["intent"] == intent end)
        new_examples = generate_intent_variations(existing, needed, intent)

        IO.puts("    #{String.pad_trailing(intent, 45)} #{count} -> #{count + length(new_examples)} (+#{length(new_examples)})")

        {acc_examples ++ new_examples, acc_count + length(new_examples)}
      end)

    IO.puts("\n  Total new examples: #{total_generated}")

    if preview? do
      IO.puts("\n  Preview mode - no files written.")

      all_new_examples
      |> Enum.take(10)
      |> Enum.each(fn ex ->
        IO.puts("    [#{ex["intent"]}] #{ex["text"]}")
      end)

      if length(all_new_examples) > 10 do
        IO.puts("    ... and #{length(all_new_examples) - 10} more")
      end
    else
      updated = gold ++ all_new_examples
      content = Jason.encode!(updated, pretty: true)

      build_path = EvaluationStore.gold_standard_path("intent")
      File.mkdir_p!(Path.dirname(build_path))
      File.write!(build_path, content)
      IO.puts("\n  Written #{length(updated)} total examples to: #{build_path}")

      source_path = source_gold_standard_path("intent")
      File.mkdir_p!(Path.dirname(source_path))
      File.write!(source_path, content)
      IO.puts("  Also written to source: #{source_path}")
    end

    IO.puts("")
  end

  defp augment_ner(min_threshold, preview?) do
    gold = EvaluationStore.load_gold_standard("ner")

    type_counts =
      gold
      |> Enum.flat_map(fn ex ->
        entities = ex["entities"] || ex["expected"] || []
        Enum.map(entities, fn e -> e["type"] || e["entity_type"] || "unknown" end)
      end)
      |> Enum.frequencies()

    sparse_types =
      type_counts
      |> Enum.filter(fn {_type, count} -> count < min_threshold end)
      |> Enum.sort_by(fn {_type, count} -> count end)

    if sparse_types == [] do
      IO.puts("\n  All NER types have >= #{min_threshold} examples. Nothing to augment.")
      return_early()
    end

    IO.puts("\n  #{length(sparse_types)} entity types below threshold:")

    for {type, count} <- sparse_types do
      IO.puts("    #{String.pad_trailing(type, 25)} #{count}")
    end

    IO.puts("\n  NER augmentation requires manual entity annotations.")
    IO.puts("  Consider adding more annotated examples to:")
    IO.puts("    #{EvaluationStore.gold_standard_path("ner")}")

    if preview? do
      IO.puts("\n  Sparse NER types that need more data:")

      for {type, count} <- sparse_types do
        examples_with_type =
          Enum.filter(gold, fn ex ->
            entities = ex["entities"] || ex["expected"] || []
            Enum.any?(entities, fn e -> (e["type"] || e["entity_type"]) == type end)
          end)

        IO.puts("\n    #{type} (#{count} entities, #{length(examples_with_type)} examples):")

        examples_with_type
        |> Enum.take(3)
        |> Enum.each(fn ex ->
          IO.puts("      \"#{String.slice(ex["text"] || "", 0, 50)}\"")
        end)
      end
    end

    IO.puts("")
  end

  defp generate_intent_variations(_existing, needed, _intent) when needed <= 0, do: []

  defp generate_intent_variations(existing, needed, _intent) do
    existing_texts = MapSet.new(existing, fn ex -> normalize(ex["text"] || "") end)

    augmented =
      Stream.repeatedly(fn -> augment_one(Enum.random(existing)) end)
      |> Stream.reject(fn ex -> MapSet.member?(existing_texts, normalize(ex["text"])) end)
      |> Stream.uniq_by(fn ex -> normalize(ex["text"]) end)
      |> Enum.take(needed)

    augmented
  end

  defp augment_one(example) do
    text = example["text"] || ""
    tokens = example["tokens"] || Tokenizer.tokenize_words(text)
    pos_tags = example["pos_tags"] || []

    strategies =
      if length(tokens) > 3 do
        [:word_dropout, :adjacent_swap, :prefix_add, :prefix_remove, :truncate_end]
      else
        [:prefix_add, :prefix_remove, :adjacent_swap]
      end

    strategy = Enum.random(strategies)

    new_text =
      case strategy do
        :word_dropout -> word_dropout(tokens, pos_tags)
        :adjacent_swap -> adjacent_swap(tokens, pos_tags)
        :prefix_add -> add_prefix(text)
        :prefix_remove -> remove_prefix(text)
        :truncate_end -> truncate_end(tokens, pos_tags)
      end

    new_text = String.trim(new_text)
    new_text = if new_text == "" or new_text == text, do: add_prefix(text), else: new_text
    new_tokens = Tokenizer.tokenize_words(new_text)

    %{
      "intent" => example["intent"],
      "text" => new_text,
      "tokens" => new_tokens,
      "pos_tags" => [],
      "entities" => example["entities"] || [],
      "augmented" => true
    }
  end

  defp word_dropout(tokens, _pos_tags) when length(tokens) <= 2 do
    Enum.join(tokens, " ")
  end

  defp word_dropout(tokens, pos_tags) do
    droppable_indices =
      tokens
      |> Enum.with_index()
      |> Enum.filter(fn {_token, idx} ->
        pos = Enum.at(pos_tags, idx)
        pos != nil and pos in @droppable_pos
      end)
      |> Enum.map(fn {_token, idx} -> idx end)

    if droppable_indices == [] do
      Enum.join(tokens, " ")
    else
      drop_idx = Enum.random(droppable_indices)

      tokens
      |> Enum.with_index()
      |> Enum.reject(fn {_token, idx} -> idx == drop_idx end)
      |> Enum.map(fn {token, _idx} -> token end)
      |> Enum.join(" ")
    end
  end

  defp adjacent_swap(tokens, _pos_tags) when length(tokens) <= 2 do
    Enum.join(tokens, " ")
  end

  defp adjacent_swap(tokens, pos_tags) do
    content_pairs =
      tokens
      |> Enum.with_index()
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.filter(fn [{_t1, i1}, {_t2, _i2}] ->
        p1 = Enum.at(pos_tags, i1)
        p2 = Enum.at(pos_tags, i1 + 1)
        (p1 == nil or p1 not in @droppable_pos) and
          (p2 == nil or p2 not in @droppable_pos)
      end)

    if content_pairs == [] do
      Enum.join(tokens, " ")
    else
      [{_t1, idx1}, {_t2, _idx2}] = Enum.random(content_pairs)

      tokens
      |> List.update_at(idx1, fn _ -> Enum.at(tokens, idx1 + 1) end)
      |> List.update_at(idx1 + 1, fn _ -> Enum.at(tokens, idx1) end)
      |> Enum.join(" ")
    end
  end

  defp add_prefix(text) do
    lower = String.downcase(text)

    already_prefixed =
      Enum.any?(@removable_prefixes, fn prefix ->
        String.starts_with?(lower, prefix)
      end)

    if already_prefixed do
      lower
    else
      {prefix, _} = Enum.random(@add_prefixes)
      prefix <> lower
    end
  end

  defp remove_prefix(text) do
    lower = String.downcase(text)

    result =
      Enum.reduce_while(@removable_prefixes, lower, fn prefix, acc ->
        if String.starts_with?(acc, prefix) do
          {:halt, String.slice(acc, String.length(prefix)..-1//1)}
        else
          {:cont, acc}
        end
      end)

    if result == lower do
      add_prefix(text)
    else
      result
    end
  end

  defp truncate_end(tokens, _pos_tags) when length(tokens) <= 3 do
    Enum.join(tokens, " ")
  end

  defp truncate_end(tokens, pos_tags) do
    last_pos = List.last(pos_tags)

    if last_pos in @droppable_pos or last_pos in ["ADV", "ADJ"] do
      tokens |> Enum.slice(0..-2//1) |> Enum.join(" ")
    else
      Enum.join(tokens, " ")
    end
  end

  defp normalize(text) do
    text |> String.downcase() |> String.trim()
  end

  defp source_gold_standard_path(task) do
    Path.join([File.cwd!(), "apps", "brain", "priv", "evaluation", task, "gold_standard.json"])
  end

  defp return_early do
    :ok
  end
end
