defmodule Mix.Tasks.Audit.NerGold do
  @shortdoc "Audit NER gold standard for data quality issues"
  @moduledoc """
  Audit the NER gold standard for broken spans, type distribution, and data quality.

  ## Usage

      mix audit.ner_gold           # Run audit
      mix audit.ner_gold --fix     # Export broken spans for manual correction
  """

  use Mix.Task

  alias Brain.ML.EvaluationStore

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    fix? = "--fix" in args

    gold = EvaluationStore.load_gold_standard("ner")

    if gold == [] do
      IO.puts("\nNo gold standard data for NER.")
      IO.puts("Add examples to: priv/evaluation/ner/gold_standard.json")
      IO.puts("")
      exit(:normal)
    end

    all_entities = extract_all_entities(gold)

    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("NER GOLD STANDARD AUDIT")
    IO.puts(String.duplicate("=", 60))

    report_totals(gold, all_entities)
    report_type_distribution(all_entities)
    broken = report_broken_spans(gold)
    report_sparse_types(all_entities)
    report_empty_values(gold)
    report_duplicate_texts(gold)

    if fix? and broken != [] do
      write_broken_spans(broken)
    end

    IO.puts("")
  end

  defp extract_all_entities(gold) do
    Enum.flat_map(gold, fn example ->
      entities = example["entities"] || example["expected"] || []
      Enum.map(entities, &Map.put(&1, "_source_text", example["text"]))
    end)
  end

  defp report_totals(gold, all_entities) do
    IO.puts("\n--- Totals ---")
    IO.puts("  Examples:           #{length(gold)}")
    IO.puts("  Entity annotations: #{length(all_entities)}")
  end

  defp report_type_distribution(all_entities) do
    IO.puts("\n--- Type Distribution ---")

    all_entities
    |> Enum.frequencies_by(fn e -> e["type"] end)
    |> Enum.sort_by(fn {_type, count} -> count end, :desc)
    |> Enum.each(fn {type, count} ->
      IO.puts("  #{String.pad_trailing(to_string(type), 30)} #{count}")
    end)
  end

  defp report_broken_spans(gold) do
    broken =
      Enum.flat_map(gold, fn example ->
        text = example["text"] || ""
        text_down = String.downcase(text)
        entities = example["entities"] || example["expected"] || []

        Enum.flat_map(entities, fn entity ->
          value = entity["value"] || ""

          if value == "" do
            []
          else
            value_down = String.downcase(value)

            if String.contains?(text_down, value_down) do
              []
            else
              [%{"text" => text, "type" => entity["type"], "value" => value}]
            end
          end
        end)
      end)

    IO.puts("\n--- Broken Spans ---")
    IO.puts("  Count: #{length(broken)}")

    if broken != [] do
      show = Enum.take(broken, 20)

      Enum.each(show, fn b ->
        IO.puts("    text:  #{inspect(b["text"])}")
        IO.puts("    type:  #{b["type"]}")
        IO.puts("    value: #{inspect(b["value"])}")
        IO.puts("")
      end)

      remaining = length(broken) - length(show)

      if remaining > 0 do
        IO.puts("    ... and #{remaining} more")
      end
    end

    broken
  end

  defp report_sparse_types(all_entities) do
    sparse =
      all_entities
      |> Enum.frequencies_by(fn e -> e["type"] end)
      |> Enum.filter(fn {_type, count} -> count < 10 end)
      |> Enum.sort_by(fn {_type, count} -> count end)

    IO.puts("\n--- Types With < 10 Examples ---")

    if sparse == [] do
      IO.puts("  None")
    else
      Enum.each(sparse, fn {type, count} ->
        IO.puts("  #{String.pad_trailing(to_string(type), 30)} #{count}")
      end)
    end
  end

  defp report_empty_values(gold) do
    empty =
      Enum.flat_map(gold, fn example ->
        entities = example["entities"] || example["expected"] || []

        Enum.flat_map(entities, fn entity ->
          value = entity["value"]

          if is_nil(value) or value == "" do
            [%{"text" => example["text"], "type" => entity["type"], "value" => value}]
          else
            []
          end
        end)
      end)

    IO.puts("\n--- Empty/Null Values ---")
    IO.puts("  Count: #{length(empty)}")

    if empty != [] do
      Enum.take(empty, 20)
      |> Enum.each(fn e ->
        IO.puts("    text: #{inspect(e["text"])}  type: #{e["type"]}")
      end)
    end
  end

  defp report_duplicate_texts(gold) do
    dupes =
      gold
      |> Enum.frequencies_by(fn ex -> ex["text"] end)
      |> Enum.filter(fn {_text, count} -> count > 1 end)
      |> Enum.sort_by(fn {_text, count} -> count end, :desc)

    IO.puts("\n--- Duplicate Texts ---")
    IO.puts("  Count: #{length(dupes)}")

    if dupes != [] do
      Enum.take(dupes, 20)
      |> Enum.each(fn {text, count} ->
        IO.puts("    #{count}x  #{inspect(text)}")
      end)
    end
  end

  defp write_broken_spans(broken) do
    path =
      case :code.priv_dir(:brain) do
        {:error, _} -> "apps/brain/priv/evaluation/ner/broken_spans.json"
        dir -> Path.join([to_string(dir), "evaluation", "ner", "broken_spans.json"])
      end

    json = Jason.encode!(broken, pretty: true)
    File.write!(path, json)
    IO.puts("\nBroken spans written to: #{path}")
    IO.puts("Review and fix the entries, then update gold_standard.json accordingly.")
  end
end
