defmodule Mix.Tasks.RebuildGoldStandard do
  @shortdoc "Rebuild intent gold standard using Dialogflow anchoring"
  @moduledoc """
  Rebuilds the intent gold standard with a three-layer labeling strategy:

  1. **Dialogflow anchoring** — exact-match against Dialogflow training phrases
  2. **Confidence-gated model labels** — for unmatched texts, use the feature-vector
     classifier when confidence >= 0.7, otherwise mark `"needs_review"`
  3. **Validation** — warn when a Dialogflow-matched label disagrees with the model

  ## Usage

      mix rebuild_gold_standard              # Dry-run (prints summary only)
      mix rebuild_gold_standard --save       # Write rebuilt gold standard
      mix rebuild_gold_standard --verbose    # Show per-entry details
      mix rebuild_gold_standard --save --verbose
  """

  use Mix.Task
  require Logger

  alias Brain.Analysis.{FeatureExtractor, Pipeline}
  alias Brain.ML.{EvaluationStore, MicroClassifiers}

  @dialogflow_dir Path.join(["data", "intents"])

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    save? = "--save" in args
    verbose? = "--verbose" in args

    IO.puts("\nAwaiting MicroClassifiers readiness...")
    MicroClassifiers.await_ready(:infinity)
    IO.puts("MicroClassifiers ready.\n")

    IO.puts(String.duplicate("=", 60))
    IO.puts("REBUILD INTENT GOLD STANDARD")
    IO.puts(String.duplicate("=", 60) <> "\n")

    dialogflow_map = load_dialogflow_phrases()
    IO.puts("  Dialogflow: #{map_size(dialogflow_map)} unique phrases loaded\n")

    gold = EvaluationStore.load_gold_standard("intent")

    if gold == [] do
      Mix.raise("No gold standard data found. Cannot rebuild.")
    end

    IO.puts("  Gold standard: #{length(gold)} entries\n")

    {rebuilt, stats} = rebuild(gold, dialogflow_map, verbose?)

    print_summary(stats, length(gold))

    if save? do
      write_output(rebuilt)
    else
      IO.puts("  Dry-run mode. Use --save to write the rebuilt file.\n")
    end
  end

  # ---------------------------------------------------------------------------
  # Dialogflow loading
  # ---------------------------------------------------------------------------

  defp load_dialogflow_phrases do
    usersays_files = Path.wildcard(Path.join(@dialogflow_dir, "*_usersays_en.json"))

    Enum.reduce(usersays_files, %{}, fn usersays_path, acc ->
      metadata_path = derive_metadata_path(usersays_path)
      intent_label = read_intent_label(metadata_path)

      case intent_label do
        nil ->
          Logger.warning("No metadata for #{Path.basename(usersays_path)}, skipping")
          acc

        label ->
          phrases = read_usersays(usersays_path)

          Enum.reduce(phrases, acc, fn phrase, inner_acc ->
            key = phrase |> String.downcase() |> String.trim()
            Map.put_new(inner_acc, key, label)
          end)
      end
    end)
  end

  defp derive_metadata_path(usersays_path) do
    usersays_path
    |> String.replace("_usersays_en.json", ".json")
  end

  defp read_intent_label(metadata_path) do
    case File.read(metadata_path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, %{"name" => name}} ->
            name
            |> String.downcase()
            |> String.replace(~r/\s+/, "_")

          _ ->
            nil
        end

      {:error, _} ->
        nil
    end
  end

  defp read_usersays(path) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, entries} when is_list(entries) ->
            Enum.map(entries, fn entry ->
              entry
              |> Map.get("data", [])
              |> Enum.map_join(&Map.get(&1, "text", ""))
            end)

          _ ->
            []
        end

      {:error, _} ->
        []
    end
  end

  # ---------------------------------------------------------------------------
  # Rebuild logic
  # ---------------------------------------------------------------------------

  defp rebuild(gold, dialogflow_map, verbose?) do
    total = length(gold)

    {entries, stats} =
      gold
      |> Enum.with_index(1)
      |> Enum.map_reduce(
        %{dialogflow_matched: 0, model_labeled: 0, needs_review: 0, validation_warnings: 0},
        fn {example, idx}, stats ->
          if rem(idx, 500) == 0, do: IO.write("\r  Progress: #{idx}/#{total}")

          text = example["text"]
          key = text |> String.downcase() |> String.trim()

          case Map.get(dialogflow_map, key) do
            nil ->
              layer2_result(text, example, stats, verbose?)

            df_label ->
              entry = %{"text" => text, "intent" => df_label}
              stats = Map.update!(stats, :dialogflow_matched, &(&1 + 1))

              stats = validate_against_model(text, df_label, stats, verbose?)

              if verbose? do
                IO.puts("  [dialogflow] \"#{truncate(text, 50)}\" => #{df_label}")
              end

              {entry, stats}
          end
        end
      )

    if total >= 500, do: IO.write("\r  Progress: #{total}/#{total}\n")

    {entries, stats}
  end

  defp layer2_result(text, _example, stats, verbose?) do
    {label, confidence} = classify_with_model(text)

    if confidence >= 0.7 do
      entry = %{"text" => text, "intent" => label}
      stats = Map.update!(stats, :model_labeled, &(&1 + 1))

      if verbose? do
        IO.puts(
          "  [model #{Float.round(confidence * 100, 1)}%] \"#{truncate(text, 50)}\" => #{label}"
        )
      end

      {entry, stats}
    else
      entry = %{"text" => text, "intent" => label, "status" => "needs_review"}
      stats = Map.update!(stats, :needs_review, &(&1 + 1))

      if verbose? do
        IO.puts(
          "  [needs_review #{Float.round(confidence * 100, 1)}%] \"#{truncate(text, 50)}\" => #{label}"
        )
      end

      {entry, stats}
    end
  end

  defp validate_against_model(text, df_label, stats, verbose?) do
    {model_label, _confidence} = classify_with_model(text)

    if model_label != df_label do
      stats = Map.update!(stats, :validation_warnings, &(&1 + 1))

      if verbose? do
        Logger.warning(
          "Label mismatch: \"#{truncate(text, 40)}\" dialogflow=#{df_label} model=#{model_label}"
        )
      end

      stats
    else
      stats
    end
  end

  defp classify_with_model(text) do
    analysis = Pipeline.analyze_chunk(text, side_effects: false)
    {feature_vector, _word_feats} = FeatureExtractor.extract(analysis)

    case MicroClassifiers.classify_vector(:intent_full, feature_vector) do
      {:ok, label, confidence} -> {to_string(label), confidence}
      _ -> {"unknown", 0.0}
    end
  rescue
    e ->
      Logger.warning("Classification failed for \"#{truncate(text, 40)}\": #{Exception.message(e)}")
      {"unknown", 0.0}
  catch
    :exit, reason ->
      Logger.warning("Classification exit for \"#{truncate(text, 40)}\": #{inspect(reason)}")
      {"unknown", 0.0}
  end

  # ---------------------------------------------------------------------------
  # Output
  # ---------------------------------------------------------------------------

  defp print_summary(stats, total) do
    IO.puts("\n--- Rebuild Summary ---\n")
    IO.puts("  Total entries:          #{total}")
    IO.puts("  Dialogflow matched:     #{stats.dialogflow_matched}")
    IO.puts("  Model labeled (>=0.7):  #{stats.model_labeled}")
    IO.puts("  Needs review:           #{stats.needs_review}")
    IO.puts("  Validation warnings:    #{stats.validation_warnings}")

    pct_anchored = Float.round(stats.dialogflow_matched / max(total, 1) * 100, 1)
    IO.puts("\n  Dialogflow coverage:    #{pct_anchored}%\n")
  end

  defp write_output(entries) do
    output_path = EvaluationStore.gold_standard_path("intent")
    backup_path = output_path <> ".bak"

    if File.exists?(output_path) do
      File.cp!(output_path, backup_path)
      IO.puts("  Backed up original to: #{backup_path}")
    end

    json = Jason.encode!(entries, pretty: true)
    File.write!(output_path, json)
    IO.puts("  Wrote #{length(entries)} entries to: #{output_path}\n")
  end

  defp truncate(text, max_len) do
    if String.length(text) > max_len do
      String.slice(text, 0, max_len) <> "..."
    else
      text
    end
  end
end
