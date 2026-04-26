defmodule Mix.Tasks.Split.HeldOut do
  @moduledoc false
  @shortdoc "Split a stratified held-out test set from a gold standard"

  use Mix.Task

  @default_size 1000
  @default_seed 42

  @label_keys %{
    "sentiment" => "sentiment",
    "speech_act" => "speech_act",
    "intent" => "intent"
  }

  @impl Mix.Task
  def run(args) do
    {opts, positional, _} =
      OptionParser.parse(args,
        strict: [size: :integer, seed: :integer],
        aliases: [s: :size]
      )

    task =
      case positional do
        [t | _] -> t
        [] -> Mix.raise("Usage: mix split.held_out <task> [--size N] [--seed S]")
      end

    label_key =
      Map.get(@label_keys, task) ||
        Mix.raise(
          "Unknown task #{inspect(task)}. Supported: #{@label_keys |> Map.keys() |> Enum.join(", ")}"
        )

    size = Keyword.get(opts, :size, @default_size)
    seed = Keyword.get(opts, :seed, @default_seed)

    held_path = held_out_path(task)

    if File.exists?(held_path) do
      Mix.raise(
        "#{held_path} already exists. Delete it first to prevent double-splitting."
      )
    end

    gold_path = gold_path(task)

    examples =
      case File.read(gold_path) do
        {:ok, content} ->
          case Jason.decode(content) do
            {:ok, data} when is_list(data) -> data
            _ -> Mix.raise("Failed to parse #{gold_path} as a JSON array")
          end

        {:error, reason} ->
          Mix.raise("Cannot read #{gold_path}: #{inspect(reason)}")
      end

    total = length(examples)

    if size >= div(total, 2) do
      Mix.raise(
        "--size #{size} must be less than 50% of total dataset (#{total} examples, max #{div(total, 2) - 1})"
      )
    end

    grouped = Enum.group_by(examples, &Map.fetch!(&1, label_key))

    :rand.seed(:exsss, {seed, seed, seed})

    {held_out, remaining} = stratified_split(grouped, size, total)

    File.mkdir_p!(Path.dirname(held_path))
    File.write!(held_path, Jason.encode!(held_out, pretty: true) <> "\n")
    File.write!(gold_path, Jason.encode!(remaining, pretty: true) <> "\n")

    report(examples, held_out, remaining, label_key, gold_path, held_path)
  end

  defp stratified_split(grouped, target_size, total) do
    allocations =
      grouped
      |> Enum.map(fn {label, items} ->
        proportion = length(items) / total
        count = round(proportion * target_size)
        {label, max(count, 1)}
      end)

    allocated_total = Enum.reduce(allocations, 0, fn {_, c}, acc -> acc + c end)
    diff = target_size - allocated_total

    allocations =
      if diff != 0 do
        adjust_allocations(allocations, diff, grouped)
      else
        allocations
      end

    {held, rest} =
      Enum.reduce(allocations, {[], []}, fn {label, count}, {held_acc, rest_acc} ->
        items = Map.fetch!(grouped, label)
        shuffled = Enum.shuffle(items)
        {taken, kept} = Enum.split(shuffled, count)
        {held_acc ++ taken, rest_acc ++ kept}
      end)

    {held, rest}
  end

  defp adjust_allocations(allocations, diff, grouped) when diff > 0 do
    sorted =
      Enum.sort_by(allocations, fn {label, count} ->
        -(length(Map.fetch!(grouped, label)) - count)
      end)

    do_adjust(sorted, diff)
  end

  defp adjust_allocations(allocations, diff, grouped) when diff < 0 do
    sorted =
      Enum.sort_by(allocations, fn {label, _count} ->
        length(Map.fetch!(grouped, label))
      end)

    do_adjust(sorted, diff)
  end

  defp do_adjust(allocations, 0), do: allocations

  defp do_adjust([{label, count} | rest], diff) when diff > 0 do
    [{label, count + 1} | do_adjust(rest, diff - 1)]
  end

  defp do_adjust([{label, count} | rest], diff) when diff < 0 and count > 1 do
    [{label, count - 1} | do_adjust(rest, diff + 1)]
  end

  defp do_adjust([head | rest], diff), do: [head | do_adjust(rest, diff)]

  defp report(original, held_out, remaining, label_key, gold_path, held_path) do
    IO.puts("\n=== Held-Out Split Report ===\n")

    orig_dist = Enum.frequencies_by(original, &Map.fetch!(&1, label_key))
    held_dist = Enum.frequencies_by(held_out, &Map.fetch!(&1, label_key))
    remain_dist = Enum.frequencies_by(remaining, &Map.fetch!(&1, label_key))

    IO.puts("Original: #{length(original)} examples")
    print_distribution(orig_dist)

    IO.puts("\nHeld-out: #{length(held_out)} examples")
    print_distribution(held_dist)

    IO.puts("\nRemaining training: #{length(remaining)} examples")
    print_distribution(remain_dist)

    IO.puts("\nFiles written:")
    IO.puts("  Held-out test set: #{held_path}")
    IO.puts("  Training set:      #{gold_path}")
    IO.puts("")
  end

  defp print_distribution(dist) do
    dist
    |> Enum.sort_by(fn {_, count} -> -count end)
    |> Enum.each(fn {label, count} ->
      IO.puts("  #{label}: #{count}")
    end)
  end

  defp gold_path(task) do
    case :code.priv_dir(:brain) do
      {:error, _} -> "apps/brain/priv/evaluation/#{task}/gold_standard.json"
      priv -> Path.join(priv, "evaluation/#{task}/gold_standard.json")
    end
  end

  defp held_out_path(task) do
    case :code.priv_dir(:brain) do
      {:error, _} -> "apps/brain/priv/evaluation/#{task}/held_out.json"
      priv -> Path.join(priv, "evaluation/#{task}/held_out.json")
    end
  end
end
