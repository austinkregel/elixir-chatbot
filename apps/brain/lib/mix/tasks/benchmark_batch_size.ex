defmodule Mix.Tasks.BenchmarkBatchSize do
  @moduledoc """
  Runs short training experiments across different batch sizes to find the
  optimal setting for the current hardware and dataset.

  Each experiment trains the encoder+intent head for a fixed number of
  epochs and reports wall-clock time and dataset geometry. Accuracy
  for each run is visible in the Axon training output above the summary.

  ## Usage

      mix benchmark_batch_size                    # Test default sizes
      mix benchmark_batch_size --sizes 16,32,64   # Test specific sizes
      mix benchmark_batch_size --epochs 15         # More epochs per test
  """

  use Mix.Task
  require Logger

  alias Brain.ML.LSTM.UnifiedModel

  @shortdoc "Benchmark batch sizes for GPU training"

  @default_sizes [16, 32, 48, 64, 128, 256, 512]
  @default_epochs 10

  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          sizes: :string,
          epochs: :integer
        ]
      )

    sizes =
      case opts[:sizes] do
        nil ->
          @default_sizes

        str ->
          str
          |> String.split(",")
          |> Enum.map(&String.trim/1)
          |> Enum.map(&String.to_integer/1)
      end

    epochs = opts[:epochs] || @default_epochs

    Application.put_env(:brain, :skip_ml_init, true)
    Mix.Task.run("app.start")

    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("  Batch Size Benchmark — Encoder + Intent Head Only")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("")
    Mix.shell().info("  Batch sizes: #{Enum.join(sizes, ", ")}")
    Mix.shell().info("  Epochs per test: #{epochs}")
    Mix.shell().info("  EXLA client: #{inspect(Application.get_env(:exla, :default_client))}")
    Mix.shell().info("")
    Mix.shell().info("  Each run trains ONLY the encoder+intent head (no sentiment/speech_act)")
    Mix.shell().info("  so runs complete quickly. Check the Axon output above each summary")
    Mix.shell().info("  line for the final accuracy and loss of that run.")
    Mix.shell().info("")

    results =
      Enum.map(sizes, fn batch_size ->
        Mix.shell().info("")
        Mix.shell().info("-" |> String.duplicate(70))
        Mix.shell().info("  batch_size=#{batch_size}, epochs=#{epochs}")
        Mix.shell().info("-" |> String.duplicate(70))
        Mix.shell().info("")

        start_time = System.monotonic_time(:millisecond)

        {status, metrics} =
          try do
            case UnifiedModel.train(
                   batch_size: batch_size,
                   epochs: epochs,
                   sentiment_epochs: 0,
                   speech_act_epochs: 0,
                   name: "bench_bs#{batch_size}_ep#{epochs}"
                 ) do
              {:ok, result} ->
                {:ok, Map.get(result, :intent_metrics, %{})}

              {:error, reason} ->
                {{:error, reason}, %{}}
            end
          rescue
            e -> {{:error, Exception.message(e)}, %{}}
          end

        elapsed_ms = System.monotonic_time(:millisecond) - start_time
        elapsed_s = Float.round(elapsed_ms / 1000, 1)

        batches_per_epoch = div(5300, batch_size)
        total_updates = batches_per_epoch * epochs

        accuracy = metrics[:accuracy]
        val_accuracy = metrics[:validation_accuracy]
        loss = metrics[:loss]
        final_epoch = metrics[:epoch]

        Mix.shell().info("")
        Mix.shell().info("  >> batch_size=#{batch_size}: #{elapsed_s}s, " <>
          "acc=#{fmt_pct(accuracy)}, val_acc=#{fmt_pct(val_accuracy)}, " <>
          "loss=#{fmt_num(loss)}, epoch=#{final_epoch || "?"}")
        Mix.shell().info("")

        %{
          batch_size: batch_size,
          elapsed_s: elapsed_s,
          status: status,
          accuracy: accuracy,
          val_accuracy: val_accuracy,
          loss: loss,
          final_epoch: final_epoch,
          batches_per_epoch: batches_per_epoch,
          total_updates: total_updates
        }
      end)

    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("  Summary")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("")

    header =
      String.pad_trailing("Batch", 8) <>
        String.pad_trailing("Time(s)", 10) <>
        String.pad_trailing("Train Acc", 12) <>
        String.pad_trailing("Val Acc", 12) <>
        String.pad_trailing("Loss", 10) <>
        String.pad_trailing("Epoch", 8) <>
        String.pad_trailing("Batches/Ep", 13) <>
        "Updates"

    Mix.shell().info("  #{header}")
    Mix.shell().info("  " <> String.duplicate("-", String.length(header)))

    Enum.each(results, fn r ->
      row =
        String.pad_trailing("#{r.batch_size}", 8) <>
          String.pad_trailing("#{r.elapsed_s}", 10) <>
          String.pad_trailing(fmt_pct(r.accuracy), 12) <>
          String.pad_trailing(fmt_pct(r.val_accuracy), 12) <>
          String.pad_trailing(fmt_num(r.loss), 10) <>
          String.pad_trailing("#{r.final_epoch || "?"}", 8) <>
          String.pad_trailing("#{r.batches_per_epoch}", 13) <>
          "#{r.total_updates}"

      Mix.shell().info("  #{row}")
    end)

    successful = Enum.filter(results, &(&1.status == :ok && r_acc(&1) > 0))

    if length(successful) > 1 do
      best_acc = Enum.max_by(successful, &r_acc/1)
      fastest = Enum.min_by(successful, & &1.elapsed_s)

      best_efficiency =
        Enum.max_by(successful, fn r ->
          if r.elapsed_s > 0, do: r_acc(r) / r.elapsed_s, else: 0
        end)

      Mix.shell().info("")
      Mix.shell().info("  Best accuracy:   batch_size=#{best_acc.batch_size} (#{fmt_pct(r_acc(best_acc))})")
      Mix.shell().info("  Fastest:         batch_size=#{fastest.batch_size} (#{fastest.elapsed_s}s)")
      Mix.shell().info("  Best acc/time:   batch_size=#{best_efficiency.batch_size}")
    end

    Mix.shell().info("")
  end

  defp r_acc(%{val_accuracy: v}) when is_number(v) and v > 0, do: v
  defp r_acc(%{accuracy: a}) when is_number(a), do: a
  defp r_acc(_), do: 0

  defp fmt_pct(nil), do: "n/a"
  defp fmt_pct(v) when is_number(v), do: "#{Float.round(v * 100, 1)}%"

  defp fmt_num(nil), do: "n/a"
  defp fmt_num(v) when is_number(v), do: "#{Float.round(v, 4)}"
end
