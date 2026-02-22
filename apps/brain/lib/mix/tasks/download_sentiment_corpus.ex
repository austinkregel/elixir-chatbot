defmodule Mix.Tasks.DownloadSentimentCorpus do
  @moduledoc """
  Downloads the TweetEval sentiment dataset and merges it with the existing
  sentiment gold standard to expand training data.

  ## Usage

      mix download_sentiment_corpus [options]

  ## Options

    --max N          Maximum examples per class (default: 2000)
    --download       Force re-download even if cached
    --preview        Show stats without writing
    --output PATH    Output path (default: apps/brain/priv/evaluation/sentiment/gold_standard.json)

  ## Data Source

  Downloads from: https://github.com/cardiffnlp/tweeteval
  (SemEval-2017 Task 4 sentiment, 3-class: negative/neutral/positive)
  """

  use Mix.Task
  require Logger

  @shortdoc "Download TweetEval sentiment data and merge with gold standard"

  @text_url "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/train_text.txt"
  @labels_url "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/train_labels.txt"
  @cache_dir "priv/data_cache"
  @label_map %{0 => "negative", 1 => "neutral", 2 => "positive"}

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    {opts, _, _} =
      OptionParser.parse(args,
        strict: [max: :integer, download: :boolean, preview: :boolean, output: :string]
      )

    max_per_class = Keyword.get(opts, :max, 2000)
    force_download = Keyword.get(opts, :download, false)
    preview? = Keyword.get(opts, :preview, false)

    output_path =
      Keyword.get(opts, :output, Brain.priv_path("evaluation/sentiment/gold_standard.json"))

    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("  DOWNLOAD SENTIMENT CORPUS")
    IO.puts(String.duplicate("=", 60))
    IO.puts("  Source: TweetEval (SemEval-2017 Task 4)")
    IO.puts("  Max per class: #{max_per_class}")
    IO.puts("")

    File.mkdir_p!(@cache_dir)

    texts = download_or_cache("tweeteval_train_text.txt", @text_url, force_download)
    labels = download_or_cache("tweeteval_train_labels.txt", @labels_url, force_download)

    text_lines = String.split(texts, "\n", trim: true)
    label_lines = String.split(labels, "\n", trim: true)

    if length(text_lines) != length(label_lines) do
      IO.puts("  ERROR: text lines (#{length(text_lines)}) != label lines (#{length(label_lines)})")
      System.halt(1)
    end

    IO.puts("  Downloaded #{length(text_lines)} examples")

    new_examples =
      Enum.zip(text_lines, label_lines)
      |> Enum.map(fn {text, label_str} ->
        label_idx = String.trim(label_str) |> String.to_integer()
        %{"text" => String.trim(text), "sentiment" => Map.get(@label_map, label_idx, "neutral")}
      end)
      |> Enum.filter(fn %{"text" => t} -> byte_size(t) > 5 end)

    grouped = Enum.group_by(new_examples, & &1["sentiment"])

    IO.puts("\n  TweetEval class distribution:")

    for label <- ["negative", "neutral", "positive"] do
      count = length(Map.get(grouped, label, []))
      IO.puts("    #{String.pad_trailing(label, 10)}: #{count}")
    end

    sampled =
      Enum.flat_map(["negative", "neutral", "positive"], fn label ->
        examples = Map.get(grouped, label, [])
        examples |> Enum.shuffle() |> Enum.take(max_per_class)
      end)

    IO.puts("\n  Sampled #{length(sampled)} examples (#{max_per_class} max/class)")

    existing =
      if File.exists?(output_path) do
        output_path |> File.read!() |> Jason.decode!()
      else
        []
      end

    IO.puts("  Existing gold standard: #{length(existing)} examples")

    existing_texts = MapSet.new(existing, fn %{"text" => t} -> normalize(t) end)

    novel =
      Enum.filter(sampled, fn %{"text" => t} ->
        not MapSet.member?(existing_texts, normalize(t))
      end)

    merged = existing ++ novel

    merged_grouped = Enum.group_by(merged, & &1["sentiment"])

    IO.puts("\n  Final merged distribution:")

    for label <- ["negative", "neutral", "positive"] do
      count = length(Map.get(merged_grouped, label, []))
      IO.puts("    #{String.pad_trailing(label, 10)}: #{count}")
    end

    IO.puts("  Total: #{length(merged)} (#{length(novel)} new)")

    if preview? do
      IO.puts("\n  Preview mode — no files written.")
    else
      output_path |> Path.dirname() |> File.mkdir_p!()
      json = Jason.encode!(merged, pretty: true)
      File.write!(output_path, json)
      IO.puts("\n  Written to: #{output_path}")
    end

    IO.puts("")
  end

  defp download_or_cache(filename, url, force?) do
    path = Path.join(@cache_dir, filename)

    if force? or not File.exists?(path) do
      IO.puts("  Downloading #{filename}...")

      case Req.get(url, receive_timeout: 120_000, connect_options: [timeout: 30_000]) do
        {:ok, %{status: 200, body: body}} ->
          File.write!(path, body)
          IO.puts("    #{byte_size(body)} bytes cached")
          body

        {:ok, %{status: status}} ->
          IO.puts("  ERROR: HTTP #{status} for #{url}")
          System.halt(1)

        {:error, reason} ->
          IO.puts("  ERROR: #{inspect(reason)}")
          System.halt(1)
      end
    else
      IO.puts("  Using cached #{filename}")
      File.read!(path)
    end
  end

  defp normalize(text) do
    text |> String.downcase() |> String.trim()
  end
end
