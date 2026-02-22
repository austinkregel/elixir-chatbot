defmodule Mix.Tasks.DownloadSpeechActCorpus do
  @moduledoc """
  Downloads the DailyDialog dataset and converts its dialog act + emotion
  annotations into speech act gold standard data.

  ## Usage

      mix download_speech_act_corpus [options]

  ## Options

    --max N          Maximum examples per class (default: 500)
    --download       Force re-download even if cached
    --preview        Show stats without writing
    --output PATH    Output path (default: apps/brain/priv/evaluation/speech_act/gold_standard.json)

  ## Data Source

  Downloads from: https://aclanthology.org/attachments/I17-1099.Datasets.zip
  (DailyDialog, CC-BY-NC-SA-4.0)

  ## Mapping

    DailyDialog act 1 (inform)     -> assertive
    DailyDialog act 2 (question)   -> directive
    DailyDialog act 3 (directive)  -> directive
    DailyDialog act 4 (commissive) -> commissive
    Emotion + short utterance      -> expressive
    Curated seed examples          -> declarative
  """

  use Mix.Task
  require Logger

  @shortdoc "Download DailyDialog speech act data and merge with gold standard"

  @dataset_url "https://aclanthology.org/attachments/I17-1099.Datasets.zip"
  @cache_dir "priv/data_cache"
  @cache_file "dailydialog.zip"

  @act_map %{
    1 => "assertive",
    2 => "directive",
    3 => "directive",
    4 => "commissive"
  }

  @expressive_markers [
    "thank", "thanks", "hello", "hi", "hey", "goodbye", "bye",
    "sorry", "congratulations", "congrats", "wow", "oh",
    "great", "awesome", "wonderful", "terrible", "amazing",
    "welcome", "cheers", "bravo", "ouch", "yay", "hooray",
    "good morning", "good evening", "good night", "good afternoon",
    "happy birthday", "merry christmas", "happy new year"
  ]

  @declarative_seeds [
    "I hereby declare this meeting adjourned.",
    "You are fired.",
    "You're hired.",
    "I now pronounce you husband and wife.",
    "I sentence you to five years in prison.",
    "I name this ship the Queen Mary.",
    "You are under arrest.",
    "I resign from my position effective immediately.",
    "I declare bankruptcy.",
    "Court is now in session.",
    "I appoint you as the new chairman.",
    "The defendant is found guilty.",
    "The defendant is found not guilty.",
    "I christen this building the Community Center.",
    "I officially open this ceremony.",
    "This meeting is called to order.",
    "I declare a state of emergency.",
    "You are hereby promoted to senior manager.",
    "I award you the medal of honor.",
    "I ban you from this establishment.",
    "I grant you permission to proceed.",
    "I revoke your access privileges.",
    "The motion is carried.",
    "The motion is denied.",
    "I declare the winner of the competition.",
    "You are suspended without pay.",
    "I accept your resignation.",
    "I reject your proposal.",
    "Class dismissed.",
    "I declare this project complete.",
    "You are excused from jury duty.",
    "I authorize the release of funds.",
    "The bill is vetoed.",
    "I certify this document as authentic.",
    "I withdraw my objection.",
    "The case is dismissed.",
    "I overrule the objection.",
    "I sustain the objection.",
    "You are granted asylum.",
    "I declare a mistrial.",
    "War is declared.",
    "I surrender.",
    "I forfeit the match.",
    "The contract is terminated.",
    "I dissolve this partnership.",
    "I annul this agreement.",
    "Your application is approved.",
    "Your application is denied.",
    "I waive my right to an attorney.",
    "I plead guilty."
  ]

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    {opts, _, _} =
      OptionParser.parse(args,
        strict: [max: :integer, download: :boolean, preview: :boolean, output: :string]
      )

    max_per_class = Keyword.get(opts, :max, 500)
    force_download = Keyword.get(opts, :download, false)
    preview? = Keyword.get(opts, :preview, false)

    output_path =
      Keyword.get(opts, :output, Brain.priv_path("evaluation/speech_act/gold_standard.json"))

    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("  DOWNLOAD SPEECH ACT CORPUS")
    IO.puts(String.duplicate("=", 60))
    IO.puts("  Source: DailyDialog (IJCNLP 2017)")
    IO.puts("  Max per class: #{max_per_class}")
    IO.puts("")

    File.mkdir_p!(@cache_dir)

    zip_data = download_or_cache(force_download)
    {utterances, acts, emotions} = parse_dailydialog_zip(zip_data)

    IO.puts("  Parsed #{length(utterances)} utterances")

    examples = build_speech_act_examples(utterances, acts, emotions)

    grouped = Enum.group_by(examples, & &1["speech_act"])

    IO.puts("\n  DailyDialog mapped distribution:")

    for label <- ["assertive", "directive", "commissive", "expressive", "declarative"] do
      count = length(Map.get(grouped, label, []))
      IO.puts("    #{String.pad_trailing(label, 14)}: #{count}")
    end

    sampled =
      Enum.flat_map(
        ["assertive", "directive", "commissive", "expressive", "declarative"],
        fn label ->
          Map.get(grouped, label, []) |> Enum.shuffle() |> Enum.take(max_per_class)
        end
      )

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

    merged_grouped = Enum.group_by(merged, & &1["speech_act"])

    IO.puts("\n  Final merged distribution:")

    for label <- ["assertive", "directive", "commissive", "expressive", "declarative"] do
      count = length(Map.get(merged_grouped, label, []))
      IO.puts("    #{String.pad_trailing(label, 14)}: #{count}")
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

  defp download_or_cache(force?) do
    path = Path.join(@cache_dir, @cache_file)

    if force? or not File.exists?(path) do
      IO.puts("  Downloading DailyDialog dataset...")

      case Req.get(@dataset_url,
             receive_timeout: 120_000,
             connect_options: [timeout: 30_000],
             redirect: true,
             max_redirects: 5,
             decode_body: false
           ) do
        {:ok, %{status: 200, body: body}} ->
          File.write!(path, body, [:binary])
          IO.puts("    #{byte_size(body)} bytes cached")
          body

        {:ok, %{status: status}} ->
          IO.puts("  ERROR: HTTP #{status} from #{@dataset_url}")
          IO.puts("  You can manually download from: #{@dataset_url}")
          IO.puts("  And place it at: #{path}")
          System.halt(1)

        {:error, reason} ->
          IO.puts("  ERROR: #{inspect(reason)}")
          System.halt(1)
      end
    else
      IO.puts("  Using cached #{@cache_file}")
      File.read!(path)
    end
  end

  defp parse_dailydialog_zip(zip_data) do
    case :zip.unzip(zip_data, [:memory]) do
      {:ok, files} ->
        file_map =
          Map.new(files, fn {name, content} ->
            {to_string(name) |> Path.basename(), to_string(content)}
          end)

        text_content =
          find_file(file_map, "dialogues_text.txt") ||
            find_file(file_map, "train/dialogues_text.txt")

        act_content =
          find_file(file_map, "dialogues_act.txt") ||
            find_file(file_map, "train/dialogues_act.txt")

        emotion_content =
          find_file(file_map, "dialogues_emotion.txt") ||
            find_file(file_map, "train/dialogues_emotion.txt")

        if is_nil(text_content) do
          IO.puts("  WARNING: Could not find dialogues_text.txt in ZIP")
          IO.puts("  ZIP contains: #{Map.keys(file_map) |> Enum.join(", ")}")

          nested = try_nested_zip(files)

          if nested do
            nested
          else
            {[], [], []}
          end
        else
          parse_aligned(text_content, act_content || "", emotion_content || "")
        end

      {:error, reason} ->
        IO.puts("  ERROR: Failed to extract ZIP: #{inspect(reason)}")
        {[], [], []}
    end
  end

  defp try_nested_zip(files) do
    inner_zip =
      Enum.find(files, fn {name, _} ->
        name_str = to_string(name)
        String.ends_with?(name_str, ".zip") and name_str != ""
      end)

    case inner_zip do
      {name, content} ->
        IO.puts("  Found nested ZIP: #{name}, extracting...")
        parse_dailydialog_zip(content)

      nil ->
        nil
    end
  end

  defp find_file(file_map, target) do
    basename = Path.basename(target)

    case Map.get(file_map, basename) do
      nil ->
        Enum.find_value(file_map, fn {key, val} ->
          if String.ends_with?(key, basename), do: val
        end)

      content ->
        content
    end
  end

  defp parse_aligned(text_content, act_content, emotion_content) do
    text_lines = String.split(text_content, "\n", trim: true)
    act_lines = String.split(act_content, "\n", trim: true)
    emotion_lines = String.split(emotion_content, "\n", trim: true)

    max_len = length(text_lines)
    act_lines = pad_lines(act_lines, max_len)
    emotion_lines = pad_lines(emotion_lines, max_len)

    triples =
      [text_lines, act_lines, emotion_lines]
      |> Enum.zip()
      |> Enum.flat_map(fn {text_line, act_line, emo_line} ->
        utts =
          text_line
          |> String.split("__eou__", trim: true)
          |> Enum.map(&String.trim/1)

        acts =
          act_line
          |> String.split(" ", trim: true)
          |> Enum.map(fn s -> String.trim(s) |> String.to_integer() end)

        emos =
          emo_line
          |> String.split(" ", trim: true)
          |> Enum.map(fn s -> String.trim(s) |> String.to_integer() end)

        utts
        |> Enum.with_index()
        |> Enum.flat_map(fn {utt, i} ->
          if byte_size(utt) > 2 do
            [{utt, Enum.at(acts, i), Enum.at(emos, i)}]
          else
            []
          end
        end)
      end)

    {Enum.map(triples, &elem(&1, 0)),
     Enum.map(triples, &elem(&1, 1)),
     Enum.map(triples, &elem(&1, 2))}
  end

  defp pad_lines(lines, target) when length(lines) >= target, do: lines
  defp pad_lines(lines, target), do: lines ++ List.duplicate("", target - length(lines))

  defp build_speech_act_examples(utterances, acts, emotions) do
    act_examples =
      if length(acts) == length(utterances) do
        Enum.zip(utterances, acts)
        |> Enum.map(fn {text, act} -> {text, act, nil} end)
      else
        utterances |> Enum.map(fn text -> {text, nil, nil} end)
      end

    act_examples =
      if length(emotions) == length(utterances) do
        act_examples
        |> Enum.zip(emotions)
        |> Enum.map(fn {{text, act, _}, emo} -> {text, act, emo} end)
      else
        act_examples
      end

    mapped =
      Enum.map(act_examples, fn {text, act, emotion} ->
        speech_act = classify_utterance(text, act, emotion)
        %{"text" => text, "speech_act" => speech_act}
      end)
      |> Enum.filter(fn %{"speech_act" => sa} -> sa != nil end)

    declarative_examples =
      Enum.map(@declarative_seeds, fn text ->
        %{"text" => text, "speech_act" => "declarative"}
      end)

    mapped ++ declarative_examples
  end

  defp classify_utterance(text, act, emotion) do
    lower = String.downcase(text)
    tokens = String.split(lower)
    short? = length(tokens) <= 8

    has_expressive_marker =
      Enum.any?(@expressive_markers, fn marker -> String.contains?(lower, marker) end)

    cond do
      has_expressive_marker and short? -> "expressive"
      act != nil and Map.has_key?(@act_map, act) -> Map.get(@act_map, act)
      emotion != nil and emotion > 0 and short? -> "expressive"
      String.ends_with?(lower, "!") and short? -> "expressive"
      true -> nil
    end
  end


  defp normalize(text) do
    text |> String.downcase() |> String.trim()
  end
end
