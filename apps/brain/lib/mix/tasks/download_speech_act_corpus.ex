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
    # Legal / judicial
    "I hereby declare this meeting adjourned.",
    "I sentence you to five years in prison.",
    "The defendant is found guilty.",
    "The defendant is found not guilty.",
    "The case is dismissed.",
    "I overrule the objection.",
    "I sustain the objection.",
    "I declare a mistrial.",
    "I plead guilty.",
    "I plead not guilty.",
    "I waive my right to an attorney.",
    "You are granted asylum.",
    "You are under arrest.",
    "I find you in contempt of court.",
    "Bail is set at fifty thousand dollars.",
    "The jury has reached a verdict.",
    "I order a recess until tomorrow morning.",
    "The witness is excused.",
    "I move to strike that from the record.",
    "The evidence is admitted.",
    "The evidence is inadmissible.",
    "I order the defendant to pay restitution.",
    "Probation is granted for two years.",
    # Employment / organizational
    "You are fired.",
    "You're hired.",
    "You are hereby promoted to senior manager.",
    "You are suspended without pay.",
    "I accept your resignation.",
    "I resign from my position effective immediately.",
    "I appoint you as the new chairman.",
    "You are excused from jury duty.",
    "I nominate you for the position.",
    "Your employment is terminated effective today.",
    "I assign you to the marketing team.",
    "You are transferred to the London office.",
    "I designate you as team lead.",
    "Your contract is renewed for another year.",
    "I relieve you of your duties.",
    "You are on probation for ninety days.",
    # Ceremonies / formal
    "I now pronounce you husband and wife.",
    "I name this ship the Queen Mary.",
    "I christen this building the Community Center.",
    "I officially open this ceremony.",
    "I declare the games open.",
    "I declare the winner of the competition.",
    "I award you the medal of honor.",
    "I confer upon you the degree of Doctor of Philosophy.",
    "I bestow this honor upon the recipient.",
    "The ribbon is cut and the bridge is open.",
    "I dedicate this monument to the fallen soldiers.",
    "I inaugurate the new library wing.",
    # Meetings / governance
    "Court is now in session.",
    "This meeting is called to order.",
    "The motion is carried.",
    "The motion is denied.",
    "I declare a state of emergency.",
    "The bill is vetoed.",
    "Class dismissed.",
    "I authorize the release of funds.",
    "I move to table the discussion.",
    "The amendment is adopted.",
    "The amendment is rejected.",
    "I call the vote.",
    "The session is adjourned until next week.",
    "I second the motion.",
    "The quorum is established.",
    "I yield the floor.",
    "The chair recognizes the senator.",
    "Debate is closed.",
    "The resolution is passed.",
    "I invoke cloture.",
    # Permissions / access
    "I grant you permission to proceed.",
    "I revoke your access privileges.",
    "I ban you from this establishment.",
    "Your application is approved.",
    "Your application is denied.",
    "Access granted.",
    "Access denied.",
    "Permission denied.",
    "I lift the restriction on travel.",
    "The embargo is imposed.",
    "The embargo is lifted.",
    "I suspend your license.",
    "Your license is reinstated.",
    "I clear you for takeoff.",
    "Landing permission granted.",
    # Agreements / contracts
    "The contract is terminated.",
    "I dissolve this partnership.",
    "I annul this agreement.",
    "I reject your proposal.",
    "I declare this project complete.",
    "I certify this document as authentic.",
    "I withdraw my objection.",
    "War is declared.",
    "I surrender.",
    "I forfeit the match.",
    "I declare bankruptcy.",
    "The deal is finalized.",
    "I void this transaction.",
    "The terms are accepted.",
    "I nullify the agreement.",
    "I ratify the treaty.",
    # Everyday performatives
    "I promise to be there on time.",
    "I bet you ten dollars it will rain.",
    "I dare you to try it.",
    "I challenge you to a rematch.",
    "I name my dog Buddy.",
    "I call this meeting to order.",
    "I quit.",
    "I volunteer for the task.",
    "I opt out of the program.",
    "I formally request a review.",
    "I lodge a complaint.",
    "I retract my previous statement.",
    "I concede the point.",
    "I acknowledge receipt of the package.",
    "I confirm your reservation.",
    "Your reservation is cancelled.",
    "I approve the budget.",
    "The project is greenlit.",
    "I veto this proposal.",
    "I endorse this candidate.",
    "I second that nomination.",
    "I absolve you of responsibility.",
    "I excuse you from the meeting.",
    "I welcome you to the team.",
    "I dismiss the class early today.",
    "Checkmate.",
    "Game over.",
    "I call a timeout.",
    "The match is postponed.",
    "The game is cancelled due to weather.",
    "I default on the payment.",
    "I recuse myself from this case.",
    "I table this discussion for now.",
    "I rescind my earlier offer.",
    "I bequeath my estate to my children.",
    "The will is executed as written.",
    "I renounce my citizenship.",
    "I invoke my fifth amendment rights.",
    # Informal performatives
    "You're grounded.",
    "You're off the team.",
    "You're in charge while I'm gone.",
    "I'm calling it a day.",
    "I'm out.",
    "Deal.",
    "No deal.",
    "I'm in.",
    "Count me out.",
    "I pass.",
    "Sold to the highest bidder.",
    "Going once, going twice, sold.",
    "I fold.",
    "I raise you fifty.",
    "I'm all in.",
    "Tag, you're it.",
    "Time out.",
    "I claim this seat.",
    "I reserve the right to change my mind.",
    "I take back what I said.",
    "I owe you one.",
    "We're even.",
    "I forgive you.",
    "I accept your apology.",
    "I apologize for the inconvenience.",
    "I swear to tell the truth.",
    "I give you my word."
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
