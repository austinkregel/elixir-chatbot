defmodule Mix.Tasks.GenerateGoldStandard do
  @shortdoc "Generate gold standard data for sentiment and speech_act evaluation"
  @moduledoc """
  Generates gold standard evaluation data for sentiment and speech act
  classification based on existing intent training data.

  ## Usage

      mix generate_gold_standard                    # Generate all (sentiment + speech_act)
      mix generate_gold_standard --speech-act       # Generate speech_act only
      mix generate_gold_standard --sentiment        # Generate sentiment only
      mix generate_gold_standard --preview          # Preview without writing
      mix generate_gold_standard --limit 50         # Max examples per category
      mix generate_gold_standard --force            # Overwrite existing gold standard files

  ## How it works

  Speech acts are derived from intent patterns:
  - directive: intents with query, check, set, control, play, search
  - expressive: smalltalk intents (greetings, appraisals, emotions)
  - assertive: meta, news, factual intents
  - commissive: help, offer intents

  Sentiment is derived from intent patterns:
  - positive: likes, good, beautiful, thanks
  - negative: bad, annoying, hate, sorry
  - neutral: query, check, factual, meta
  """

  use Mix.Task

  alias Brain.ML.EvaluationStore

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    preview? = "--preview" in args
    force? = "--force" in args
    speech_act_only? = "--speech-act" in args
    sentiment_only? = "--sentiment" in args
    limit = parse_limit(args)

    both? = not speech_act_only? and not sentiment_only?

    # Load intent gold standard
    intent_examples = EvaluationStore.load_gold_standard("intent")

    if intent_examples == [] do
      IO.puts("\nNo intent gold standard found. Run `mix migrate_gold_standard` first.\n")
      exit(:normal)
    end

    IO.puts("\nLoaded #{length(intent_examples)} intent examples")

    if both? or speech_act_only? do
      generate_speech_act_gold(intent_examples, preview?, force?, limit)
    end

    if both? or sentiment_only? do
      generate_sentiment_gold(intent_examples, preview?, force?, limit)
    end

    IO.puts("")
  end

  defp generate_speech_act_gold(intent_examples, preview?, force?, limit) do
    IO.puts("\n" <> String.duplicate("=", 50))
    IO.puts("SPEECH ACT GOLD STANDARD")
    IO.puts(String.duplicate("=", 50))

    path = EvaluationStore.gold_standard_path("speech_act")

    if not force? and not preview? and File.exists?(path) do
      existing = EvaluationStore.load_gold_standard("speech_act")

      IO.puts(
        "\nExisting gold standard found with #{length(existing)} examples. Use --force to overwrite."
      )

      return_early()
    else
      speech_act_examples =
        intent_examples
        |> Enum.map(fn ex ->
          %{
            "text" => ex["text"],
            "speech_act" => infer_speech_act(ex["intent"])
          }
        end)
        |> Enum.reject(fn ex -> ex["speech_act"] == "unknown" end)
        |> maybe_limit_per_category(limit, "speech_act")

      # Count by category
      by_category = Enum.group_by(speech_act_examples, & &1["speech_act"])

      IO.puts("\nDistribution:")

      Enum.each(by_category, fn {cat, examples} ->
        IO.puts("  #{cat}: #{length(examples)}")
      end)

      IO.puts("\nTotal: #{length(speech_act_examples)} examples")

      if preview? do
        IO.puts("\nSample examples:")

        speech_act_examples
        |> Enum.take(5)
        |> Enum.each(fn ex ->
          IO.puts("  [#{ex["speech_act"]}] #{String.slice(ex["text"], 0, 50)}")
        end)

        IO.puts("\nRun without --preview to write to gold_standard.json")
      else
        File.mkdir_p!(Path.dirname(path))
        File.write!(path, Jason.encode!(speech_act_examples, pretty: true))
        IO.puts("\nWritten to: #{path}")
      end
    end
  end

  defp generate_sentiment_gold(intent_examples, preview?, force?, limit) do
    IO.puts("\n" <> String.duplicate("=", 50))
    IO.puts("SENTIMENT GOLD STANDARD")
    IO.puts(String.duplicate("=", 50))

    path = EvaluationStore.gold_standard_path("sentiment")

    if not force? and not preview? and File.exists?(path) do
      existing = EvaluationStore.load_gold_standard("sentiment")

      IO.puts(
        "\nExisting gold standard found with #{length(existing)} examples. Use --force to overwrite."
      )

      return_early()
    else
      sentiment_examples =
        intent_examples
        |> Enum.map(fn ex ->
          %{
            "text" => ex["text"],
            "sentiment" => infer_sentiment(ex["intent"])
          }
        end)
        |> Enum.reject(fn ex -> ex["sentiment"] == "unknown" end)
        |> maybe_limit_per_category(limit, "sentiment")

      # Count by category
      by_category = Enum.group_by(sentiment_examples, & &1["sentiment"])

      IO.puts("\nDistribution:")

      Enum.each(by_category, fn {cat, examples} ->
        IO.puts("  #{cat}: #{length(examples)}")
      end)

      IO.puts("\nTotal: #{length(sentiment_examples)} examples")

      if preview? do
        IO.puts("\nSample examples:")

        sentiment_examples
        |> Enum.take(5)
        |> Enum.each(fn ex ->
          IO.puts("  [#{ex["sentiment"]}] #{String.slice(ex["text"], 0, 50)}")
        end)

        IO.puts("\nRun without --preview to write to gold_standard.json")
      else
        File.mkdir_p!(Path.dirname(path))
        File.write!(path, Jason.encode!(sentiment_examples, pretty: true))
        IO.puts("\nWritten to: #{path}")
      end
    end
  end

  # Infer speech act from intent name using IntentRegistry metadata
  defp infer_speech_act(intent) when is_binary(intent) do
    case Brain.Analysis.IntentRegistry.category(intent) do
      nil -> "directive"
      category -> to_string(category)
    end
  end

  defp infer_speech_act(_), do: "unknown"

  # Sentiment domains/categories known to carry sentiment
  @positive_domains MapSet.new(~w(likes appraisal.positive love appreciation gratitude))
  @negative_domains MapSet.new(~w(hate appraisal.negative frustration complaint))

  # Infer sentiment from intent using IntentRegistry domain metadata
  defp infer_sentiment(intent) when is_binary(intent) do
    domain =
      case Brain.Analysis.IntentRegistry.domain(intent) do
        nil -> nil
        d -> to_string(d)
      end

    category =
      case Brain.Analysis.IntentRegistry.category(intent) do
        nil -> nil
        c -> to_string(c)
      end

    intent_tokens = Brain.ML.Tokenizer.tokenize_normalized(intent)
    intent_token_set = MapSet.new(intent_tokens)

    cond do
      domain != nil and MapSet.member?(@positive_domains, domain) -> "positive"
      domain != nil and MapSet.member?(@negative_domains, domain) -> "negative"
      not MapSet.disjoint?(intent_token_set, MapSet.new(~w(likes good beautiful thanks love great nice awesome happy))) -> "positive"
      not MapSet.disjoint?(intent_token_set, MapSet.new(~w(bad annoying hate sorry sad angry frustrated disappointed))) -> "negative"
      category == "expressive" -> "neutral"
      true -> "neutral"
    end
  end

  defp infer_sentiment(_), do: "unknown"

  defp maybe_limit_per_category(examples, nil, _field), do: examples

  defp maybe_limit_per_category(examples, limit, field) do
    examples
    |> Enum.group_by(& &1[field])
    |> Enum.flat_map(fn {_cat, cat_examples} ->
      Enum.take(Enum.shuffle(cat_examples), limit)
    end)
  end

  defp return_early, do: :skipped

  defp parse_limit(args) do
    case Enum.find_index(args, &(&1 == "--limit")) do
      nil ->
        nil

      idx ->
        case Enum.at(args, idx + 1) do
          nil -> nil
          val -> String.to_integer(val)
        end
    end
  end
end
