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
      generate_speech_act_gold(intent_examples, preview?, limit)
    end

    if both? or sentiment_only? do
      generate_sentiment_gold(intent_examples, preview?, limit)
    end

    IO.puts("")
  end

  defp generate_speech_act_gold(intent_examples, preview?, limit) do
    IO.puts("\n" <> String.duplicate("=", 50))
    IO.puts("SPEECH ACT GOLD STANDARD")
    IO.puts(String.duplicate("=", 50))

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
      path = EvaluationStore.gold_standard_path("speech_act")
      File.mkdir_p!(Path.dirname(path))
      File.write!(path, Jason.encode!(speech_act_examples, pretty: true))
      IO.puts("\nWritten to: #{path}")
    end
  end

  defp generate_sentiment_gold(intent_examples, preview?, limit) do
    IO.puts("\n" <> String.duplicate("=", 50))
    IO.puts("SENTIMENT GOLD STANDARD")
    IO.puts(String.duplicate("=", 50))

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
      path = EvaluationStore.gold_standard_path("sentiment")
      File.mkdir_p!(Path.dirname(path))
      File.write!(path, Jason.encode!(sentiment_examples, pretty: true))
      IO.puts("\nWritten to: #{path}")
    end
  end

  # Infer speech act from intent name
  defp infer_speech_act(intent) when is_binary(intent) do
    intent_lower = String.downcase(intent)

    cond do
      # Directive: requests, commands, questions
      String.contains?(intent_lower, ~w(query check set control play search
        turn switch volume brightness temperature open close lock unlock
        remind alarm timer schedule navigate)) ->
        "directive"

      # Expressive: emotions, social acts
      String.contains?(intent_lower, ~w(greeting bye thank sorry appraisal
        likes hate love good bad beautiful annoying user. emotion)) ->
        "expressive"

      # Assertive: statements of fact
      String.contains?(intent_lower, ~w(news fact tell explain describe
        meta self_knowledge info)) ->
        "assertive"

      # Commissive: promises, offers
      String.contains?(intent_lower, ~w(help offer promise will can)) ->
        "commissive"

      # Default based on prefix
      String.starts_with?(intent_lower, "smalltalk") ->
        "expressive"

      String.starts_with?(intent_lower, "meta") ->
        "assertive"

      true ->
        # Most intents are directive (user asking for something)
        "directive"
    end
  end

  defp infer_speech_act(_), do: "unknown"

  # Infer sentiment from intent name
  defp infer_sentiment(intent) when is_binary(intent) do
    intent_lower = String.downcase(intent)

    cond do
      # Positive sentiment
      String.contains?(intent_lower, ~w(likes good beautiful thanks love
        great nice awesome happy excited clever)) ->
        "positive"

      # Negative sentiment
      String.contains?(intent_lower, ~w(bad annoying hate sorry sad angry
        frustrated disappointed upset)) ->
        "negative"

      # Most intents are neutral (informational requests)
      true ->
        "neutral"
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
