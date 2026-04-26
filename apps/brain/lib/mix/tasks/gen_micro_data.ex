defmodule Mix.Tasks.GenMicroData do
  @moduledoc """
  Generate **training data** for the six ChunkProfile axis micro-classifiers from
  `apps/brain/priv/evaluation/intent/gold_standard.json`.

  Writes `data/classifiers/<name>.json` for:

  - `intent_full` — fine-grained intent label (e.g. `weather.query`, `code.explain`)
  - `intent_domain` — consolidated topical domain from intent prefixes
  - `tense_class` — past / present / future / atemporal (POS + token heuristics)
  - `aspect_class` — simple / progressive / perfect / perfect_progressive
  - `urgency` — low / normal / high / critical
  - `certainty_level` — committed / tentative / hedged / speculative
  - `coarse_semantic_class` — coarse semantic bucket (action, quality, manner, …)

  Heuristic labels are noisy by design; spot-check distributions and edge cases after
  each run. Then train models with `mix train_micro` (or stage 9 of `mix train`).

  ## Output formats

  Five axis classifiers (`intent_domain`, `tense_class`, `aspect_class`,
  `urgency`, `certainty_level`) are **feature-vector classifiers** under
  `Brain.ML.FeatureVectorClassifier`. Their training records have the
  shape:

      %{"feature_vector" => [float], "label" => string}

  `coarse_semantic_class` remains a **text-based** classifier (it
  operates on single words, not chunks). Its records are:

      %{"text" => string, "label" => string}

  Feature-vector generation requires booting the Brain app and running
  `Brain.Analysis.Pipeline.process/1` on every gold_standard entry to
  compute the dense feature vector. This is slow but is a one-time cost
  per regen.

  ## Usage

      mix gen_micro_data [options]

  ## Options

      --only NAME    Generate a single classifier file (e.g. `intent_domain`)
      --stats        Print label histograms without writing files
  """

  use Mix.Task
  require Logger

  alias Brain.Analysis.FeatureExtractor
  alias Brain.Analysis.Pipeline

  @shortdoc "Generate training data for new micro-classifiers from gold_standard.json"

  @feature_vector_classifiers ~w(intent_full intent_domain tense_class aspect_class urgency certainty_level)
  @text_classifiers ~w(coarse_semantic_class)
  @new_classifiers @feature_vector_classifiers ++ @text_classifiers

  @domain_consolidation %{
    "account" => "account",
    "alarm" => "reminder",
    "analysis" => "knowledge",
    "calendar" => "calendar",
    "code" => "code",
    "communication" => "communication",
    "date" => "time",
    "dialog" => "smalltalk",
    "display" => "smarthome",
    "knowledge" => "knowledge",
    "meta" => "meta",
    "music" => "music",
    "navigation" => "navigation",
    "news" => "knowledge",
    "payment" => "account",
    "reminder" => "reminder",
    "search" => "knowledge",
    "smalltalk" => "smalltalk",
    "smarthome" => "smarthome",
    "statement" => "smalltalk",
    "status" => "meta",
    "time" => "time",
    "timer" => "reminder",
    "todo" => "calendar",
    "weather" => "weather",
    "web" => "knowledge"
  }

  @modal_urgency_words MapSet.new(~w(
    urgent immediately now asap emergency hurry quick quickly rush
    please help important critical right_away stat
  ))

  @hedge_words MapSet.new(~w(
    maybe perhaps probably possibly might could potentially
    somewhat kind_of sort_of apparently seemingly supposedly
    arguably conceivably presumably
  ))

  @certainty_words MapSet.new(~w(
    definitely certainly absolutely surely clearly obviously
    undoubtedly always never must certainly indeed truly
    without_a_doubt for_sure no_doubt
  ))

  @impl Mix.Task
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [only: :string, stats: :boolean],
        aliases: [o: :only, s: :stats]
      )

    gold_standard = load_gold_standard()
    Mix.shell().info("Loaded #{length(gold_standard)} examples from gold_standard.json")

    names =
      case opts[:only] do
        nil -> @new_classifiers
        name -> [name]
      end

    if Enum.any?(names, &(&1 in @feature_vector_classifiers)) do
      check_lexicon_data()
    end

    enriched = maybe_enrich_with_feature_vectors(gold_standard, names)

    Enum.each(names, fn name ->
      data = generate_data(name, enriched)

      if opts[:stats] do
        show_stats(name, data)
      else
        write_data(name, data)
        show_stats(name, data)
      end
    end)
  end

  # Boots the Brain app (if not already started) and computes a dense feature
  # vector for every gold_standard entry by running the full analysis
  # pipeline. This is the per-example cost of having feature-vector axis
  # classifiers instead of token-bag TF-IDF ones; it runs once per regen.
  defp maybe_enrich_with_feature_vectors(entries, names) do
    needs_vectors? = Enum.any?(names, &(&1 in @feature_vector_classifiers))

    if needs_vectors? do
      Mix.Task.run("app.start")

      Mix.shell().info(
        "Computing feature vectors for #{length(entries)} entries " <>
          "(max_concurrency=#{System.schedulers_online()})..."
      )

      started_at = System.monotonic_time(:millisecond)

      enriched =
        entries
        |> Task.async_stream(
          &enrich_entry/1,
          max_concurrency: System.schedulers_online(),
          timeout: 60_000,
          on_timeout: :kill_task,
          ordered: true
        )
        |> Enum.map(fn
          {:ok, entry} -> entry
          {:exit, reason} -> %{"feature_vector" => nil, "__pipeline_error" => inspect(reason)}
        end)

      elapsed_ms = System.monotonic_time(:millisecond) - started_at
      with_vec = Enum.count(enriched, &has_feature_vector?/1)

      Mix.shell().info(
        "Feature vector computation complete: #{with_vec}/#{length(enriched)} " <>
          "successful in #{elapsed_ms} ms"
      )

      enriched
    else
      entries
    end
  end

  # Uses Pipeline.analyze_chunk/2 (single-chunk, no graph/memory side effects)
  # rather than Pipeline.process/1 (full pipeline with persistence). Each
  # gold-standard entry already corresponds to a single chunk.
  defp enrich_entry(%{"text" => text} = entry) when is_binary(text) do
    try do
      analysis = Pipeline.analyze_chunk(text)
      {feature_vector, _word_feats} = FeatureExtractor.extract(analysis)

      tokens = extract_tokens(analysis)
      pos_tags = Map.get(analysis, :pos_tags, [])

      entry
      |> Map.put("feature_vector", feature_vector)
      |> Map.put("tokens", tokens)
      |> Map.put("pos_tags", pos_tags)
    rescue
      e ->
        entry
        |> Map.put("feature_vector", nil)
        |> Map.put("tokens", nil)
        |> Map.put("pos_tags", nil)
        |> Map.put("__pipeline_error", Exception.message(e))
    catch
      kind, reason ->
        entry
        |> Map.put("feature_vector", nil)
        |> Map.put("tokens", nil)
        |> Map.put("pos_tags", nil)
        |> Map.put("__pipeline_error", "#{kind}: #{inspect(reason)}")
    end
  end

  defp enrich_entry(entry) do
    entry
    |> Map.put("feature_vector", nil)
    |> Map.put("tokens", nil)
    |> Map.put("pos_tags", nil)
  end

  defp extract_tokens(analysis) do
    case Map.get(analysis, :pos_tags, []) do
      tags when is_list(tags) ->
        Enum.map(tags, fn
          {token, _tag} -> token
          token when is_binary(token) -> token
          _ -> ""
        end)

      _ ->
        []
    end
  end

  defp has_feature_vector?(%{"feature_vector" => v}) when is_list(v) and length(v) > 0, do: true
  defp has_feature_vector?(_), do: false

  defp generate_data("intent_full", entries) do
    entries
    |> Enum.filter(&has_feature_vector?/1)
    |> Enum.filter(fn entry -> is_binary(entry["intent"]) and entry["intent"] != "" end)
    |> Enum.map(fn entry ->
      %{"feature_vector" => entry["feature_vector"], "label" => entry["intent"]}
    end)
  end

  defp generate_data("intent_domain", entries) do
    entries
    |> Enum.filter(&has_feature_vector?/1)
    |> Enum.map(fn entry ->
      raw_domain = entry["intent"] |> String.split(".") |> List.first()
      label = Map.get(@domain_consolidation, raw_domain, "other")
      %{"feature_vector" => entry["feature_vector"], "label" => label}
    end)
  end

  defp generate_data("tense_class", entries) do
    entries
    |> Enum.filter(&has_feature_vector?/1)
    |> Enum.map(fn entry ->
      label = derive_tense(entry["text"], entry["pos_tags"], entry["tokens"])
      %{"feature_vector" => entry["feature_vector"], "label" => label}
    end)
  end

  defp generate_data("aspect_class", entries) do
    entries
    |> Enum.filter(&has_feature_vector?/1)
    |> Enum.map(fn entry ->
      label = derive_aspect(entry["text"], entry["pos_tags"], entry["tokens"])
      %{"feature_vector" => entry["feature_vector"], "label" => label}
    end)
  end

  defp generate_data("urgency", entries) do
    entries
    |> Enum.filter(&has_feature_vector?/1)
    |> Enum.map(fn entry ->
      label = derive_urgency(entry["text"], entry["tokens"])
      %{"feature_vector" => entry["feature_vector"], "label" => label}
    end)
  end

  defp generate_data("certainty_level", entries) do
    entries
    |> Enum.filter(&has_feature_vector?/1)
    |> Enum.map(fn entry ->
      label = derive_certainty(entry["text"], entry["tokens"])
      %{"feature_vector" => entry["feature_vector"], "label" => label}
    end)
  end

  defp generate_data("coarse_semantic_class", entries) do
    Enum.map(entries, fn entry ->
      label = derive_coarse_semantic(entry["intent"], entry["pos_tags"])
      %{"text" => entry["text"], "label" => label}
    end)
  end

  defp generate_data(name, _entries) do
    Mix.shell().error("Unknown classifier: #{name}")
    []
  end

  # -- Tense derivation -------------------------------------------------------

  defp derive_tense(_text, pos_tags, tokens) do
    lower_tokens = Enum.map(tokens || [], &String.downcase/1)
    tag_pairs = Enum.zip(lower_tokens, pos_tags || [])

    cond do
      has_past_markers?(lower_tokens, tag_pairs) -> "past"
      has_future_markers?(lower_tokens) -> "future"
      has_present_markers?(lower_tokens, tag_pairs) -> "present"
      all_nouns_or_adj?(pos_tags) -> "atemporal"
      true -> "present"
    end
  end

  defp has_past_markers?(tokens, tag_pairs) do
    past_aux = MapSet.new(~w(was were had did been))
    past_verbs_ending = fn t -> String.ends_with?(t, "ed") end

    Enum.any?(tokens, &MapSet.member?(past_aux, &1)) or
      Enum.any?(tag_pairs, fn
        {token, "VERB"} -> past_verbs_ending.(token)
        _ -> false
      end)
  end

  defp has_future_markers?(tokens) do
    future_markers = MapSet.new(~w(will shall gonna going_to))

    Enum.any?(tokens, &MapSet.member?(future_markers, &1)) or
      contains_going_to?(tokens)
  end

  defp contains_going_to?(tokens) do
    tokens
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.any?(fn [a, b] -> a == "going" and b == "to" end)
  end

  defp has_present_markers?(tokens, tag_pairs) do
    present_aux = MapSet.new(~w(am is are do does can could would should may might))

    Enum.any?(tokens, &MapSet.member?(present_aux, &1)) or
      Enum.any?(tag_pairs, fn
        {_token, "AUX"} -> true
        {_token, "VERB"} -> true
        _ -> false
      end)
  end

  defp all_nouns_or_adj?(pos_tags) do
    nominal = MapSet.new(~w(NOUN PROPN ADJ DET ADP NUM))
    Enum.all?(pos_tags || [], &MapSet.member?(nominal, &1))
  end

  # -- Aspect derivation -------------------------------------------------------

  defp derive_aspect(_text, pos_tags, tokens) do
    lower_tokens = Enum.map(tokens || [], &String.downcase/1)
    tag_pairs = Enum.zip(lower_tokens, pos_tags || [])

    cond do
      has_perfect_progressive?(lower_tokens) -> "perfect_progressive"
      has_perfect?(lower_tokens) -> "perfect"
      has_progressive?(lower_tokens, tag_pairs) -> "progressive"
      true -> "simple"
    end
  end

  defp has_perfect_progressive?(tokens) do
    has_have = Enum.any?(tokens, &(&1 in ~w(have has had)))
    has_been = Enum.member?(tokens, "been")
    has_ing = Enum.any?(tokens, &String.ends_with?(&1, "ing"))
    has_have and has_been and has_ing
  end

  defp has_perfect?(tokens) do
    has_have = Enum.any?(tokens, &(&1 in ~w(have has had)))
    has_past_participle = Enum.any?(tokens, &String.ends_with?(&1, "ed")) or
                          Enum.any?(tokens, &(&1 in ~w(been done gone seen taken given known)))
    has_have and has_past_participle
  end

  defp has_progressive?(tokens, _tag_pairs) do
    progressive_aux = MapSet.new(~w(am is are was were))
    has_aux = Enum.any?(tokens, &MapSet.member?(progressive_aux, &1))
    has_ing = Enum.any?(tokens, &String.ends_with?(&1, "ing"))
    has_aux and has_ing
  end

  # -- Urgency derivation ------------------------------------------------------

  defp derive_urgency(_text, tokens) do
    lower_tokens = (tokens || []) |> Enum.map(&String.downcase/1) |> MapSet.new()

    urgency_count = MapSet.intersection(lower_tokens, @modal_urgency_words) |> MapSet.size()

    safe_tokens = tokens || []
    has_imperative = Enum.any?(safe_tokens, fn _ -> false end)
    has_exclamation = Enum.any?(safe_tokens, &(&1 == "!"))

    cond do
      urgency_count >= 2 or (urgency_count >= 1 and has_exclamation) -> "critical"
      urgency_count >= 1 -> "high"
      has_imperative -> "normal"
      true -> "low"
    end
  end

  # -- Certainty derivation ----------------------------------------------------

  defp derive_certainty(_text, tokens) do
    lower_tokens = (tokens || []) |> Enum.map(&String.downcase/1) |> MapSet.new()

    hedge_count = MapSet.intersection(lower_tokens, @hedge_words) |> MapSet.size()
    certainty_count = MapSet.intersection(lower_tokens, @certainty_words) |> MapSet.size()

    has_question = Enum.any?(tokens || [], &(&1 == "?"))
    has_conditional = Enum.any?(tokens || [], fn t ->
      String.downcase(t) in ~w(if unless whether)
    end)

    cond do
      certainty_count >= 1 -> "committed"
      hedge_count >= 2 or (hedge_count >= 1 and has_conditional) -> "speculative"
      hedge_count >= 1 -> "hedged"
      has_question -> "tentative"
      true -> "committed"
    end
  end

  # -- Coarse semantic class derivation ----------------------------------------

  @action_domains MapSet.new(~w(
    alarm calendar code communication navigation reminder
    smarthome timer todo music payment
  ))

  @knowledge_domains MapSet.new(~w(
    knowledge analysis news search web
  ))

  @person_intents MapSet.new(~w(
    smalltalk.greetings smalltalk.introduction smalltalk.name
    account.profile communication.call communication.message
  ))

  defp derive_coarse_semantic(intent, pos_tags) do
    domain = intent |> String.split(".") |> List.first()
    full_intent_prefix = intent |> String.split(".") |> Enum.take(2) |> Enum.join(".")

    primary_pos = (pos_tags || []) |> Enum.frequencies() |> Enum.max_by(fn {_k, v} -> v end, fn -> {"NOUN", 0} end) |> elem(0)

    cond do
      MapSet.member?(@person_intents, full_intent_prefix) -> "person"
      domain in ["weather", "time", "date"] -> "quality"
      MapSet.member?(@action_domains, domain) -> "action"
      MapSet.member?(@knowledge_domains, domain) -> "abstract"
      domain == "meta" or domain == "status" -> "abstract"
      domain == "smalltalk" and primary_pos in ["ADJ", "ADV"] -> "quality"
      domain == "smalltalk" -> "manner"
      domain == "statement" -> "thing"
      primary_pos == "VERB" -> "action"
      primary_pos in ["ADJ", "ADV"] -> "quality"
      true -> "thing"
    end
  end

  # -- Output ------------------------------------------------------------------

  defp write_data(name, data) do
    dir = output_dir()
    File.mkdir_p!(dir)
    path = Path.join(dir, "#{name}.json")
    json = Jason.encode!(data, pretty: true)
    File.write!(path, json)
    Mix.shell().info("Wrote #{length(data)} examples to #{path}")
  end

  defp show_stats(name, data) do
    labels = data |> Enum.map(& &1["label"]) |> Enum.frequencies() |> Enum.sort_by(&elem(&1, 1), :desc)

    Mix.shell().info("\n#{name} (#{length(data)} total):")

    Enum.each(labels, fn {label, count} ->
      pct = Float.round(count / max(length(data), 1) * 100, 1)
      Mix.shell().info("  #{String.pad_trailing(label, 25)} #{count} (#{pct}%)")
    end)
  end

  defp load_gold_standard do
    path =
      case :code.priv_dir(:brain) do
        {:error, _} -> "apps/brain/priv/evaluation/intent/gold_standard.json"
        priv -> Path.join(priv, "evaluation/intent/gold_standard.json")
      end

    case File.read(path) do
      {:ok, json} ->
        case Jason.decode(json) do
          {:ok, data} -> data
          {:error, reason} ->
            Mix.raise("Failed to parse gold_standard.json: #{inspect(reason)}")
        end

      {:error, reason} ->
        Mix.raise("Failed to read #{path}: #{inspect(reason)}")
    end
  end

  defp output_dir do
    priv_dir =
      case :code.priv_dir(:brain) do
        {:error, _} -> "apps/brain/priv"
        dir -> to_string(dir)
      end

    umbrella_root =
      case File.read_link(priv_dir) do
        {:ok, link_target} ->
          parent = Path.dirname(priv_dir)
          real_priv = Path.join(parent, link_target) |> Path.expand()
          Path.join(real_priv, "../../..") |> Path.expand()

        {:error, _} ->
          Path.join(priv_dir, "../../../../..") |> Path.expand()
      end

    Path.join(umbrella_root, "data/classifiers")
  end

  defp check_lexicon_data do
    priv_dir =
      case :code.priv_dir(:brain) do
        {:error, _} -> "apps/brain/priv"
        dir -> to_string(dir)
      end

    wordnet_dir = Path.join(priv_dir, "wordnet")
    conceptnet_path = Path.join(priv_dir, "lexicon/conceptnet.term")

    wordnet_ok = File.dir?(wordnet_dir) and File.exists?(Path.join(wordnet_dir, "wn_s.pl"))
    conceptnet_ok = File.exists?(conceptnet_path)

    warnings = []
    warnings = if wordnet_ok, do: warnings, else: warnings ++ ["WordNet not found — lexical domains, supersenses, selectional preferences will be empty. Run: mix download_wordnet"]
    warnings = if conceptnet_ok, do: warnings, else: warnings ++ ["ConceptNet not found — ConceptNet edge fingerprint dims (218-229) will be empty. Run: mix ingest_lexicon"]

    unless warnings == [] do
      Mix.shell().info("")
      Mix.shell().info("╔══════════════════════════════════════════════════════════════╗")
      Mix.shell().info("║  LEXICON DATA PREFLIGHT WARNING                             ║")
      Mix.shell().info("╚══════════════════════════════════════════════════════════════╝")
      Enum.each(warnings, fn w -> Mix.shell().info("  ! #{w}") end)
      Mix.shell().info("")
      Mix.shell().info("  Run `mix setup_lexicon` to download and ingest all lexicon data.")
      Mix.shell().info("")
    end
  end
end
