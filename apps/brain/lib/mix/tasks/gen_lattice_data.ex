defmodule Mix.Tasks.GenLatticeData do
  @moduledoc """
  Generates the phrase inventory and transition scores for the lattice realizer.

  Processes DialogFlow intent data, domain knowledge files, and response
  primitives into typed fragments indexed by chunk type and primitive type.

  ## Usage

      mix gen_lattice_data [options]

  ## Options

      --verbose           Show detailed progress
      --skip-vectorize    Skip prototype vectorization (fast template regen)
      --vectorize-from    Re-vectorize fragments in an existing phrase_inventory.json
  """

  use Mix.Task
  require Logger

  alias Brain.Analysis.FeatureExtractor.ChunkFeatures
  alias Brain.Lattice.FragmentVectorizer
  alias Brain.ML.Tokenizer

  @shortdoc "Generate lattice phrase inventory from training data"

  @intents_dir "data/intents"
  @domains_dir "apps/brain/priv/knowledge/domains"
  @output_dir "apps/brain/priv/ml_models/lattice"
  @config_path "apps/brain/priv/response/system_config.json"

  @primitive_type_map %{
    "greeting" => {"acknowledgment", "social"},
    "acknowledgment" => {"acknowledgment", "action"},
    "body" => {"content", "enriched"},
    "offer" => {"follow_up", "continuation"},
    "clarification" => {"follow_up", "clarification"},
    "closing" => {"follow_up", "farewell"}
  }

  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [verbose: :boolean, skip_vectorize: :boolean, vectorize_from: :string]
      )

    case opts[:vectorize_from] do
      nil -> run_full_build(opts)
      path -> run_vectorize_from(path, opts)
    end
  end

  defp run_full_build(opts) do
    verbose = opts[:verbose] || false
    skip_vectorize? = opts[:skip_vectorize] || false

    Mix.shell().info("Generating lattice phrase inventory...")

    tone_vectors = load_tone_vectors()

    dialogflow_fragments = process_dialogflow_intents(verbose)
    domain_fragments = process_domain_knowledge(verbose)
    primitive_fragments = process_primitives(verbose)

    all_fragments = dialogflow_fragments ++ domain_fragments ++ primitive_fragments

    all_fragments =
      Enum.map(all_fragments, fn frag ->
        {tone_label, tone_vector} = auto_tag_tone(frag["text"], tone_vectors)
        Map.merge(frag, %{"tone" => tone_label, "tone_vector" => tone_vector})
      end)

    all_fragments = assign_fragment_ids(all_fragments)

    all_fragments =
      if skip_vectorize? do
        Mix.shell().info("  Skipping vectorization (--skip-vectorize)")
        all_fragments
      else
        vectorize_fragments!(all_fragments, verbose)
      end

    write_inventory_and_transitions(all_fragments, verbose)
  end

  defp run_vectorize_from(path, opts) do
    verbose = opts[:verbose] || false

    Mix.shell().info("Re-vectorizing lattice inventory from #{path}...")

    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, inventory} when is_map(inventory) ->
            {flat, _fragments_key} = flatten_inventory_fragments(inventory)
            needs = Enum.filter(flat, &needs_vectorization?/1)

            Mix.shell().info("  #{length(needs)}/#{length(flat)} fragments need vectors")

            if needs == [] do
              Mix.shell().info("  Nothing to vectorize.")
              :ok
            else
              needs = ensure_fragment_ids(needs)

              Mix.Task.run("app.start")

              case FragmentVectorizer.vectorize_fragments(needs, verbose: verbose) do
                {:ok, vectorized} ->
                  updated = merge_vectorized_fragments(flat, vectorized)
                  by_chunk = Enum.group_by(updated, & &1["chunk_type"])

                  feature_dim = ensure_app_started_for_dimension!()

                  new_inventory =
                    Map.merge(inventory, %{
                      "fragments" => by_chunk,
                      "feature_dimension_count" => feature_dim,
                      "generated_at" => DateTime.utc_now() |> DateTime.to_iso8601()
                    })

                  File.write!(path, Jason.encode!(new_inventory, pretty: true))
                  write_scorer_weights_if_needed(Path.dirname(path))
                  Mix.shell().info("  Written: #{path}")
                  :ok

                {:error, reason} ->
                  Mix.shell().error("  Vectorization failed: #{reason}")
                  System.halt(1)
              end
            end

          _ ->
            Mix.shell().error("  Invalid inventory JSON at #{path}")
            System.halt(1)
        end

      {:error, reason} ->
        Mix.shell().error("  Could not read #{path}: #{inspect(reason)}")
        System.halt(1)
    end
  end

  defp flatten_inventory_fragments(%{"fragments" => fragments_map}) when is_map(fragments_map) do
    flat =
      fragments_map
      |> Enum.flat_map(fn {_chunk_type, frags} ->
        if is_list(frags), do: frags, else: []
      end)

    {flat, "fragments"}
  end

  defp flatten_inventory_fragments(_), do: {[], "fragments"}

  defp ensure_fragment_ids(fragments) do
    if Enum.all?(fragments, &Map.get(&1, "id")) do
      fragments
    else
      assign_fragment_ids(fragments)
    end
  end

  defp needs_vectorization?(frag) do
    case Map.get(frag, "prototype_vector") do
      nil -> true
      [] -> true
      _ -> false
    end
  end

  defp merge_vectorized_fragments(all, vectorized) do
    vectorized_by_id = Map.new(vectorized, fn f -> {Map.get(f, "id"), f} end)

    Enum.map(all, fn frag ->
      id = Map.get(frag, "id")

      case id && Map.get(vectorized_by_id, id) do
        %{} = updated -> updated
        _ -> frag
      end
    end)
  end

  defp vectorize_fragments!(fragments, verbose) do
    ensure_app_started_for_dimension!()

    case FragmentVectorizer.vectorize_fragments(fragments, verbose: verbose) do
      {:ok, vectorized} -> vectorized
      {:error, reason} ->
        Mix.shell().error("  Fragment vectorization failed: #{reason}")
        System.halt(1)
    end
  end

  defp write_inventory_and_transitions(all_fragments, verbose) do
    feature_dim = ensure_app_started_for_dimension!()

    Mix.shell().info("  Total fragments: #{length(all_fragments)}")

    by_chunk_type = Enum.group_by(all_fragments, & &1["chunk_type"])
    fragment_sequences = build_fragment_sequences(all_fragments)
    transition_scores = build_transition_scores(fragment_sequences)
    primitive_index = build_primitive_index(by_chunk_type)

    inventory = %{
      "version" => 1,
      "generated_at" => DateTime.utc_now() |> DateTime.to_iso8601(),
      "feature_dimension_count" => feature_dim,
      "tone_dimension_count" => 10,
      "fragments" => by_chunk_type,
      "primitive_index" => primitive_index
    }

    File.mkdir_p!(@output_dir)

    inventory_path = Path.join(@output_dir, "phrase_inventory.json")
    File.write!(inventory_path, Jason.encode!(inventory, pretty: true))
    Mix.shell().info("  Written: #{inventory_path}")

    transition_data = %{
      "version" => 1,
      "generated_at" => DateTime.utc_now() |> DateTime.to_iso8601(),
      "bigrams" => transition_scores
    }

    transition_path = Path.join(@output_dir, "transition_scores.json")
    File.write!(transition_path, Jason.encode!(transition_data, pretty: true))
    Mix.shell().info("  Written: #{transition_path}")

    write_scorer_weights_if_needed(@output_dir)

    if verbose do
      with_vec = Enum.count(all_fragments, fn f -> Map.get(f, "prototype_vector", []) != [] end)
      Mix.shell().info("  Fragments with prototype_vector: #{with_vec}/#{length(all_fragments)}")
    end

    Mix.shell().info("Lattice data generation complete!")
  end

  defp assign_fragment_ids(fragments) do
    fragments
    |> Enum.group_by(fn f -> {Map.get(f, "origin", "unknown"), Map.get(f, "source_intent")} end)
    |> Enum.flat_map(fn {{origin, intent}, group} ->
      prefix = fragment_id_prefix(origin, intent)

      group
      |> Enum.with_index(1)
      |> Enum.map(fn {frag, idx} ->
        Map.put(frag, "id", "#{prefix}/#{idx}")
      end)
    end)
  end

  defp fragment_id_prefix(origin, intent) do
    origin_short =
      case origin do
        "dialogflow" -> "df"
        "domain_knowledge" -> "dk"
        "primitives" -> "prim"
        other when is_binary(other) -> String.replace(other, " ", "_")
        _ -> "unk"
      end

    safe_intent =
      (intent || "unknown")
      |> to_string()
      |> String.replace(".", "_")

    "#{origin_short}_#{safe_intent}"
  end

  defp ensure_app_started_for_dimension! do
    Application.put_env(:brain, :skip_ml_init, true)
    Mix.Task.run("app.start")
    wait_for_type_hierarchy!(5_000)

    dim = ChunkFeatures.vector_dimension()

    if not is_integer(dim) or dim <= 0 do
      Mix.raise("ChunkFeatures.vector_dimension/0 returned invalid dimension: #{inspect(dim)}")
    end

    dim
  end

  defp wait_for_type_hierarchy!(timeout_ms) do
    started = System.monotonic_time(:millisecond)
    do_wait_type_hierarchy(started, timeout_ms)
  end

  defp do_wait_type_hierarchy(_started, timeout_ms) when timeout_ms <= 0 do
    unless Brain.Analysis.TypeHierarchy.ready?() do
      Mix.raise(
        "TypeHierarchy not ready. Cannot compute feature_dimension_count for lattice inventory."
      )
    end
  end

  defp do_wait_type_hierarchy(started, timeout_ms) do
    if Brain.Analysis.TypeHierarchy.ready?() do
      :ok
    else
      elapsed = System.monotonic_time(:millisecond) - started

      if elapsed >= timeout_ms do
        Mix.raise(
          "TypeHierarchy not ready after #{timeout_ms}ms. " <>
            "Cannot compute feature_dimension_count for lattice inventory."
        )
      else
        Process.sleep(50)
        do_wait_type_hierarchy(started, timeout_ms)
      end
    end
  end

  defp write_scorer_weights_if_needed(output_dir) do
    dim = ensure_app_started_for_dimension!()
    weights_path = Path.join(output_dir, "scorer_weights.json")

    needs_write =
      case File.read(weights_path) do
        {:ok, content} ->
          case Jason.decode(content) do
            {:ok, %{"weights" => w}} when is_list(w) -> length(w) != dim
            _ -> true
          end

        {:error, _} ->
          true
      end

    if needs_write do
      weights = %{
        "version" => 1,
        "weights" => List.duplicate(1.0, dim),
        "feature_names" => []
      }

      File.write!(weights_path, Jason.encode!(weights, pretty: true))
      Mix.shell().info("  Written: #{weights_path} (default weights, dim=#{dim})")
    end
  end

  defp process_dialogflow_intents(verbose) do
    case File.ls(@intents_dir) do
      {:ok, files} ->
        usersays_files = Enum.filter(files, &String.ends_with?(&1, "_usersays_en.json"))

        if verbose do
          Mix.shell().info("  Found #{length(usersays_files)} usersays files")
        end

        usersays_files
        |> Enum.flat_map(fn usersays_file ->
          intent_name = String.replace(usersays_file, "_usersays_en.json", "")
          intent_file = "#{intent_name}.json"

          templates = load_intent_templates(Path.join(@intents_dir, intent_file))

          if templates == [] do
            []
          else
            if verbose do
              Mix.shell().info("    #{intent_name}: #{length(templates)} templates")
            end

            Enum.flat_map(templates, fn template ->
              segment_template(template, intent_name, "dialogflow")
            end)
          end
        end)

      {:error, _} ->
        Mix.shell().info("  Warning: #{@intents_dir} not found")
        []
    end
  end

  defp load_intent_templates(path) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} ->
            responses = Map.get(data, "responses", [])

            responses
            |> Enum.flat_map(fn response ->
              messages = Map.get(response, "messages", [])

              messages
              |> Enum.flat_map(fn message ->
                case message do
                  %{"speech" => speeches} when is_list(speeches) ->
                    Enum.filter(speeches, &is_binary/1)

                  %{"speech" => speech} when is_binary(speech) ->
                    [speech]

                  _ ->
                    []
                end
              end)
            end)
            |> Enum.filter(&(String.trim(&1) != ""))
            |> Enum.uniq()

          _ -> []
        end

      {:error, _} -> []
    end
  end

  defp process_domain_knowledge(verbose) do
    case File.ls(@domains_dir) do
      {:ok, files} ->
        json_files = Enum.filter(files, &String.ends_with?(&1, ".json"))
        json_files = Enum.reject(json_files, &(&1 == "primitives.json"))

        if verbose do
          Mix.shell().info("  Processing #{length(json_files)} domain files")
        end

        Enum.flat_map(json_files, fn file ->
          path = Path.join(@domains_dir, file)

          case File.read(path) do
            {:ok, content} ->
              case Jason.decode(content) do
                {:ok, data} ->
                  domain = Map.get(data, "domain", Path.rootname(file))
                  frames = Map.get(data, "response_frames", %{})
                  enriched_frames = Map.get(data, "enriched_response_frames", %{})

                  frame_fragments =
                    frames
                    |> Enum.flat_map(fn {_key, templates} ->
                      templates = if is_list(templates), do: templates, else: []
                      Enum.flat_map(templates, fn template ->
                        segment_template(template, "#{domain}.response", "domain_knowledge")
                      end)
                    end)

                  enriched_fragments =
                    enriched_frames
                    |> Enum.flat_map(fn {_name, frame_config} ->
                      templates = Map.get(frame_config, "templates", [])
                      requires = Map.get(frame_config, "requires_enrichment", [])

                      Enum.flat_map(templates, fn template ->
                        frags = segment_template(template, "#{domain}.enriched", "domain_knowledge")
                        Enum.map(frags, fn frag ->
                          Map.merge(frag, %{
                            "enrichment_fields" => requires,
                            "conditions" => %{"enriched" => requires}
                          })
                        end)
                      end)
                    end)

                  if verbose and (frame_fragments != [] or enriched_fragments != []) do
                    Mix.shell().info("    #{domain}: #{length(frame_fragments)} frame + #{length(enriched_fragments)} enriched fragments")
                  end

                  frame_fragments ++ enriched_fragments

                _ -> []
              end

            {:error, _} -> []
          end
        end)

      {:error, _} ->
        Mix.shell().info("  Warning: #{@domains_dir} not found")
        []
    end
  end

  defp process_primitives(verbose) do
    path = Path.join(@domains_dir, "primitives.json")

    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} ->
            fragments =
              data
              |> Enum.flat_map(fn {prim_type, variants_or_phrases} ->
                cond do
                  is_map(variants_or_phrases) ->
                    Enum.flat_map(variants_or_phrases, fn {variant_or_level, phrases} ->
                      phrases = if is_list(phrases), do: phrases, else: []
                      Enum.map(phrases, fn phrase ->
                        chunk_type = infer_chunk_type_from_primitive(prim_type)
                        %{
                          "text" => phrase,
                          "chunk_type" => chunk_type,
                          "primitive_type" => prim_type,
                          "primitive_variant" => variant_or_level,
                          "slots" => extract_slots(phrase),
                          "enrichment_fields" => [],
                          "prototype_vector" => [],
                          "source_intent" => "primitives.#{prim_type}.#{variant_or_level}",
                          "conditions" => %{},
                          "origin" => "primitives"
                        }
                      end)
                    end)

                  is_list(variants_or_phrases) ->
                    Enum.map(variants_or_phrases, fn phrase ->
                      chunk_type = infer_chunk_type_from_primitive(prim_type)
                      %{
                        "text" => phrase,
                        "chunk_type" => chunk_type,
                        "primitive_type" => prim_type,
                        "primitive_variant" => nil,
                        "slots" => extract_slots(phrase),
                        "enrichment_fields" => [],
                        "prototype_vector" => [],
                        "source_intent" => "primitives.#{prim_type}",
                        "conditions" => %{},
                        "origin" => "primitives"
                      }
                    end)

                  true ->
                    []
                end
              end)

            if verbose do
              Mix.shell().info("    primitives.json: #{length(fragments)} fragments")
            end

            fragments

          _ -> []
        end

      {:error, _} -> []
    end
  end

  defp segment_template(template, intent_name, origin) when is_binary(template) do
    sentences = split_sentences_preserving_slots(template)

    sentences
    |> Enum.map(fn sentence ->
      chunk_type = classify_chunk_heuristic(sentence)
      {prim_type, prim_variant} = Map.get(@primitive_type_map, chunk_type, {"content", "enriched"})
      slots = extract_slots(sentence)
      enrichment = extract_enrichment_fields(slots)

      %{
        "text" => sentence,
        "chunk_type" => chunk_type,
        "primitive_type" => prim_type,
        "primitive_variant" => prim_variant,
        "slots" => slots,
        "enrichment_fields" => enrichment,
        "prototype_vector" => [],
        "source_intent" => intent_name,
        "conditions" => if(enrichment != [], do: %{"enriched" => enrichment}, else: %{}),
        "origin" => origin
      }
    end)
  end

  defp split_sentences_preserving_slots(text) do
    sentences =
      text
      |> Tokenizer.split_sentences()
      |> Enum.map(fn sent -> String.trim(sent.text) end)
      |> Enum.filter(&(&1 != ""))

    case sentences do
      [] -> [text]
      result -> result
    end
  end

  @greeting_words MapSet.new(~w(hello hi hey welcome howdy greetings))
  @closing_words MapSet.new(~w(goodbye bye farewell))
  @offer_words MapSet.new(~w(anything else help assist))
  @ack_words MapSet.new(~w(sure okay done got understood acknowledged right absolutely))

  defp classify_chunk_heuristic(sentence) do
    tokens = Tokenizer.tokenize_words(String.downcase(sentence))
    first = List.first(tokens) || ""
    token_set = MapSet.new(tokens)

    cond do
      first in @greeting_words or
        (first == "good" and Enum.at(tokens, 1) in ~w(morning afternoon evening)) ->
        "greeting"

      first in @closing_words or
        (first == "have" and MapSet.member?(token_set, "day")) or
        (first == "take" and Enum.at(tokens, 1) == "care") ->
        "closing"

      not MapSet.disjoint?(token_set, @offer_words) and
        (MapSet.member?(token_set, "else") or MapSet.member?(token_set, "can")) ->
        "offer"

      (first in ~w(which what where when how) and length(tokens) < 10) or
        (String.ends_with?(sentence, "?") and length(tokens) < 8) ->
        "clarification"

      first in @ack_words or
        (first == "i" and Enum.at(tokens, 1) in ~w(understand understood)) ->
        "acknowledgment"

      true ->
        "body"
    end
  end

  @entity_slots MapSet.new(~w(location device room action color temperature time duration event))

  defp extract_slots(text) do
    text
    |> Tokenizer.tokenize()
    |> Enum.filter(fn t -> String.starts_with?(t.text, "$") end)
    |> Enum.map(fn t -> String.trim_leading(t.text, "$") end)
    |> Enum.uniq()
  end

  defp extract_enrichment_fields(slots) do
    Enum.reject(slots, &MapSet.member?(@entity_slots, &1))
  end

  defp infer_chunk_type_from_primitive("acknowledgment"), do: "acknowledgment"
  defp infer_chunk_type_from_primitive("hedges"), do: "body"
  defp infer_chunk_type_from_primitive("attunement"), do: "body"
  defp infer_chunk_type_from_primitive("closers"), do: "closing"
  defp infer_chunk_type_from_primitive("openers"), do: "greeting"
  defp infer_chunk_type_from_primitive("follow_ups"), do: "offer"
  defp infer_chunk_type_from_primitive(_), do: "body"

  @hedge_words MapSet.new(~w(maybe perhaps possibly might could probably))

  defp auto_tag_tone(text, tone_vectors) when is_binary(text) do
    tokens = Tokenizer.tokenize(text)
    token_texts = Enum.map(tokens, & &1.text)
    lower_tokens = Enum.map(token_texts, &String.downcase/1)

    exclamation_count = Enum.count(token_texts, &(&1 == "!"))
    question_count = Enum.count(token_texts, &(&1 == "?"))
    word_count = length(Enum.filter(lower_tokens, &(String.length(&1) > 1)))

    has_hedge = Enum.any?(lower_tokens, &MapSet.member?(@hedge_words, &1))

    positive = min(1.0, exclamation_count * 0.3)
    negative = 0.0
    neutral = if question_count > 0, do: 0.7, else: max(0.0, 1.0 - positive)
    confidence = if has_hedge, do: 0.3, else: 0.6
    polarity = positive

    arousal = min(1.0, exclamation_count * 0.3)
    warmth = if exclamation_count > 0, do: 0.6, else: 0.4
    formality = if word_count > 5, do: 0.5, else: 0.4
    playfulness = if exclamation_count > 1, do: 0.4, else: 0.2
    directness = if word_count < 4, do: 0.8, else: 0.5

    estimated_vector = [positive, negative, neutral, confidence, polarity,
                        arousal, warmth, formality, playfulness, directness]

    {tone_label, _} = find_nearest_tone(estimated_vector, tone_vectors)
    {tone_label, estimated_vector}
  end

  defp auto_tag_tone(_, _), do: {"neutral", List.duplicate(0.0, 10)}

  defp find_nearest_tone(vector, tone_vectors) when is_map(tone_vectors) and map_size(tone_vectors) > 0 do
    tone_vectors
    |> Enum.map(fn {name, tv} ->
      sim = cosine_similarity(vector, tv)
      {name, sim}
    end)
    |> Enum.max_by(fn {_, sim} -> sim end, fn -> {"neutral", 0.0} end)
  end

  defp find_nearest_tone(_, _), do: {"neutral", 0.0}

  defp cosine_similarity(a, b) when is_list(a) and is_list(b) and length(a) == length(b) do
    dot = Enum.zip(a, b) |> Enum.reduce(0.0, fn {x, y}, sum -> sum + x * y end)
    mag_a = :math.sqrt(Enum.reduce(a, 0.0, fn x, sum -> sum + x * x end))
    mag_b = :math.sqrt(Enum.reduce(b, 0.0, fn x, sum -> sum + x * x end))
    if mag_a == 0.0 or mag_b == 0.0, do: 0.0, else: dot / (mag_a * mag_b)
  end

  defp cosine_similarity(_, _), do: 0.0

  defp load_tone_vectors do
    case File.read(@config_path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, %{"tone" => %{"tone_vectors" => vectors}}} -> vectors
          _ -> %{}
        end
      {:error, _} -> %{}
    end
  end

  defp build_fragment_sequences(all_fragments) do
    all_fragments
    |> Enum.group_by(& &1["source_intent"])
    |> Enum.map(fn {_intent, frags} ->
      Enum.sort_by(frags, fn f ->
        chunk_order(f["chunk_type"])
      end)
      |> Enum.map(& &1["text"])
    end)
    |> Enum.filter(&(length(&1) >= 2))
  end

  defp chunk_order("greeting"), do: 0
  defp chunk_order("acknowledgment"), do: 1
  defp chunk_order("body"), do: 2
  defp chunk_order("offer"), do: 3
  defp chunk_order("clarification"), do: 4
  defp chunk_order("closing"), do: 5
  defp chunk_order(_), do: 2

  defp build_transition_scores(sequences) do
    bigram_counts =
      sequences
      |> Enum.flat_map(fn seq ->
        pairs = Enum.zip(seq, tl(seq))
        Enum.flat_map(pairs, fn {a, b} ->
          a_tokens = last_n_tokens(a, 3)
          b_tokens = first_n_tokens(b, 3)

          case {List.last(a_tokens), List.first(b_tokens)} do
            {nil, _} -> []
            {_, nil} -> []
            {la, fb} -> ["#{String.downcase(la)}|#{String.downcase(fb)}"]
          end
        end)
      end)
      |> Enum.frequencies()

    max_count = Enum.max(Map.values(bigram_counts), fn -> 1 end)

    Map.new(bigram_counts, fn {key, count} ->
      {key, Float.round(count / max_count, 4)}
    end)
  end

  defp last_n_tokens(text, n) do
    text
    |> Tokenizer.tokenize_words()
    |> Enum.take(-n)
  end

  defp first_n_tokens(text, n) do
    text
    |> Tokenizer.tokenize_words()
    |> Enum.take(n)
  end

  defp build_primitive_index(by_chunk_type) do
    by_chunk_type
    |> Enum.flat_map(fn {chunk_type, frags} ->
      frags
      |> Enum.with_index()
      |> Enum.map(fn {frag, idx} ->
        prim_type = frag["primitive_type"] || "content"
        prim_variant = frag["primitive_variant"] || "_"
        key = "#{prim_type}:#{prim_variant}"
        ref = "#{chunk_type}/#{idx}"
        {key, ref}
      end)
    end)
    |> Enum.group_by(fn {key, _ref} -> key end, fn {_key, ref} -> ref end)
  end
end
