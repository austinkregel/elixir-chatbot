defmodule Mix.Tasks.GenLatticeData do
  @moduledoc """
  Generates the phrase inventory and transition scores for the lattice realizer.

  Processes the intents' response templates (`priv/response/templates.json`),
  domain knowledge files, and response primitives into typed fragments indexed
  by chunk type and primitive type.

  ## Primitive types

  A fragment is addressed by the planner's primitive types
  (`Brain.Response.PrimitiveTypes`): the `type:variant` keys
  `Brain.Response.DiscoursePlanner` plans and `Brain.Response.PhraseLatticeRealizer`
  looks fragments up by. Each section of `primitives.json` is mapped onto one
  of them explicitly (`@primitive_sections`), or named as a pool something
  other than the lattice words; a section or pool the table does not name, or
  one whose shape is not a list of phrases, raises, so a pool added to the file
  is never silently left out of the inventory.

  A template sentence is typed by its chunk type (`@primitive_type_map`). The
  smalltalk domain's frames keyed by an expressive sub-type are social
  acknowledgments of it (`@social_frames`).

  ## Conditions

  A template's `condition` (`Brain.Response.ConditionEvaluator` syntax, such
  as `has_entity:person`) is kept on each of its fragments as `"condition"`,
  and the realizer considers the fragment only where it holds. A primitives
  pool true only in some turns is mapped with its condition
  (`{type, variant, condition}`), which each of its phrases' fragments keeps.
  A condition `Brain.Response.ConditionEvaluator.validate!/1` does not accept
  raises.

  ## Domain frames

  A domain's response frames and enriched response frames are chosen by
  `Brain.Response.Synthesizer.chosen_frame/3` (an enriched frame's `when`
  and `requires_enrichment`, a response frame's slot key). Each of their
  fragments keeps the frame it is a sentence of as `"frame"`
  (`%{"domain", "kind", "key"}`), and the realizer offers it only for a turn
  that rule names that frame for. A frame the synthesizer's compiled
  knowledge does not hold raises.

  ## Usage

      mix gen_lattice_data --world <id> [options]

  ## Options

      --world ID          Required. The world fragments are analyzed in.
      --verbose           Show detailed progress
      --skip-vectorize    Skip prototype vectorization (fast template regen)
      --vectorize-from    Re-vectorize fragments in an existing phrase_inventory.json
  """

  use Mix.Task
  require Logger

  alias Brain.Analysis.FeatureExtractor.ChunkFeatures
  alias Brain.Lattice.FragmentVectorizer
  alias Brain.ML.Tokenizer
  alias Brain.Response.{ConditionEvaluator, Synthesizer}

  @shortdoc "Generate lattice phrase inventory from training data"

  @templates_path "apps/brain/priv/response/templates.json"
  @domains_dir "apps/brain/priv/knowledge/domains"
  @output_dir "apps/brain/priv/ml_models/lattice"
  @config_path "apps/brain/priv/response/system_config.json"

  # A closing ("Goodbye!", "Take care!") answers a farewell, which the planner
  # answers with a social acknowledgment (discourse_patterns.json's
  # "farewell"); `:farewell` is no variant of `:follow_up`.
  @primitive_type_map %{
    "greeting" => {"acknowledgment", "social"},
    "acknowledgment" => {"acknowledgment", "action"},
    "body" => {"content", "enriched"},
    "offer" => {"follow_up", "continuation"},
    "clarification" => {"follow_up", "clarification"},
    "closing" => {"acknowledgment", "social"}
  }

  # The smalltalk domain's response frames are keyed by the expressive
  # sub-type they answer (`Brain.Response.Synthesizer.synthesize_expressive/2`
  # reads them by it): each is a social acknowledgment said when the turn's
  # speech act is of that sub-type, and its "acknowledgment" frame is a general
  # acknowledgment. Each frame's text is one fragment.
  @social_frames %{
    "smalltalk" => %{
      "greeting" => {"acknowledgment", "social", "speech_act:greeting"},
      "farewell" => {"acknowledgment", "social", "speech_act:farewell"},
      "thanks" => {"acknowledgment", "social", "speech_act:thanks"},
      "how_are_you" => {"acknowledgment", "social", "speech_act:how_are_you"},
      "apology" => {"acknowledgment", "social", "speech_act:apology"},
      "acknowledgment" => {"acknowledgment", "general", nil}
    }
  }

  # Each section of primitives.json: the planner's `{type, variant}` its
  # phrases are fragments of (a nil variant for a type without variants), or
  # `{type, variant, condition}` for a pool whose phrases are true only where
  # the condition (`Brain.Response.ConditionEvaluator`) holds, a map of its
  # pools to those, or `{:not_lattice, reason}` for a pool the lattice does
  # not word.
  @primitive_sections %{
    "description" => {:not_lattice, "the file's description, not a pool"},
    "hedges" => %{
      "high_confidence" =>
        {:not_lattice,
         "a hedging primitive is planned only for low confidence (DiscoursePlanner.maybe_insert_hedging/2); " <>
           "these phrases assert confidence"},
      "medium_confidence" => {"hedging", nil},
      "low_confidence" => {"hedging", nil}
    },
    "acknowledgments" => %{
      "affirmative" => {"acknowledgment", "action"},
      "neutral" => {"acknowledgment", "general"}
    },
    "clarification_requests" => %{
      "missing_required" =>
        {:not_lattice, "leads into the slot question Synthesizer.synthesize_clarification/4 words after it"},
      "ambiguous" => {:not_lattice, "leads into the candidates a clarification names after it"},
      "generic" => {"follow_up", "clarification"}
    },
    "service_replies" =>
      {:not_lattice, "a service's error is worded by Synthesizer.synthesize_service_reply/1 (SurfaceRealizer.realize/2)"},
    "service_items" => {:not_lattice, "describes the items a service reply names (Synthesizer.synthesize_service_reply/1)"},
    "transition_phrases" => %{
      "additional_info" => {"transition", nil},
      "contrast" =>
        {:not_lattice, "the planner's transition joins chunks without naming their relation; a contrast needs one"},
      "conclusion" =>
        {:not_lattice, "the planner's transition joins chunks without naming their relation; a conclusion needs one"}
    },
    "uncertainty_markers" => {:not_lattice, "clauses the synthesizer appends to a sentence, not sentences"},
    "correction_invites" => {"follow_up", "correction_invite"},
    "fallback_responses" => {:not_lattice, "no primitive is planned for them"},
    "defer_responses" => {:not_lattice, "Brain's reply to a turn deferred to the user (Synthesizer.get_defer_response/0)"},
    "cannot_respond" => {:not_lattice, "Brain's reply to a turn whose strategy is :cannot_respond"},
    "empty_knowledge" =>
      {:not_lattice, "the self-knowledge reply when nothing is known (Synthesizer.get_empty_knowledge_response/0)"},
    "quality_fallback" => {:not_lattice, "ResponseQuality's reply (Synthesizer.get_quality_fallback/0)"},
    "attunement" => %{
      "empathy" => %{
        "negative" => {"attunement", "empathy"},
        "general" => {"attunement", "empathy"}
      },
      "validation" => {"attunement", "validation"},
      "interest" => {"attunement", "interest"},
      "concern" => {"attunement", "concern"}
    },
    "reflective_patterns" => {"content", "reflective"},
    # An action's result is what the content specifier sets as a
    # `content:action_result` primitive's `:result`, :pending or :incapable
    # (ContentSpecifier.capability_to_result/1); only the incapable pool words
    # a result it sets.
    "action_responses" => %{
      "success" => {:not_lattice, "no content specifier sets an action's result to success"},
      "in_progress" => {:not_lattice, "no content specifier sets an action's result to in progress"},
      "incapable" => {"content", "action_result"},
      "failed" => {:not_lattice, "no content specifier sets an action's result to failed"}
    },
    "learning_acknowledgments" => {"acknowledgment", "learning"},
    "learning_update_acknowledgments" => {"acknowledgment", "learning"},
    "repair_acknowledgments" => {"acknowledgment", "repair"},
    "boundary_frames" => %{
      "general" => {"framing", "boundary"},
      # Its phrases continue into $capability and $alternative as verb phrases
      # ("perhaps I can help by $alternative"); a framing:boundary primitive's
      # capability is a label (:unknown) and its alternative "ask about
      # <domain>" (ContentSpecifier), which they cannot be said with.
      "with_alternative" =>
        {:not_lattice, "its phrases need verb phrases the boundary primitive's content does not hold"}
    },
    # An informative framing opens what the reply tells the user. It is planned
    # before a service's result (enrichable_question and enriched_directive in
    # discourse_patterns.json, which match only a successful enrichment),
    # before what is known about the chunk (factual_question and
    # explanation_request_known, which match only when something is), and
    # before nothing but a follow-up (question_general and default_directive,
    # when neither a service nor anything known gave an answer): a pool that
    # introduces a result is said only after a successful enrichment, one that
    # introduces what is known only where something is, and one that says
    # there is no answer only where there was none.
    "informative_frames" => %{
      "with_result" => {"framing", "informative", "enrichment_success"},
      "with_facts" => {"framing", "informative", "facts_known"},
      "no_answer" => {"framing", "informative", "NOT enrichment_success AND NOT facts_known"}
    },
    # A polar question's answer, planned only for a chunk whose known
    # statements confirm or disconfirm its proposition
    # (polar_question_confirmed and polar_question_disconfirmed), before the
    # statement that does.
    "polar_answers" => %{
      "affirmative" => {"framing", "affirmative"},
      "negative" => {"framing", "negative"}
    },
    # What is known, said as stored: a factual primitive's `fact`, an
    # explanatory one's `explanation` (Brain.Response.ContentSpecifier).
    "fact_statements" => {"content", "factual"},
    "explanations" => {"content", "explanatory"},
    # A self-knowledge narrative's `narrative`, the self-knowledge reply
    # (Synthesizer.synthesize_self_knowledge_response/2).
    "self_knowledge_narratives" => {"content", "narrative"},
    "reframe_patterns" => {"framing", "reframe"},
    "contradiction_responses" => {"contradiction_response", nil},
    "creative_engagement" => {"content", "creative"},
    "elaboration_prompts" => {"follow_up", "elaboration"},
    "context_probes" => {"follow_up", "context_probe"},
    "novelty_responses" => {:not_lattice, "no primitive type words novelty"},
    "continuation_prompts" => {"follow_up", "continuation"}
  }

  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [verbose: :boolean, skip_vectorize: :boolean, vectorize_from: :string, world: :string]
      )

    world_id =
      Keyword.get(opts, :world) ||
        Mix.raise("mix gen_lattice_data needs --world <id>, the world fragments are analyzed in")

    case opts[:vectorize_from] do
      nil -> run_full_build(opts, world_id)
      path -> run_vectorize_from(path, opts, world_id)
    end
  end

  defp run_full_build(opts, world_id) do
    verbose = opts[:verbose] || false
    skip_vectorize? = opts[:skip_vectorize] || false

    Mix.shell().info("Generating lattice phrase inventory...")

    tone_vectors = load_tone_vectors()

    template_fragments = process_response_templates(verbose)
    domain_fragments = process_domain_knowledge(verbose)
    primitive_fragments = process_primitives(verbose)

    all_fragments = template_fragments ++ domain_fragments ++ primitive_fragments
    validate_conditions!(all_fragments)

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
        vectorize_fragments!(all_fragments, verbose, world_id)
      end

    write_inventory_and_transitions(all_fragments, verbose)
  end

  defp run_vectorize_from(path, opts, world_id) do
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

              vectorized = FragmentVectorizer.vectorize_fragments(needs, verbose: verbose, world_id: world_id)
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

  defp vectorize_fragments!(fragments, verbose, world_id) do
    ensure_app_started_for_dimension!()
    FragmentVectorizer.vectorize_fragments(fragments, verbose: verbose, world_id: world_id)
  end

  defp write_inventory_and_transitions(all_fragments, verbose) do
    feature_dim = ensure_app_started_for_dimension!()

    Mix.shell().info("  Total fragments: #{length(all_fragments)}")

    by_chunk_type = Enum.group_by(all_fragments, & &1["chunk_type"])
    fragment_sequences = build_fragment_sequences(all_fragments)
    transition_scores = build_transition_scores(fragment_sequences)
    primitive_index = build_primitive_index(by_chunk_type)

    inventory = %{
      "version" => Brain.Response.PhraseInventory.inventory_version(),
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
        "response_templates" -> "rt"
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

  # Every intent's response templates, the tracked reply store the registry's
  # intents are reconciled with (`mix templates.reconcile`). It holds the
  # DialogFlow replies of data/intents and the templates added since, with the
  # condition each template is said under.
  defp process_response_templates(verbose) do
    templates = @templates_path |> File.read!() |> Jason.decode!()

    fragments =
      templates
      |> Enum.sort_by(&elem(&1, 0))
      |> Enum.flat_map(fn
        {intent, %{"templates" => [_ | _] = entries}} ->
          Enum.flat_map(entries, fn
            %{"text" => text, "condition" => condition} when is_binary(text) and (is_binary(condition) or is_nil(condition)) ->
              if String.trim(text) == "" do
                Mix.raise("gen_lattice_data: #{@templates_path} has a blank template for #{intent}")
              end

              text
              |> segment_template(intent, "response_templates")
              |> Enum.map(&Map.put(&1, "condition", condition))

            other ->
              Mix.raise("gen_lattice_data: #{@templates_path} has a template for #{intent} without text and condition: #{inspect(other)}")
          end)

        {intent, other} ->
          Mix.raise("gen_lattice_data: #{@templates_path} has no templates for #{intent}: #{inspect(other, limit: 5)}")
      end)

    if verbose do
      Mix.shell().info("    templates.json: #{map_size(templates)} intents, #{length(fragments)} fragments")
    end

    fragments
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

                  social_frames = Map.get(@social_frames, domain, %{})

                  frame_fragments =
                    frames
                    |> Enum.flat_map(fn
                      {key, templates} when is_list(templates) and is_map_key(social_frames, key) ->
                        {type, variant, condition} = Map.fetch!(social_frames, key)
                        Enum.map(templates, &social_frame_fragment(&1, domain, type, variant, condition, path))

                      {key, templates} when is_list(templates) ->
                        frame = domain_frame!(domain, :response, key, path)

                        Enum.flat_map(templates, fn template ->
                          template
                          |> segment_template("#{domain}.response", "domain_knowledge")
                          |> Enum.map(&Map.put(&1, "frame", frame))
                        end)

                      {key, other} ->
                        Mix.raise("gen_lattice_data: #{path}'s response frame #{key} is not a list of templates: #{inspect(other)}")
                    end)

                  enriched_fragments =
                    enriched_frames
                    |> Enum.flat_map(fn {name, frame_config} ->
                      frame = domain_frame!(domain, :enriched, name, path)
                      templates = Map.get(frame_config, "templates", [])
                      requires = Map.get(frame_config, "requires_enrichment", [])

                      Enum.flat_map(templates, fn template ->
                        frags = segment_template(template, "#{domain}.enriched", "domain_knowledge")
                        Enum.map(frags, fn frag ->
                          Map.merge(frag, %{
                            "enrichment_fields" => requires,
                            "conditions" => %{"enriched" => requires},
                            "frame" => frame
                          })
                        end)
                      end)
                    end)

                  if verbose and (frame_fragments != [] or enriched_fragments != []) do
                    Mix.shell().info("    #{domain}: #{length(frame_fragments)} frame + #{length(enriched_fragments)} enriched fragments")
                  end

                  frame_fragments ++ enriched_fragments

                {:error, reason} ->
                  Mix.raise("gen_lattice_data: #{path} is not valid JSON: #{inspect(reason)}")
              end

            {:error, reason} ->
              Mix.raise("gen_lattice_data: cannot read #{path}: #{inspect(reason)}")
          end
        end)

      {:error, reason} ->
        Mix.raise("gen_lattice_data: cannot list #{@domains_dir}: #{inspect(reason)}")
    end
  end

  # The domain frame a fragment is a sentence of, as
  # `Brain.Response.Synthesizer.chosen_frame/3` names it: the realizer offers
  # the fragment only for a turn the synthesizer would word from that frame.
  # A frame the synthesizer's compiled knowledge does not hold, or holds
  # malformed, raises.
  defp domain_frame!(domain, kind, key, path) do
    try do
      :ok = Synthesizer.validate_frame!(domain, kind, key)
    rescue
      e in ArgumentError -> Mix.raise("gen_lattice_data: #{path}'s #{kind} frame #{key}: #{Exception.message(e)}")
    end

    %{"domain" => domain, "kind" => Atom.to_string(kind), "key" => key}
  end

  # Every fragment's condition is one the realizer can evaluate; one that is
  # not would never hold, and its fragment would never be said.
  defp validate_conditions!(fragments) do
    Enum.each(fragments, fn frag ->
      try do
        ConditionEvaluator.validate!(frag["condition"])
      rescue
        e in ArgumentError ->
          Mix.raise("gen_lattice_data: the fragment #{inspect(frag["text"])} of #{frag["source_intent"]} has a condition the realizer cannot evaluate: #{Exception.message(e)}")
      end
    end)
  end

  defp social_frame_fragment(text, domain, type, variant, condition, path) when is_binary(text) do
    if String.trim(text) == "" do
      Mix.raise("gen_lattice_data: #{path} has a blank #{domain} frame")
    end

    %{
      "text" => text,
      "chunk_type" => infer_chunk_type_from_primitive(type),
      "primitive_type" => type,
      "primitive_variant" => variant,
      "slots" => extract_slots(text),
      "enrichment_fields" => [],
      "prototype_vector" => [],
      "source_intent" => "#{domain}.response",
      "conditions" => %{},
      "condition" => condition,
      "origin" => "domain_knowledge"
    }
  end

  defp social_frame_fragment(other, domain, _type, _variant, _condition, path) do
    Mix.raise("gen_lattice_data: #{path} has a #{domain} frame that is not a string: #{inspect(other)}")
  end

  # Every section of primitives.json is mapped by `@primitive_sections`: its
  # phrases become fragments of the planner primitive it names, or it is a pool
  # the lattice does not word. A section or pool the table does not name raises.
  defp process_primitives(verbose) do
    path = Path.join(@domains_dir, "primitives.json")
    data = path |> File.read!() |> Jason.decode!()

    case Map.keys(data) -- Map.keys(@primitive_sections) do
      [] -> :ok
      unmapped -> Mix.raise("gen_lattice_data: #{path} has sections no primitive type is mapped for: #{inspect(Enum.sort(unmapped))}")
    end

    fragments =
      data
      |> Enum.sort_by(&elem(&1, 0))
      |> Enum.flat_map(fn {section, pools} ->
        primitive_fragments(Map.fetch!(@primitive_sections, section), pools, [section], path)
      end)

    if verbose do
      Mix.shell().info("    primitives.json: #{length(fragments)} fragments")
    end

    fragments
  end

  defp primitive_fragments({:not_lattice, _reason}, _pools, _keys, _path), do: []

  defp primitive_fragments({type, variant}, phrases, keys, path) when is_list(phrases),
    do: primitive_fragments({type, variant, nil}, phrases, keys, path)

  defp primitive_fragments({type, variant, condition}, phrases, keys, path) when is_list(phrases) do
    Enum.map(phrases, fn
      phrase when is_binary(phrase) ->
        if String.trim(phrase) == "" do
          Mix.raise("gen_lattice_data: #{path}'s #{Enum.join(keys, ".")} has a blank phrase")
        end

        %{
          "text" => phrase,
          "chunk_type" => infer_chunk_type_from_primitive(type),
          "primitive_type" => type,
          "primitive_variant" => variant,
          "slots" => extract_slots(phrase),
          "enrichment_fields" => [],
          "prototype_vector" => [],
          "source_intent" => "primitives." <> Enum.join(keys, "."),
          "conditions" => %{},
          "condition" => condition,
          "origin" => "primitives"
        }

      other ->
        Mix.raise("gen_lattice_data: #{path}'s #{Enum.join(keys, ".")} holds a phrase that is not a string: #{inspect(other)}")
    end)
  end

  defp primitive_fragments(table, pools, keys, path) when is_map(table) and is_map(pools) do
    case Map.keys(pools) -- Map.keys(table) do
      [] -> :ok
      unmapped -> Mix.raise("gen_lattice_data: #{path}'s #{Enum.join(keys, ".")} has pools no primitive type is mapped for: #{inspect(Enum.sort(unmapped))}")
    end

    pools
    |> Enum.sort_by(&elem(&1, 0))
    |> Enum.flat_map(fn {pool, phrases} -> primitive_fragments(Map.fetch!(table, pool), phrases, keys ++ [pool], path) end)
  end

  defp primitive_fragments(mapping, pools, keys, path) do
    Mix.raise(
      "gen_lattice_data: #{path}'s #{Enum.join(keys, ".")} is shaped #{inspect(pools, limit: 3)}, " <>
        "which its mapping #{inspect(mapping)} does not describe"
    )
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
        "condition" => nil,
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

  # The chunk type a primitive's phrase is filed under, by the planner type
  # it is mapped to.
  defp infer_chunk_type_from_primitive("acknowledgment"), do: "acknowledgment"
  defp infer_chunk_type_from_primitive("follow_up"), do: "offer"
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
