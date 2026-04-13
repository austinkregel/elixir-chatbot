defmodule Mix.Tasks.TrainPoincare do
  @shortdoc "Train Poincare embeddings for entity type hierarchies"
  @moduledoc """
  Trains Poincare ball embeddings from hierarchical (IS_A) relations.

  ## Usage

      mix train_poincare [options]

  ## Options

    * `--world` - World ID to scope training (default: "default")
    * `--dim` - Embedding dimension (default: 5)
    * `--epochs` - Number of training epochs (default: 50)
    * `--learning-rate` - Learning rate (default: 0.01)
    * `--verbose` - Print detailed progress
  """

  use Mix.Task

  alias Brain.ML.Poincare.Embeddings
  alias Brain.ML.ModelStore

  @impl true
  def run(args) do
    {opts, _, _} = OptionParser.parse(args,
      strict: [
        world: :string,
        dim: :integer,
        epochs: :integer,
        learning_rate: :float,
        verbose: :boolean,
        publish: :boolean
      ],
      aliases: [w: :world, d: :dim, e: :epochs, v: :verbose]
    )

    Mix.Task.run("app.start")

    world_id = Keyword.get(opts, :world, "default")
    dim = Keyword.get(opts, :dim, 5)
    epochs = Keyword.get(opts, :epochs, 50)
    lr = Keyword.get(opts, :learning_rate, 0.01)
    verbose = Keyword.get(opts, :verbose, false)

    Mix.shell().info("Training Poincare embeddings for world: #{world_id}")

    pairs = load_hierarchy_pairs()

    if Enum.empty?(pairs) do
      Mix.shell().error("No hierarchy pairs found")
      exit({:shutdown, 1})
    end

    Mix.shell().info("Loaded #{length(pairs)} hierarchy pairs")

    if verbose do
      entities = pairs
      |> Enum.flat_map(fn {c, p} -> [c, p] end)
      |> Enum.uniq()
      Mix.shell().info("#{length(entities)} unique entities")
    end

    Mix.shell().info("Training (dim: #{dim}, epochs: #{epochs}, lr: #{lr})...")

    {:ok, embeddings, entity_to_idx, idx_to_entity} = Embeddings.train(pairs,
      dim: dim,
      epochs: epochs,
      learning_rate: lr,
      verbose: true
    )

    output_path = model_path(world_id)
    Embeddings.save(embeddings, entity_to_idx, idx_to_entity, dim, output_path)
    Mix.shell().info("Embeddings saved to #{output_path}")

    if opts[:publish] do
      remote_key = ModelStore.version_prefix() <> "#{world_id}/poincare/embeddings.term"
      ModelStore.publish(output_path, remote_key)
    end

    if verbose do
      evaluate_embeddings(embeddings, entity_to_idx, pairs)
    end
  end

  defp load_hierarchy_pairs do
    entity_pairs = load_entity_type_pairs()
    intent_pairs = load_intent_hierarchy_pairs()
    context_pairs = load_context_follow_up_pairs()
    response_pairs = load_response_frame_pairs()

    (entity_pairs ++ intent_pairs ++ context_pairs ++ response_pairs)
    |> Enum.uniq()
  end

  defp load_entity_type_pairs do
    entity_types_path = resolve_entity_types_path()

    case File.read(entity_types_path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} when is_map(data) ->
            extract_is_a_pairs(data)

          {:ok, entries} when is_list(entries) ->
            Enum.flat_map(entries, fn
              %{"child" => c, "parent" => p} -> [{c, p}]
              %{"entity" => e, "type" => t} -> [{e, t}]
              _ -> []
            end)

          _ -> []
        end

      {:error, _} -> []
    end
  end

  defp load_intent_hierarchy_pairs do
    intent_registry_path = resolve_intent_registry_path()

    case File.read(intent_registry_path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} when is_map(data) ->
            data
            |> Map.keys()
            |> Enum.flat_map(&intent_name_to_pairs/1)
            |> Enum.uniq()

          _ -> []
        end

      {:error, _} -> []
    end
  end

  defp intent_name_to_pairs(intent_name) do
    parts = String.split(intent_name, ".")

    if length(parts) < 2 do
      []
    else
      build_hierarchy_chain(parts, [])
    end
  end

  defp build_hierarchy_chain([_single], acc), do: acc
  defp build_hierarchy_chain(parts, acc) do
    child = Enum.join(parts, ".")
    parent = parts |> Enum.drop(-1) |> Enum.join(".")
    build_hierarchy_chain(Enum.drop(parts, -1), [{child, parent} | acc])
  end

  @context_consumer_patterns %{
    "heating" => "smarthome.heating.",
    "room" => "smarthome.",
    "music-player-control" => "music.player.",
    "play-music" => "music.",
    "search-music" => "music.",
    "schedule" => ".schedule.",
    "weather" => "weather.",
    "device-brightness" => "smarthome.lights.brightness.",
    "device-switch" => "smarthome.lights.switch.",
    "device-volume" => "smarthome.device.volume.",
    "text-context" => "communication.",
    "calendar-context" => "calendar.",
    "reminder-context" => "reminder.",
    "alarm-context" => "alarm.",
    "todo-context" => "todo.",
    "websearch-followup" => "web.",
    "web-search" => "web.",
    "control_lists_web_search" => "web.",
    "news-search" => "news.",
    "newssearch-followup" => "news.",
    "balance" => "account.",
    "spending" => "account.",
    "earning" => "account.",
    "due_date" => "account.",
    "call-context" => "communication.",
    "volume-check" => "smarthome.device.volume.",
    "timer-context" => "timer.",
    "lock" => "smarthome.locks.",
    "unlock" => "smarthome.locks.",
    "open" => "smarthome.locks.",
    "close" => "smarthome.locks."
  }

  defp load_context_follow_up_pairs do
    intent_registry_path = resolve_intent_registry_path()

    case File.read(intent_registry_path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} when is_map(data) ->
            extract_context_pairs(data)

          _ -> []
        end

      {:error, _} -> []
    end
  end

  defp extract_context_pairs(registry) do
    all_intents = Map.keys(registry)

    registry
    |> Enum.flat_map(fn {_intent, meta} ->
      case meta do
        %{"output_contexts" => contexts} when is_list(contexts) ->
          Enum.flat_map(contexts, fn
            %{"name" => name} -> [name]
            _ -> []
          end)

        _ -> []
      end
    end)
    |> Enum.uniq()
    |> Enum.flat_map(fn context_name ->
      pattern = Map.get(@context_consumer_patterns, context_name)

      if pattern do
        consumers = Enum.filter(all_intents, fn intent ->
          cond do
            String.contains?(pattern, ".schedule.") ->
              String.contains?(intent, "schedule")

            true ->
              String.starts_with?(intent, pattern)
          end
        end)

        Enum.map(consumers, fn intent -> {intent, "ctx:" <> context_name} end)
      else
        []
      end
    end)
    |> Enum.uniq()
  end

  defp load_response_frame_pairs do
    domain_pairs = load_domain_frame_pairs()
    template_pairs = load_template_store_pairs()

    (domain_pairs ++ template_pairs) |> Enum.uniq()
  end

  defp load_domain_frame_pairs do
    domains_dir = resolve_brain_priv_path("knowledge/domains")
    intent_registry_path = resolve_intent_registry_path()

    domain_to_intents = case File.read(intent_registry_path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, registry} when is_map(registry) ->
            Enum.reduce(registry, %{}, fn {intent_name, meta}, acc ->
              domain = Map.get(meta, "domain", "")
              Map.update(acc, domain, [intent_name], &[intent_name | &1])
            end)

          _ -> %{}
        end

      {:error, _} -> %{}
    end

    case File.ls(domains_dir) do
      {:ok, files} ->
        files
        |> Enum.filter(&String.ends_with?(&1, ".json"))
        |> Enum.reject(&(&1 == "primitives.json"))
        |> Enum.flat_map(fn filename ->
          domain_name = String.replace_suffix(filename, ".json", "")
          filepath = Path.join(domains_dir, filename)

          case File.read(filepath) do
            {:ok, content} ->
              case Jason.decode(content) do
                {:ok, config} when is_map(config) ->
                  primary_intent = find_primary_intent(domain_name, domain_to_intents)
                  extract_frame_pairs(config, domain_name, primary_intent)

                _ -> []
              end

            {:error, _} -> []
          end
        end)

      {:error, _} -> []
    end
  end

  defp find_primary_intent(domain_name, domain_to_intents) do
    intents = Map.get(domain_to_intents, domain_name, [])

    common_primary = "#{domain_name}.query"
    if common_primary in intents do
      common_primary
    else
      case Enum.sort_by(intents, &String.length/1) do
        [shortest | _] -> shortest
        [] -> domain_name
      end
    end
  end

  defp extract_frame_pairs(config, domain_name, primary_intent) do
    frame_pairs =
      case Map.get(config, "response_frames") do
        frames when is_map(frames) ->
          Enum.map(frames, fn {frame_key, _templates} ->
            {"resp:#{domain_name}.#{frame_key}", primary_intent}
          end)

        _ -> []
      end

    enriched_pairs =
      case Map.get(config, "enriched_response_frames") do
        frames when is_map(frames) ->
          Enum.map(frames, fn {frame_key, _config} ->
            {"resp:#{domain_name}.#{frame_key}", primary_intent}
          end)

        _ -> []
      end

    frame_pairs ++ enriched_pairs
  end

  defp load_template_store_pairs do
    templates_path = resolve_brain_priv_path("response/templates.json")

    case File.read(templates_path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} when is_map(data) ->
            Enum.flat_map(data, fn {intent_name, %{"templates" => templates}} when is_list(templates) ->
              templates
              |> Enum.with_index()
              |> Enum.map(fn {_template, idx} ->
                {"resp:#{intent_name}:#{idx}", intent_name}
              end)

              {_intent_name, _} ->
                []
            end)

          _ -> []
        end

      {:error, _} -> []
    end
  end

  defp extract_is_a_pairs(data) do
    hierarchy = Map.get(data, "type_hierarchy", data)

    Enum.flat_map(hierarchy, fn
      {"description", _} -> []
      {"config", _} -> []

      {parent, %{"subtypes" => subtypes}} when is_list(subtypes) ->
        Enum.map(subtypes, &{&1, parent})

      {parent, children} when is_list(children) ->
        Enum.map(children, &{&1, parent})

      _ -> []
    end)
  end

  defp resolve_entity_types_path do
    resolve_brain_priv_path("analysis/entity_types.json")
  end

  defp resolve_intent_registry_path do
    resolve_brain_priv_path("analysis/intent_registry.json")
  end

  defp resolve_brain_priv_path(relative) do
    candidates = [
      "priv/#{relative}",
      "../../priv/#{relative}",
      "apps/brain/priv/#{relative}"
    ]

    Enum.find(candidates, hd(candidates), &File.exists?/1)
  end

  defp model_path(world_id) do
    priv = :code.priv_dir(:brain) |> to_string()
    Path.join([priv, "ml_models", world_id, "poincare", "embeddings.term"])
  end

  defp evaluate_embeddings(embeddings, entity_to_idx, pairs) do
    alias Brain.ML.Poincare.Distance

    num_entities = map_size(entity_to_idx)

    {total_rank, count} = Enum.reduce(pairs, {0, 0}, fn {child, parent}, {rank_acc, count_acc} ->
      child_idx = Map.get(entity_to_idx, child)
      parent_idx = Map.get(entity_to_idx, parent)

      if child_idx && parent_idx do
        child_emb = embeddings[child_idx]

        distances = for i <- 0..(num_entities - 1) do
          {i, Distance.distance(child_emb, embeddings[i]) |> Nx.to_number()}
        end

        sorted = Enum.sort_by(distances, fn {_i, d} -> d end)
        rank = Enum.find_index(sorted, fn {i, _d} -> i == parent_idx end) + 1

        {rank_acc + rank, count_acc + 1}
      else
        {rank_acc, count_acc}
      end
    end)

    if count > 0 do
      mean_rank = total_rank / count
      Mix.shell().info("Mean rank: #{Float.round(mean_rank, 2)} (lower is better)")
    end
  end
end
