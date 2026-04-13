defmodule Brain.Analysis.IntentPromoter do
  @moduledoc "Handles promotion of novel intent candidates to training data and intent registry.\n\nWhen a candidate is approved:\n1. Writes training example to data/intents/*_usersays_en.json\n2. Updates apps/brain/priv/analysis/intent_registry.json (for new intents)\n3. Triggers model retraining\n4. Reloads the classifier model\n"

  alias Brain.ML
  require Logger

  alias Brain.Analysis.Types.IntentReviewCandidate
  alias ML.{Trainer, IntentClassifierSimple}

  @doc "Promotes a candidate to training data and/or intent registry.\n\n## Options\n  - `:domain` - Domain for new intent (required if promotion_action is :new_intent)\n  - `:category` - Category for new intent (required if promotion_action is :new_intent)\n  - `:speech_act` - Speech act type for new intent (required if promotion_action is :new_intent)\n"
  def promote(%IntentReviewCandidate{} = candidate, opts \\ []) do
    Logger.info("Promoting intent candidate",
      id: candidate.id,
      action: candidate.promotion_action,
      intent: candidate.promoted_to_intent
    )

    case candidate.promotion_action do
      :variation ->
        promote_as_variation(candidate)

      :new_intent ->
        promote_as_new_intent(candidate, opts)

      nil ->
        Logger.warning("No promotion action specified", id: candidate.id)
        {:error, :no_promotion_action}
    end
  end

  defp promote_as_variation(%IntentReviewCandidate{promoted_to_intent: intent_name, text: text})
       when is_binary(intent_name) and intent_name != "" do
    intent_file = get_intent_file_path(intent_name)

    case append_training_example(intent_file, text) do
      :ok ->
        Logger.info("Added training example", intent: intent_name, file: intent_file)
        retrain_and_reload()
        {:ok, :variation_added}

      {:error, reason} ->
        Logger.error("Failed to add training example",
          intent: intent_name,
          reason: inspect(reason)
        )

        {:error, reason}
    end
  end

  defp promote_as_variation(_) do
    {:error, :invalid_intent_name}
  end

  defp promote_as_new_intent(
         %IntentReviewCandidate{promoted_to_intent: intent_name, text: text},
         opts
       )
       when is_binary(intent_name) and intent_name != "" do
    domain = Keyword.get(opts, :domain) || ""
    category = Keyword.get(opts, :category) || "directive"
    speech_act = Keyword.get(opts, :speech_act) || "request_information"

    if domain == "" do
      Logger.warning("New intent promotion requires domain", intent: intent_name)
      {:error, :domain_required}
    else
      intent_file = get_intent_file_path(intent_name)

      case append_training_example(intent_file, text) do
        :ok ->
          Logger.info("Created training example for new intent",
            intent: intent_name,
            file: intent_file
          )

          case update_intent_registry(intent_name, domain, category, speech_act) do
            :ok ->
              Logger.info("Updated intent registry", intent: intent_name)
              retrain_and_reload()
              {:ok, :new_intent_created}

            {:error, reason} ->
              Logger.error("Failed to update intent registry",
                intent: intent_name,
                reason: inspect(reason)
              )

              {:error, reason}
          end

        {:error, reason} ->
          Logger.error("Failed to create training example",
            intent: intent_name,
            reason: inspect(reason)
          )

          {:error, reason}
      end
    end
  end

  defp promote_as_new_intent(_, _) do
    {:error, :invalid_intent_name}
  end

  defp append_training_example(file_path, text) do
    file_path |> Path.dirname() |> File.mkdir_p!()

    existing_data =
      if File.exists?(file_path) do
        case File.read(file_path) do
          {:ok, content} ->
            case Jason.decode(content) do
              {:ok, data} when is_list(data) -> data
              {:ok, %{"userSays" => user_says}} when is_list(user_says) -> user_says
              _ -> []
            end

          {:error, _} ->
            []
        end
      else
        []
      end

    new_example = %{
      "id" => generate_id(),
      "data" => [
        %{
          "text" => text,
          "userDefined" => false
        }
      ],
      "isTemplate" => false,
      "count" => 0,
      "lang" => "en",
      "updated" => 0
    }

    updated_data = existing_data ++ [new_example]
    json_content = Jason.encode!(updated_data, pretty: true)

    case File.write(file_path, json_content) do
      :ok -> :ok
      {:error, reason} -> {:error, reason}
    end
  end

  defp update_intent_registry(intent_name, domain, category, speech_act) do
    registry_path = Brain.priv_path("analysis/intent_registry.json")
    registry_path |> Path.dirname() |> File.mkdir_p!()

    registry =
      case File.read(registry_path) do
        {:ok, content} ->
          case Jason.decode(content) do
            {:ok, data} when is_map(data) -> data
            _ -> %{}
          end

        {:error, _} ->
          %{}
      end

    new_entry = %{
      "description" => "Intent created from novel candidate review",
      "domain" => domain,
      "category" => category,
      "speech_act" => speech_act,
      "required" => [],
      "optional" => [],
      "defaults" => %{},
      "entity_mappings" => %{},
      "clarification_templates" => %{}
    }

    updated_registry = Map.put(registry, intent_name, new_entry)
    json_content = Jason.encode!(updated_registry, pretty: true)

    case File.write(registry_path, json_content) do
      :ok ->
        Logger.info("Intent registry updated - restart required for IntentRegistry to reload")
        :ok

      {:error, reason} ->
        {:error, {:file_write, reason}}
    end
  end

  defp retrain_and_reload do
    Logger.info("Starting model retraining after intent promotion...")

    Brain.AtlasIntegration.async(fn ->
      case Trainer.train_and_save() do
        {:ok, stats} ->
          Logger.info("Model retraining completed", stats)

          case IntentClassifierSimple.load_models() do
            {:ok, _} ->
              Logger.info("Intent classifier reloaded after promotion")

            {:error, reason} ->
              Logger.warning("Failed to reload intent classifier", reason: inspect(reason))
          end

        {:error, reason} ->
          Logger.error("Model retraining failed", reason: inspect(reason))
      end
    end)
  end

  defp get_intent_file_path(intent_name) do
    filename = "#{intent_name}_usersays_en.json"
    base_path = Application.get_env(:brain, :ml)[:training_data_path] || "data"
    Path.join([base_path, "intents", filename])
  end

  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end
end
