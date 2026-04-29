defmodule Brain.ML.TrainingData.SourceDescriptors do
  @moduledoc """
  Static registry of every training-data source file the system owns.

  Each descriptor declares a source's identity, file path, record schema,
  category (for sidebar grouping), and editability tag. Adding a new source
  file to the system means adding one entry here — everything else (catalog,
  diagnostics, studio UI) picks it up automatically.
  """

  @type tag :: :authoring | :registry | :gazetteer | :build_artifact | :external_corpus
  @type record_kind ::
          :intent_example
          | :text_classifier_row
          | :fv_classifier_row
          | :registry_entry
          | :slot_schema_entry
          | :speech_act_map_entry
          | :entity_type_entry
          | :gazetteer_entry
          | :kg_negative
          | :csv_row

  @type t :: %{
          id: atom(),
          label: String.t(),
          category: String.t(),
          tag: tag(),
          record_kind: record_kind(),
          path: String.t(),
          description: String.t(),
          upstream_of: [atom()],
          generated_by: String.t() | nil
        }

  defp brain_priv(rel), do: Path.join(Application.app_dir(:brain, "priv"), rel)
  defp data_path(rel), do: Path.join(Application.get_env(:brain, :ml)[:training_data_path] || "data", rel)

  @spec all() :: [t()]
  def all do
    [
      # ── Intent classification ──────────────────────────────────────────
      %{
        id: :intent_gold,
        label: "Intent Gold Standard",
        category: "Intent Classification",
        tag: :authoring,
        record_kind: :intent_example,
        path: brain_priv("evaluation/intent/gold_standard.json"),
        description: "Canonical intent training corpus (text + intent label)",
        upstream_of: [:intent_full_fv, :intent_domain_fv],
        generated_by: nil
      },
      %{
        id: :intent_registry,
        label: "Intent Registry",
        category: "Intent Classification",
        tag: :registry,
        record_kind: :registry_entry,
        path: brain_priv("analysis/intent_registry.json"),
        description: "Per-intent metadata: domain, slots, speech act, entity mappings",
        upstream_of: [],
        generated_by: nil
      },
      %{
        id: :intent_full_fv,
        label: "intent_full (build artifact)",
        category: "Intent Classification",
        tag: :build_artifact,
        record_kind: :fv_classifier_row,
        path: data_path("classifiers/intent_full.json"),
        description: "Feature-vector training rows for :intent_full classifier",
        upstream_of: [],
        generated_by: "mix gen_micro_data"
      },
      %{
        id: :intent_domain_fv,
        label: "intent_domain (build artifact)",
        category: "Intent Classification",
        tag: :build_artifact,
        record_kind: :fv_classifier_row,
        path: data_path("classifiers/intent_domain.json"),
        description: "Feature-vector training rows for :intent_domain classifier",
        upstream_of: [],
        generated_by: "mix gen_micro_data"
      },

      # ── Speech Act ─────────────────────────────────────────────────────
      %{
        id: :speech_act_gold,
        label: "Speech Act Gold Standard",
        category: "Speech Act",
        tag: :authoring,
        record_kind: :intent_example,
        path: brain_priv("evaluation/speech_act/gold_standard.json"),
        description: "Speech act training corpus (text + speech_act label)",
        upstream_of: [],
        generated_by: nil
      },
      %{
        id: :speech_act_intent_map,
        label: "Speech Act → Intent Map",
        category: "Speech Act",
        tag: :registry,
        record_kind: :speech_act_map_entry,
        path: brain_priv("analysis/speech_act_intent_map.json"),
        description: "Maps speech-act subtypes/categories to canonical intents",
        upstream_of: [],
        generated_by: nil
      },

      # ── Sentiment ──────────────────────────────────────────────────────
      %{
        id: :sentiment_gold,
        label: "Sentiment Gold Standard",
        category: "Sentiment",
        tag: :authoring,
        record_kind: :intent_example,
        path: brain_priv("evaluation/sentiment/gold_standard.json"),
        description: "Sentiment training corpus (text + sentiment label)",
        upstream_of: [],
        generated_by: nil
      },

      # ── NER ────────────────────────────────────────────────────────────
      %{
        id: :ner_gold,
        label: "NER Gold Standard",
        category: "NER",
        tag: :authoring,
        record_kind: :intent_example,
        path: brain_priv("evaluation/ner/gold_standard.json"),
        description: "NER evaluation corpus (text + expected entities)",
        upstream_of: [],
        generated_by: nil
      },

      # ── Hand-authored text micro-classifiers ───────────────────────────
      %{
        id: :mc_personal_question,
        label: "personal_question",
        category: "Micro-classifiers (text)",
        tag: :authoring,
        record_kind: :text_classifier_row,
        path: data_path("classifiers/personal_question.json"),
        description: "Text classifier: is this a personal question?",
        upstream_of: [],
        generated_by: nil
      },
      %{
        id: :mc_clarification_response,
        label: "clarification_response",
        category: "Micro-classifiers (text)",
        tag: :authoring,
        record_kind: :text_classifier_row,
        path: data_path("classifiers/clarification_response.json"),
        description: "Text classifier: is this a clarification response?",
        upstream_of: [],
        generated_by: nil
      },
      %{
        id: :mc_modal_directive,
        label: "modal_directive",
        category: "Micro-classifiers (text)",
        tag: :authoring,
        record_kind: :text_classifier_row,
        path: data_path("classifiers/modal_directive.json"),
        description: "Text classifier: is this a modal directive?",
        upstream_of: [],
        generated_by: nil
      },
      %{
        id: :mc_fallback_response,
        label: "fallback_response",
        category: "Micro-classifiers (text)",
        tag: :authoring,
        record_kind: :text_classifier_row,
        path: data_path("classifiers/fallback_response.json"),
        description: "Text classifier: is this a fallback response?",
        upstream_of: [],
        generated_by: nil
      },
      %{
        id: :mc_goal_type,
        label: "goal_type",
        category: "Micro-classifiers (text)",
        tag: :authoring,
        record_kind: :text_classifier_row,
        path: data_path("classifiers/goal_type.json"),
        description: "Text classifier: what type of goal is this?",
        upstream_of: [],
        generated_by: nil
      },
      %{
        id: :mc_entity_type,
        label: "entity_type",
        category: "Micro-classifiers (text)",
        tag: :authoring,
        record_kind: :text_classifier_row,
        path: data_path("classifiers/entity_type.json"),
        description: "Text classifier: what entity type is this?",
        upstream_of: [],
        generated_by: nil
      },
      %{
        id: :mc_user_fact_type,
        label: "user_fact_type",
        category: "Micro-classifiers (text)",
        tag: :authoring,
        record_kind: :text_classifier_row,
        path: data_path("classifiers/user_fact_type.json"),
        description: "Text classifier: what user fact type is this?",
        upstream_of: [],
        generated_by: nil
      },
      %{
        id: :mc_directed_at_bot,
        label: "directed_at_bot",
        category: "Micro-classifiers (text)",
        tag: :authoring,
        record_kind: :text_classifier_row,
        path: data_path("classifiers/directed_at_bot.json"),
        description: "Text classifier: is this directed at the bot?",
        upstream_of: [],
        generated_by: nil
      },
      %{
        id: :mc_event_argument_role,
        label: "event_argument_role",
        category: "Micro-classifiers (text)",
        tag: :authoring,
        record_kind: :text_classifier_row,
        path: data_path("classifiers/event_argument_role.json"),
        description: "Text classifier: what event argument role?",
        upstream_of: [],
        generated_by: nil
      },

      # ── Feature-vector micro build artifacts ───────────────────────────
      %{
        id: :tense_class_fv,
        label: "tense_class (build artifact)",
        category: "Micro-classifiers (FV)",
        tag: :build_artifact,
        record_kind: :fv_classifier_row,
        path: data_path("classifiers/tense_class.json"),
        description: "Feature-vector rows for :tense_class micro",
        upstream_of: [],
        generated_by: "mix gen_micro_data"
      },
      %{
        id: :aspect_class_fv,
        label: "aspect_class (build artifact)",
        category: "Micro-classifiers (FV)",
        tag: :build_artifact,
        record_kind: :fv_classifier_row,
        path: data_path("classifiers/aspect_class.json"),
        description: "Feature-vector rows for :aspect_class micro",
        upstream_of: [],
        generated_by: "mix gen_micro_data"
      },
      %{
        id: :urgency_fv,
        label: "urgency (build artifact)",
        category: "Micro-classifiers (FV)",
        tag: :build_artifact,
        record_kind: :fv_classifier_row,
        path: data_path("classifiers/urgency.json"),
        description: "Feature-vector rows for :urgency micro",
        upstream_of: [],
        generated_by: "mix gen_micro_data"
      },
      %{
        id: :certainty_level_fv,
        label: "certainty_level (build artifact)",
        category: "Micro-classifiers (FV)",
        tag: :build_artifact,
        record_kind: :fv_classifier_row,
        path: data_path("classifiers/certainty_level.json"),
        description: "Feature-vector rows for :certainty_level micro",
        upstream_of: [],
        generated_by: "mix gen_micro_data"
      },
      %{
        id: :coarse_semantic_class,
        label: "coarse_semantic_class (build artifact)",
        category: "Micro-classifiers (FV)",
        tag: :build_artifact,
        record_kind: :text_classifier_row,
        path: data_path("classifiers/coarse_semantic_class.json"),
        description: "Text rows for :coarse_semantic_class micro",
        upstream_of: [],
        generated_by: "mix gen_micro_data"
      },
      %{
        id: :framing_class_fv,
        label: "framing_class (build artifact)",
        category: "Micro-classifiers (FV)",
        tag: :build_artifact,
        record_kind: :fv_classifier_row,
        path: data_path("classifiers/framing_class.json"),
        description: "Feature-vector rows for :framing_class micro",
        upstream_of: [],
        generated_by: "mix gen_framing_data"
      },

      # ── Analysis registries ────────────────────────────────────────────
      %{
        id: :slot_schemas,
        label: "Slot Schemas",
        category: "Analysis Registries",
        tag: :registry,
        record_kind: :slot_schema_entry,
        path: brain_priv("analysis/slot_schemas.json"),
        description: "Per-intent required/optional slots and clarification templates",
        upstream_of: [],
        generated_by: nil
      },
      %{
        id: :entity_types,
        label: "Entity Type Hierarchy",
        category: "Analysis Registries",
        tag: :registry,
        record_kind: :entity_type_entry,
        path: brain_priv("analysis/entity_types.json"),
        description: "Entity type hierarchy and configuration",
        upstream_of: [],
        generated_by: nil
      },

      # ── Knowledge ──────────────────────────────────────────────────────
      %{
        id: :kg_hard_negatives,
        label: "KG Hard Negatives",
        category: "Knowledge",
        tag: :authoring,
        record_kind: :kg_negative,
        path: data_path("kg/hard_negatives.json"),
        description: "Curated hard negative triples for KG-LSTM training",
        upstream_of: [],
        generated_by: nil
      },

      # ── External corpora (read-only) ───────────────────────────────────
      %{
        id: :world_cities,
        label: "World Cities CSV",
        category: "External Corpora",
        tag: :external_corpus,
        record_kind: :csv_row,
        path: data_path("world-cities.csv"),
        description: "World cities reference data for gazetteer",
        upstream_of: [],
        generated_by: nil
      },
      %{
        id: :us_cities,
        label: "US Cities CSV",
        category: "External Corpora",
        tag: :external_corpus,
        record_kind: :csv_row,
        path: data_path("us_cities.csv"),
        description: "US cities reference data for gazetteer",
        upstream_of: [],
        generated_by: nil
      },
      %{
        id: :music_artists,
        label: "Music Artists CSV",
        category: "External Corpora",
        tag: :external_corpus,
        record_kind: :csv_row,
        path: data_path("Global Music Artists.csv"),
        description: "Global music artists for gazetteer",
        upstream_of: [],
        generated_by: nil
      }
    ] ++ gazetteer_descriptors()
  end

  @spec get(atom()) :: t() | nil
  def get(id) do
    Enum.find(all(), &(&1.id == id))
  end

  @spec editable?(t()) :: boolean()
  def editable?(%{tag: tag}), do: tag in [:authoring, :registry, :gazetteer]

  @spec categories() :: [String.t()]
  def categories do
    all()
    |> Enum.map(& &1.category)
    |> Enum.uniq()
  end

  @spec by_category() :: [{String.t(), [t()]}]
  def by_category do
    all()
    |> Enum.group_by(& &1.category)
    |> Enum.sort_by(fn {cat, _} -> category_sort_key(cat) end)
  end

  defp category_sort_key("Intent Classification"), do: 0
  defp category_sort_key("Speech Act"), do: 1
  defp category_sort_key("Sentiment"), do: 2
  defp category_sort_key("NER"), do: 3
  defp category_sort_key("Micro-classifiers (text)"), do: 4
  defp category_sort_key("Micro-classifiers (FV)"), do: 5
  defp category_sort_key("Analysis Registries"), do: 6
  defp category_sort_key("Gazetteers"), do: 7
  defp category_sort_key("Knowledge"), do: 8
  defp category_sort_key("External Corpora"), do: 9
  defp category_sort_key(_), do: 10

  defp gazetteer_descriptors do
    entities_path = data_path("entities")

    case File.ls(entities_path) do
      {:ok, files} ->
        files
        |> Enum.filter(&gazetteer_entry_file?/1)
        |> Enum.sort()
        |> Enum.map(fn file ->
          base = Path.rootname(file)

          %{
            id: String.to_atom("gaz_#{base}"),
            label: base,
            category: "Gazetteers",
            tag: :gazetteer,
            record_kind: :gazetteer_entry,
            path: Path.join(entities_path, file),
            description: "Entity gazetteer: #{base}",
            upstream_of: [],
            generated_by: nil
          }
        end)

      _ ->
        []
    end
  end

  defp gazetteer_entry_file?(filename) do
    String.ends_with?(filename, "_entries_en.json") or
      filename in ["anaphora.json", "slot_mappings.json"]
  end
end
