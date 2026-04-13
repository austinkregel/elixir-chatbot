defmodule Brain.Epistemic.Types do
  @moduledoc "Core type definitions for the Epistemic Truth Maintenance System.\n\nThis module defines the fundamental structures used throughout the\nepistemic layer:\n\n- Belief: A piece of knowledge with confidence, source, and provenance\n- Justification: Links beliefs in a dependency network (JTMS)\n- Node: A node in the JTMS dependency network\n- UserModel: Aggregated knowledge about a specific user\n- SelfKnowledgeAssessment: Result of analyzing what the system knows\n- DisclosureDecision: Whether and how to share a piece of knowledge\n"

  defmodule Belief do
    @moduledoc "A belief represents a piece of knowledge the system holds about\na subject (user, world, or self).\n\nEach belief tracks:\n- What it's about (subject/predicate/object triple)\n- How confident we are (0.0 - 1.0)\n- Where it came from (source and provenance)\n- How stable it is (volatility)\n"

    @type source :: :explicit | :inferred | :assumed | :default | :learned | :consolidated | :curated_fact
    @type subject :: :user | :world | :self | String.t()

    @type t :: %__MODULE__{
            id: String.t(),
            subject: subject(),
            predicate: atom() | String.t(),
            object: any(),
            confidence: float(),
            source: source(),
            source_authority: atom() | nil,
            provenance: list(String.t()),
            volatility: float(),
            last_confirmed: DateTime.t() | nil,
            created_at: DateTime.t(),
            user_id: String.t() | nil,
            node_id: String.t() | nil,
            metadata: map()
          }

    defstruct [
      :id,
      :subject,
      :predicate,
      :object,
      :confidence,
      :source,
      :source_authority,
      :last_confirmed,
      :created_at,
      :user_id,
      :node_id,
      provenance: [],
      volatility: 0.5,
      metadata: %{}
    ]

    @doc "Creates a new belief with auto-generated ID and timestamp.\n"
    def new(subject, predicate, object, opts \\ []) do
      %__MODULE__{
        id: generate_id(),
        subject: subject,
        predicate: predicate,
        object: object,
        confidence: Keyword.get(opts, :confidence, 0.5),
        source: Keyword.get(opts, :source, :inferred),
        source_authority: Keyword.get(opts, :source_authority),
        provenance: Keyword.get(opts, :provenance, []),
        volatility: Keyword.get(opts, :volatility, 0.5),
        user_id: Keyword.get(opts, :user_id),
        node_id: Keyword.get(opts, :node_id),
        last_confirmed: Keyword.get(opts, :last_confirmed),
        created_at: DateTime.utc_now(),
        metadata: Keyword.get(opts, :metadata, %{})
      }
    end

    @doc "Updates the confidence of a belief, optionally confirming it.\n"
    def update_confidence(%__MODULE__{} = belief, new_confidence, confirm? \\ false) do
      belief
      |> Map.put(:confidence, clamp(new_confidence, 0.0, 1.0))
      |> maybe_confirm(confirm?)
    end

    @doc "Marks the belief as confirmed now.\n"
    def confirm(%__MODULE__{} = belief) do
      %{belief | last_confirmed: DateTime.utc_now()}
    end

    @doc "Checks if this belief is considered \"high confidence\" (>= threshold).\n"
    def high_confidence?(%__MODULE__{confidence: conf}, threshold \\ 0.7) do
      conf >= threshold
    end

    @doc "Checks if this belief is from an explicit user statement.\n"
    def explicit?(%__MODULE__{source: :explicit}) do
      true
    end

    def explicit?(_) do
      false
    end

    defp maybe_confirm(belief, true) do
      confirm(belief)
    end

    defp maybe_confirm(belief, false) do
      belief
    end

    defp clamp(value, min, max) do
      value |> max(min) |> min(max)
    end

    defp generate_id do
      :crypto.strong_rand_bytes(16) |> Base.encode16(case: :lower)
    end
  end

  defmodule Node do
    @moduledoc "A node in the JTMS dependency network.\n\nNodes can be:\n- Premises (unconditionally true)\n- Assumptions (can be enabled/retracted)\n- Derived (justified by other nodes)\n- Contradictions (mark inconsistent states)\n\nThe label (:in or :out) indicates the current belief status.\n"

    @type label :: :in | :out
    @type node_type :: :premise | :assumption | :derived | :contradiction

    @type t :: %__MODULE__{
            id: String.t(),
            datum: any(),
            label: label(),
            node_type: node_type(),
            justifications: list(String.t()),
            consequences: list(String.t()),
            assumption_enabled: boolean(),
            metadata: map()
          }

    defstruct [
      :id,
      :datum,
      label: :out,
      node_type: :derived,
      justifications: [],
      consequences: [],
      assumption_enabled: false,
      metadata: %{}
    ]

    @doc "Creates a new node.\n"
    def new(datum, opts \\ []) do
      node_type = Keyword.get(opts, :node_type, :derived)

      %__MODULE__{
        id: generate_id(),
        datum: datum,
        label: initial_label(node_type, opts),
        node_type: node_type,
        assumption_enabled: Keyword.get(opts, :assumption_enabled, false),
        metadata: Keyword.get(opts, :metadata, %{})
      }
    end

    @doc "Creates a premise node (always IN).\n"
    def premise(datum, opts \\ []) do
      new(datum, Keyword.merge(opts, node_type: :premise))
    end

    @doc "Creates an assumption node.\n"
    def assumption(datum, enabled? \\ false, opts \\ []) do
      new(datum, Keyword.merge(opts, node_type: :assumption, assumption_enabled: enabled?))
    end

    @doc "Creates a contradiction marker node.\n"
    def contradiction(datum, opts \\ []) do
      new(datum, Keyword.merge(opts, node_type: :contradiction))
    end

    @doc "Checks if the node is currently IN.\n"
    def in?(%__MODULE__{label: :in}) do
      true
    end

    def in?(_) do
      false
    end

    @doc "Checks if the node is currently OUT.\n"
    def out?(%__MODULE__{label: :out}) do
      true
    end

    def out?(_) do
      false
    end

    defp initial_label(:premise, _opts) do
      :in
    end

    defp initial_label(:assumption, opts) do
      if(Keyword.get(opts, :assumption_enabled)) do
        :in
      else
        :out
      end
    end

    defp initial_label(_, _opts) do
      :out
    end

    defp generate_id do
      :crypto.strong_rand_bytes(16) |> Base.encode16(case: :lower)
    end
  end

  defmodule Justification do
    @moduledoc "A justification links premise nodes to a conclusion node.\n\nIn JTMS, a justification is valid (label = :in) when:\n- All nodes in the in_list are IN\n- All nodes in the out_list are OUT\n\nWhen valid, the justification supports its conclusion being IN.\n"

    @type label :: :in | :out

    @type t :: %__MODULE__{
            id: String.t(),
            in_list: list(String.t()),
            out_list: list(String.t()),
            conclusion_id: String.t(),
            informant: String.t(),
            label: label()
          }

    defstruct [:id, :conclusion_id, :informant, in_list: [], out_list: [], label: :out]

    @doc "Creates a new justification.\n\n- in_list: Node IDs that must be IN for this justification to be valid\n- out_list: Node IDs that must be OUT for this justification to be valid\n- conclusion_id: The node this justification supports\n- informant: What/who created this justification\n"
    def new(in_list, out_list, conclusion_id, informant) do
      %__MODULE__{
        id: generate_id(),
        in_list: in_list,
        out_list: out_list,
        conclusion_id: conclusion_id,
        informant: informant,
        label: :out
      }
    end

    @doc "Creates a simple justification (no out_list).\n"
    def simple(premise_ids, conclusion_id, informant) when is_list(premise_ids) do
      new(premise_ids, [], conclusion_id, informant)
    end

    @doc "Checks if this justification would be valid given node labels.\n\nReturns true if all in_list nodes are :in and all out_list nodes are :out.\n"
    def valid?(justification, node_labels) when is_map(node_labels) do
      all_in_are_in =
        Enum.all?(justification.in_list, fn id ->
          Map.get(node_labels, id, :out) == :in
        end)

      all_out_are_out =
        Enum.all?(justification.out_list, fn id ->
          Map.get(node_labels, id, :out) == :out
        end)

      all_in_are_in and all_out_are_out
    end

    defp generate_id do
      :crypto.strong_rand_bytes(16) |> Base.encode16(case: :lower)
    end
  end

  defmodule UserModel do
    @moduledoc "An explicit, inspectable model of what the system knows about a user.\n\nThe UserModel aggregates beliefs about a specific user and tracks:\n- Facts with their values\n- Interaction patterns\n- Epistemic bounds (confidence per fact)\n- Provenance (how each fact was learned)\n- Disclosure history (what has been shared)\n"

    @type provenance :: :explicit | :inferred | :assumed | :learned

    @type t :: %__MODULE__{
            user_id: String.t(),
            facts: map(),
            interaction_patterns: map(),
            epistemic_bounds: map(),
            provenance_map: map(),
            disclosure_history: list(map()),
            created_at: DateTime.t(),
            updated_at: DateTime.t()
          }

    defstruct [
      :user_id,
      :created_at,
      :updated_at,
      facts: %{},
      interaction_patterns: %{},
      epistemic_bounds: %{},
      provenance_map: %{},
      disclosure_history: []
    ]

    @doc "Creates a new UserModel for the given user ID.\n"
    def new(user_id) do
      now = DateTime.utc_now()

      %__MODULE__{
        user_id: user_id,
        created_at: now,
        updated_at: now
      }
    end

    @doc "Updates a fact in the user model.\n"
    def update_fact(%__MODULE__{} = model, key, value, source, confidence) do
      %{
        model
        | facts: Map.put(model.facts, key, value),
          epistemic_bounds: Map.put(model.epistemic_bounds, key, confidence),
          provenance_map: Map.put(model.provenance_map, key, source),
          updated_at: DateTime.utc_now()
      }
    end

    @doc "Gets a fact with its confidence and provenance.\n"
    def get_fact(%__MODULE__{} = model, key) do
      case Map.get(model.facts, key) do
        nil ->
          nil

        value ->
          %{
            value: value,
            confidence: Map.get(model.epistemic_bounds, key, 0.5),
            provenance: Map.get(model.provenance_map, key, :unknown)
          }
      end
    end

    @doc "Gets all facts above a confidence threshold.\n"
    def get_facts_above_confidence(%__MODULE__{} = model, min_confidence) do
      model.facts
      |> Enum.filter(fn {key, _value} ->
        Map.get(model.epistemic_bounds, key, 0.0) >= min_confidence
      end)
      |> Enum.map(fn {key, value} ->
        %{
          key: key,
          value: value,
          confidence: Map.get(model.epistemic_bounds, key),
          provenance: Map.get(model.provenance_map, key)
        }
      end)
    end

    @doc "Records an interaction pattern.\n"
    def record_pattern(%__MODULE__{} = model, pattern_type, data) do
      current = Map.get(model.interaction_patterns, pattern_type, [])
      updated = [data | current] |> Enum.take(100)

      %{
        model
        | interaction_patterns: Map.put(model.interaction_patterns, pattern_type, updated),
          updated_at: DateTime.utc_now()
      }
    end

    @doc "Records that something was disclosed to the user.\n"
    def record_disclosure(%__MODULE__{} = model, disclosed_keys, context) do
      entry = %{
        keys: disclosed_keys,
        context: context,
        timestamp: DateTime.utc_now()
      }

      %{
        model
        | disclosure_history: [entry | model.disclosure_history] |> Enum.take(50),
          updated_at: DateTime.utc_now()
      }
    end
  end

  defmodule SelfKnowledgeAssessment do
    @moduledoc "The result of the system analyzing what it knows about a user.\n\nUsed to construct self-referential responses like:\n\"From what I remember, you mentioned X...\"\n\nCategorizes knowledge into:\n- discloseable: Safe to share with confidence\n- inferred_uncertain: Share with hedging\n- should_avoid: Don't disclose (too personal, too uncertain, etc.)\n"

    @type knowledge_item :: %{
            key: atom() | String.t(),
            value: any(),
            confidence: float(),
            provenance: atom()
          }

    @type t :: %__MODULE__{
            user_id: String.t(),
            discloseable: list(knowledge_item()),
            inferred_uncertain: list(knowledge_item()),
            should_avoid: list(knowledge_item()),
            total_facts: non_neg_integer(),
            assessment_timestamp: DateTime.t()
          }

    defstruct [
      :user_id,
      :assessment_timestamp,
      discloseable: [],
      inferred_uncertain: [],
      should_avoid: [],
      total_facts: 0
    ]

    @doc "Creates a new assessment for the given user.\n"
    def new(user_id) do
      %__MODULE__{
        user_id: user_id,
        assessment_timestamp: DateTime.utc_now()
      }
    end

    @doc "Builds an assessment from a UserModel.\n"
    def from_user_model(%UserModel{} = model, opts \\ []) do
      high_conf_threshold = Keyword.get(opts, :high_confidence, 0.7)
      low_conf_threshold = Keyword.get(opts, :low_confidence, 0.4)
      sensitive_keys = Keyword.get(opts, :sensitive_keys, [:password, :ssn, :credit_card])

      all_facts = UserModel.get_facts_above_confidence(model, 0.0)

      {discloseable, rest} =
        Enum.split_with(all_facts, fn fact ->
          fact.confidence >= high_conf_threshold and
            fact.key not in sensitive_keys
        end)

      {inferred_uncertain, should_avoid} =
        Enum.split_with(rest, fn fact ->
          fact.confidence >= low_conf_threshold and
            fact.key not in sensitive_keys
        end)

      %__MODULE__{
        user_id: model.user_id,
        discloseable: discloseable,
        inferred_uncertain: inferred_uncertain,
        should_avoid: should_avoid,
        total_facts: length(all_facts),
        assessment_timestamp: DateTime.utc_now()
      }
    end

    @doc "Checks if the assessment has any knowledge to share.\n"
    def has_knowledge?(%__MODULE__{discloseable: d, inferred_uncertain: i}) do
      d != [] or i != []
    end
  end

  defmodule DisclosureDecision do
    @moduledoc "The result of evaluating whether a piece of knowledge should be disclosed.\n\nAnswers the validation questions:\n- V6: Is this socially appropriate to disclose?\n- V7: Would this sound creepy if said confidently?\n- V8: Should I hedge, ask permission, or generalize?\n"

    @type hedging_level :: :none | :light | :strong

    @type t :: %__MODULE__{
            should_disclose: boolean(),
            hedging_required: hedging_level(),
            generalize: boolean(),
            ask_permission: boolean(),
            reason: String.t(),
            belief_id: String.t() | nil
          }

    defstruct [
      :reason,
      :belief_id,
      should_disclose: true,
      hedging_required: :none,
      generalize: false,
      ask_permission: false
    ]

    @doc "Creates a decision to disclose with no hedging.\n"
    def disclose(reason \\ "High confidence explicit fact") do
      %__MODULE__{
        should_disclose: true,
        hedging_required: :none,
        reason: reason
      }
    end

    @doc "Creates a decision to disclose with hedging.\n"
    def disclose_with_hedging(level, reason) do
      %__MODULE__{
        should_disclose: true,
        hedging_required: level,
        reason: reason
      }
    end

    @doc "Creates a decision to not disclose.\n"
    def do_not_disclose(reason) do
      %__MODULE__{
        should_disclose: false,
        reason: reason
      }
    end

    @doc "Creates a decision to ask permission first.\n"
    def ask_first(reason) do
      %__MODULE__{
        should_disclose: false,
        ask_permission: true,
        reason: reason
      }
    end
  end

  defmodule Config do
    @moduledoc "Configuration options for the epistemic system.\n\nAllows enabling/disabling features for testing reproducibility.\n"

    @type t :: %__MODULE__{
            enabled: boolean(),
            auto_belief_extraction: boolean(),
            reflection_mode: :async | :sync | :disabled,
            consolidation_interval_ms: non_neg_integer(),
            high_confidence_threshold: float(),
            low_confidence_threshold: float(),
            decay_rate: float(),
            decay_exempt_sources: [atom()],
            decay_interval_ms: non_neg_integer(),
            decay_min_age_ms: non_neg_integer()
          }

    defstruct enabled: true,
              auto_belief_extraction: true,
              reflection_mode: :async,
              consolidation_interval_ms: 3_600_000,
              high_confidence_threshold: 0.7,
              low_confidence_threshold: 0.4,
              decay_rate: 0.05,
              decay_exempt_sources: [:explicit, :learned, :curated_fact],
              decay_interval_ms: 3_600_000,
              decay_min_age_ms: 86_400_000

    @doc "Gets the current epistemic configuration.\n"
    def get do
      config = Application.get_env(:brain, :epistemic, [])

      %__MODULE__{
        enabled: Keyword.get(config, :enabled, true),
        auto_belief_extraction: Keyword.get(config, :auto_belief_extraction, true),
        reflection_mode: Keyword.get(config, :reflection_mode, :async),
        consolidation_interval_ms: Keyword.get(config, :consolidation_interval_ms, 3_600_000),
        high_confidence_threshold: Keyword.get(config, :high_confidence_threshold, 0.7),
        low_confidence_threshold: Keyword.get(config, :low_confidence_threshold, 0.4),
        decay_rate: Keyword.get(config, :decay_rate, 0.05),
        decay_exempt_sources: Keyword.get(config, :decay_exempt_sources, [:explicit, :learned, :curated_fact]),
        decay_interval_ms: Keyword.get(config, :decay_interval_ms, 3_600_000),
        decay_min_age_ms: Keyword.get(config, :decay_min_age_ms, 86_400_000)
      }
    end

    @doc "Checks if epistemic features are enabled.\n"
    def enabled? do
      get().enabled
    end

    @doc "Checks if automatic belief extraction is enabled.\n"
    def auto_extraction_enabled? do
      config = get()
      config.enabled and config.auto_belief_extraction
    end

    @doc "Checks if reflection is synchronous.\n"
    def sync_reflection? do
      get().reflection_mode == :sync
    end
  end
end
