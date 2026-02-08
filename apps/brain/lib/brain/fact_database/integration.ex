defmodule Brain.FactDatabase.Integration do
  @moduledoc "Integration layer between FactDatabase and the epistemic system.\n\nThis module provides:\n1. Dynamic fact addition (learned facts can be added to the database)\n2. Integration with epistemic user model (facts as beliefs)\n3. Truth maintenance verification (facts registered with JTMS)\n"

  alias Brain.FactDatabase
  alias Brain.FactDatabase.Fact
  alias Brain.Epistemic.{BeliefStore, JTMS}
  alias Brain.Epistemic.Types.Belief
  require Logger

  @doc "Adds a new fact to the database dynamically.\n\nThis allows the system to learn and grow its fact database.\nThe fact is also registered with the epistemic system for verification.\n\nOptions:\n- :category - Category for the fact (default: \"learned\")\n- :verification_source - Source of verification\n- :confidence - Confidence level (0.0-1.0, default: 0.8)\n- :register_with_jtms - Whether to register with JTMS (default: true)\n- :create_belief - Whether to create a belief (default: true)\n"
  def add_fact(entity, fact_text, opts \\ []) do
    category = Keyword.get(opts, :category, "learned")
    entity_type = Keyword.get(opts, :entity_type)
    verification_source = Keyword.get(opts, :verification_source, "learned_from_conversation")
    confidence = Keyword.get(opts, :confidence, 0.8)
    register_with_jtms? = Keyword.get(opts, :register_with_jtms, true)
    create_belief? = Keyword.get(opts, :create_belief, true)

    new_fact =
      Fact.new(
        id: "learned_#{generate_id()}",
        entity: entity,
        entity_type: entity_type,
        fact: fact_text,
        category: category,
        verification_source: verification_source,
        confidence: confidence,
        learned_at: System.system_time(:second)
      )

    store_learned_fact(new_fact)

    node_id =
      if register_with_jtms? do
        register_fact_with_jtms(new_fact.id, entity, fact_text, confidence)
      else
        nil
      end

    if create_belief? do
      create_fact_belief(entity, fact_text, confidence, verification_source, node_id)
    end

    Logger.info("Added learned fact", %{
      fact_id: new_fact.id,
      entity: entity,
      entity_type: new_fact.entity_type,
      category: category,
      confidence: confidence
    })

    {:ok, new_fact.id, new_fact}
  end

  @doc "Verifies a fact against existing beliefs and the truth maintenance system.\n\nReturns:\n- {:verified, confidence} - Fact is consistent with existing beliefs\n- {:contradicted, conflicting_beliefs} - Fact contradicts existing beliefs\n- {:uncertain, reason} - Cannot verify (low confidence, no data, etc.)\n"
  def verify_fact(entity, fact_text) do
    case BeliefStore.query_beliefs(subject: :world, predicate: normalize_entity(entity)) do
      {:ok, [_ | _] = beliefs} ->
        contradictions = find_contradictions(fact_text, beliefs)

        if contradictions != [] do
          {:contradicted, contradictions}
        else
          max_confidence = Enum.max_by(beliefs, & &1.confidence, fn -> nil end)

          if max_confidence && max_confidence.confidence >= 0.7 do
            {:verified, max_confidence.confidence}
          else
            {:uncertain, :low_confidence}
          end
        end

      _ ->
        {:uncertain, :no_existing_beliefs}
    end
  end

  @doc "Syncs facts from the FactDatabase to the epistemic system as beliefs.\n\nThis creates beliefs for all facts in the database, allowing them to be\nverified and tracked by the truth maintenance system.\n"
  def sync_facts_to_beliefs(opts \\ []) do
    category = Keyword.get(opts, :category)
    min_confidence = Keyword.get(opts, :min_confidence, 0.7)
    facts = FactDatabase.query(category: category, limit: 1000)

    verified_facts =
      Enum.filter(facts, fn fact ->
        fact.confidence >= min_confidence
      end)

    created =
      Enum.map(verified_facts, fn fact ->
        verification_source = fact.verification_source || "fact_database"

        belief =
          Belief.new(:world, normalize_entity(fact.entity), fact.fact,
            source: :learned,
            confidence: fact.confidence,
            provenance: ["fact_database", verification_source],
            metadata: %{
              fact_id: fact.id,
              category: fact.category
            }
          )

        case BeliefStore.add_belief(belief) do
          {:ok, belief_id} ->
            if fact.confidence >= 0.9 do
              case JTMS.create_premise("fact:#{belief_id}", metadata: %{fact_id: fact.id}) do
                {:ok, node_id} ->
                  BeliefStore.link_to_node(belief_id, node_id)
                  1

                _ ->
                  1
              end
            else
              1
            end

          _ ->
            0
        end
      end)
      |> Enum.sum()

    Logger.info("Synced facts to beliefs", %{facts_synced: created})
    {:ok, created}
  end

  @doc "Checks if a fact contradicts any existing beliefs or facts.\n\nUses the JTMS to check for contradictions.\n"
  def check_contradiction(entity, fact_text) do
    case BeliefStore.query_beliefs(subject: :world, predicate: normalize_entity(entity)) do
      {:ok, beliefs} ->
        contradictions =
          Enum.filter(beliefs, fn belief ->
            contradicts?(fact_text, belief.object)
          end)

        if contradictions != [] do
          {:contradiction, contradictions}
        else
          :consistent
        end

      _ ->
        :no_data
    end
  end

  defp store_learned_fact(%Fact{} = fact) do
    default_path = Path.join([File.cwd!(), "data/facts/learned.json"])
    learned_file = Application.get_env(:brain, :learned_facts_path, default_path)

    existing_facts =
      if File.exists?(learned_file) do
        case File.read(learned_file) do
          {:ok, content} ->
            case Jason.decode(content) do
              {:ok, data} -> Map.get(data, "facts", [])
              _ -> []
            end

          _ ->
            []
        end
      else
        []
      end

    fact_map = Fact.to_map(fact)
    updated_facts = [fact_map | existing_facts]

    data = %{
      "category" => "learned",
      "description" => "Facts learned dynamically from conversations",
      "facts" => updated_facts
    }

    File.mkdir_p!(Path.dirname(learned_file))
    File.write!(learned_file, Jason.encode!(data, pretty: true))
    FactDatabase.reload()
  end

  defp register_fact_with_jtms(fact_id, entity, fact_text, confidence) do
    if Process.whereis(JTMS) do
      node_type =
        if confidence >= 0.9 do
          :premise
        else
          :assumption
        end

      case JTMS.create_node("fact:#{fact_id}",
             node_type: node_type,
             metadata: %{entity: entity, fact: fact_text, fact_id: fact_id}
           ) do
        {:ok, node_id} ->
          if node_type == :assumption do
            JTMS.enable_assumption(node_id)
          end

          node_id

        _ ->
          nil
      end
    else
      nil
    end
  end

  defp create_fact_belief(entity, fact_text, confidence, verification_source, node_id) do
    belief =
      Belief.new(:world, normalize_entity(entity), fact_text,
        source: :learned,
        confidence: confidence,
        provenance: ["fact_database", verification_source],
        node_id: node_id,
        metadata: %{category: "learned"}
      )

    case BeliefStore.add_belief(belief) do
      {:ok, _belief_id} ->
        :ok

      error ->
        Logger.warning("Failed to create belief for fact", %{error: error})
        :error
    end
  end

  defp find_contradictions(fact_text, beliefs) do
    Enum.filter(beliefs, fn belief ->
      contradicts?(fact_text, belief.object)
    end)
  end

  defp contradicts?(text1, text2) when is_binary(text1) and is_binary(text2) do
    normalized1 = String.downcase(text1)
    normalized2 = String.downcase(text2)
    has_negation = String.contains?(normalized1, "not ") or String.contains?(normalized2, "not ")
    is_opposite = check_opposite_meaning(normalized1, normalized2)

    has_negation or is_opposite
  end

  defp contradicts?(_text1, _text2) do
    false
  end

  defp check_opposite_meaning(text1, text2) do
    cond do
      String.contains?(text1, " is ") and String.contains?(text2, " is not ") -> true
      String.contains?(text1, " is not ") and String.contains?(text2, " is ") -> true
      true -> false
    end
  end

  defp normalize_entity(entity) when is_binary(entity) do
    entity
    |> String.downcase()
    |> String.replace(" ", "_")
    |> String.to_atom()
  end

  defp normalize_entity(entity) when is_atom(entity) do
    entity
  end

  defp normalize_entity(_) do
    :unknown
  end

  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end
end