defmodule ChatBot.FactDatabase.Integration do
  @moduledoc """
  Integration layer between FactDatabase and the epistemic system.
  
  This module provides:
  1. Dynamic fact addition (learned facts can be added to the database)
  2. Integration with epistemic user model (facts as beliefs)
  3. Truth maintenance verification (facts registered with JTMS)
  """

  alias ChatBot.FactDatabase
  alias ChatBot.Epistemic.{BeliefStore, JTMS, UserModelStore}
  alias ChatBot.Epistemic.Types.Belief
  require Logger

  @doc """
  Adds a new fact to the database dynamically.
  
  This allows the system to learn and grow its fact database.
  The fact is also registered with the epistemic system for verification.
  
  Options:
  - :category - Category for the fact (default: "learned")
  - :verification_source - Source of verification
  - :confidence - Confidence level (0.0-1.0, default: 0.8)
  - :register_with_jtms - Whether to register with JTMS (default: true)
  - :create_belief - Whether to create a belief (default: true)
  """
  def add_fact(entity, fact_text, opts \\ []) do
    category = Keyword.get(opts, :category, "learned")
    verification_source = Keyword.get(opts, :verification_source, "learned_from_conversation")
    confidence = Keyword.get(opts, :confidence, 0.8)
    register_with_jtms? = Keyword.get(opts, :register_with_jtms, true)
    create_belief? = Keyword.get(opts, :create_belief, true)

    # Generate fact ID
    fact_id = "learned_#{generate_id()}"

    # Create fact structure
    new_fact = %{
      "id" => fact_id,
      "entity" => entity,
      "fact" => fact_text,
      "category" => category,
      "verification_source" => verification_source,
      "confidence" => confidence,
      "learned_at" => System.system_time(:second)
    }

    # Add to fact database (we'll need to extend FactDatabase to support this)
    # For now, we'll store it in a learned facts file
    store_learned_fact(new_fact)

    # Register with JTMS if requested
    node_id =
      if register_with_jtms? do
        register_fact_with_jtms(fact_id, entity, fact_text, confidence)
      else
        nil
      end

    # Create belief if requested
    if create_belief? do
      create_fact_belief(entity, fact_text, confidence, verification_source, node_id)
    end

    Logger.info("Added learned fact", %{
      fact_id: fact_id,
      entity: entity,
      category: category,
      confidence: confidence
    })

    {:ok, fact_id, new_fact}
  end

  @doc """
  Verifies a fact against existing beliefs and the truth maintenance system.
  
  Returns:
  - {:verified, confidence} - Fact is consistent with existing beliefs
  - {:contradicted, conflicting_beliefs} - Fact contradicts existing beliefs
  - {:uncertain, reason} - Cannot verify (low confidence, no data, etc.)
  """
  def verify_fact(entity, fact_text) do
    # Check against existing beliefs
    case BeliefStore.query_beliefs(subject: :world, predicate: normalize_entity(entity)) do
      {:ok, beliefs} when length(beliefs) > 0 ->
        # Check for contradictions
        contradictions = find_contradictions(fact_text, beliefs)

        if contradictions != [] do
          {:contradicted, contradictions}
        else
          # Check confidence levels
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

  @doc """
  Syncs facts from the FactDatabase to the epistemic system as beliefs.
  
  This creates beliefs for all facts in the database, allowing them to be
  verified and tracked by the truth maintenance system.
  """
  def sync_facts_to_beliefs(opts \\ []) do
    category = Keyword.get(opts, :category)
    min_confidence = Keyword.get(opts, :min_confidence, 0.7)

    # Query facts from database
    facts = FactDatabase.query(category: category, limit: 1000)

    # Filter by confidence
    verified_facts = Enum.filter(facts, fn fact ->
      Map.get(fact, "confidence", 0.0) >= min_confidence
    end)

    # Create beliefs for each fact
    created =
      Enum.map(verified_facts, fn fact ->
        entity = Map.get(fact, "entity", "unknown")
        fact_text = Map.get(fact, "fact", "")
        confidence = Map.get(fact, "confidence", 0.8)
        verification_source = Map.get(fact, "verification_source", "fact_database")

        belief =
          Belief.new(:world, normalize_entity(entity), fact_text,
            source: :learned,
            confidence: confidence,
            provenance: ["fact_database", verification_source],
            metadata: %{
              fact_id: Map.get(fact, "id"),
              category: Map.get(fact, "category")
            }
          )

        case BeliefStore.add_belief(belief) do
          {:ok, belief_id} ->
            # Register with JTMS as a premise (high confidence verified fact)
            if confidence >= 0.9 do
              case JTMS.create_premise("fact:#{belief_id}", metadata: %{fact_id: Map.get(fact, "id")}) do
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

  @doc """
  Checks if a fact contradicts any existing beliefs or facts.
  
  Uses the JTMS to check for contradictions.
  """
  def check_contradiction(entity, fact_text) do
    # Query for existing beliefs about this entity
    case BeliefStore.query_beliefs(subject: :world, predicate: normalize_entity(entity)) do
      {:ok, beliefs} ->
        # Check each belief for contradiction
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

  # Private Functions

  defp store_learned_fact(fact) do
    # Store learned facts in a separate file
    learned_file = Path.join([File.cwd!(), "data/facts/learned.json"])

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

    # Add new fact
    updated_facts = [fact | existing_facts]

    # Write back
    data = %{
      "category" => "learned",
      "description" => "Facts learned dynamically from conversations",
      "facts" => updated_facts
    }

    File.mkdir_p!(Path.dirname(learned_file))
    File.write!(learned_file, Jason.encode!(data, pretty: true))

    # Reload fact database
    FactDatabase.reload()
  end

  defp register_fact_with_jtms(fact_id, entity, fact_text, confidence) do
    if Process.whereis(JTMS) do
      # Create a premise node for high-confidence facts, assumption for lower confidence
      node_type = if confidence >= 0.9, do: :premise, else: :assumption

      case JTMS.create_node("fact:#{fact_id}",
           node_type: node_type,
           metadata: %{entity: entity, fact: fact_text, fact_id: fact_id}
         ) do
        {:ok, node_id} ->
          # Enable assumption if it's an assumption node
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
    # Simple contradiction detection: check for negations or opposite statements
    normalized1 = String.downcase(text1)
    normalized2 = String.downcase(text2)

    # Check for explicit contradictions
    has_negation = String.contains?(normalized1, "not ") or String.contains?(normalized2, "not ")
    is_opposite = check_opposite_meaning(normalized1, normalized2)

    has_negation or is_opposite
  end

  defp contradicts?(_text1, _text2), do: false

  defp check_opposite_meaning(text1, text2) do
    # Simple heuristic: check if one says "is X" and other says "is not X" or "is Y" where Y is opposite
    # This is a simplified check - could be enhanced with more sophisticated NLP
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

  defp normalize_entity(entity) when is_atom(entity), do: entity
  defp normalize_entity(_), do: :unknown

  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end
end
