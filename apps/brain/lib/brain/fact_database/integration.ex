defmodule Brain.FactDatabase.Integration do
  @moduledoc "Integration layer between FactDatabase and the epistemic system.\n\nThis module provides:\n1. Dynamic fact addition (learned facts can be added to the database)\n2. Integration with epistemic user model (facts as beliefs)\n3. Truth maintenance verification (facts registered with JTMS)\n\nBeliefs and JTMS networks are held per world, so every function here takes\nthe world first: a fact's belief and JTMS node are created in that world, and\nverification reads only that world's beliefs. A missing world raises.\n"

  alias Brain.FactDatabase
  alias Brain.FactDatabase.Fact
  alias Brain.Epistemic.{BeliefStore, JTMS}
  alias Brain.Epistemic.Types.Belief
  alias Brain.Lexicon.ClosedClass
  alias Brain.LinguisticData
  alias Brain.ML.{POSTagger, Tokenizer}
  require Logger

  @doc "Adds a new fact to the database dynamically.\n\nThis allows the system to learn and grow its fact database.\nThe fact is also registered with the epistemic system for verification.\n\nOptions:\n- :category - Category for the fact (default: \"learned\")\n- :verification_source - Source of verification\n- :confidence - Confidence level (0.0-1.0, default: 0.8)\n- :register_with_jtms - Whether to register with JTMS (default: true)\n- :create_belief - Whether to create a belief (default: true)\n"
  def add_fact(world_id, entity, fact_text, opts \\ []) do
    world_id = world_id!(world_id, "add_fact")
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
        register_fact_with_jtms(world_id, new_fact.id, entity, fact_text, confidence)
      else
        nil
      end

    if create_belief? do
      create_fact_belief(world_id, entity, fact_text, confidence, verification_source, node_id)
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
  def verify_fact(world_id, entity, fact_text) do
    world_id = world_id!(world_id, "verify_fact")

    case BeliefStore.query_beliefs(world_id, subject: :world, predicate: normalize_entity(entity)) do
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

  @doc """
  Syncs facts from the FactDatabase to the world's epistemic system as beliefs.

  Curated facts are registered with `:curated_fact` source authority and as
  JTMS premises (unconditionally true, no decay). Learned facts use their
  original source and are registered as JTMS assumptions.

  Options:
  - `:category` - Filter by category
  - `:min_confidence` - Minimum confidence threshold (default: 0.7)
  - `:layer` - `:curated`, `:learned`, or `:all` (default: `:all`)
  """
  def sync_facts_to_beliefs(world_id, opts \\ []) do
    world_id = world_id!(world_id, "sync_facts_to_beliefs")
    category = Keyword.get(opts, :category)
    min_confidence = Keyword.get(opts, :min_confidence, 0.7)
    layer = Keyword.get(opts, :layer, :all)

    facts = FactDatabase.query(category: category, limit: 1000, layer: layer)

    verified_facts =
      Enum.filter(facts, fn fact ->
        fact.confidence >= min_confidence
      end)

    created =
      Enum.map(verified_facts, fn fact ->
        is_curated = FactDatabase.curated?(fact.id)
        sync_fact_to_belief(world_id, fact, is_curated)
      end)
      |> Enum.sum()

    Logger.info("Synced facts to beliefs", %{facts_synced: created, layer: layer, world_id: world_id})
    {:ok, created}
  end

  defp sync_fact_to_belief(world_id, fact, true = _is_curated) do
    verification_source = fact.verification_source || "curated_fact_database"

    belief =
      Belief.new(:world, normalize_entity(fact.entity), fact.fact,
        source: :curated_fact,
        source_authority: :curated_fact,
        confidence: 1.0,
        provenance: ["curated_fact_database", verification_source],
        metadata: %{
          fact_id: fact.id,
          category: fact.category,
          immutable: true
        }
      )

    case BeliefStore.add_belief(world_id, belief) do
      {:ok, belief_id} ->
        case JTMS.create_premise(world_id, "curated_fact:#{belief_id}",
               metadata: %{fact_id: fact.id, immutable: true}
             ) do
          {:ok, node_id} ->
            BeliefStore.link_to_node(world_id, belief_id, node_id)
            1

          _ ->
            1
        end

      _ ->
        0
    end
  end

  defp sync_fact_to_belief(world_id, fact, false = _is_curated) do
    verification_source = fact.verification_source || "learned_fact_database"

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

    case BeliefStore.add_belief(world_id, belief) do
      {:ok, belief_id} ->
        if fact.confidence >= 0.9 do
          case JTMS.create_premise(world_id, "fact:#{belief_id}", metadata: %{fact_id: fact.id}) do
            {:ok, node_id} ->
              BeliefStore.link_to_node(world_id, belief_id, node_id)
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
  end

  @doc "Checks if a fact contradicts any existing beliefs or facts.\n\nUses the JTMS to check for contradictions.\n"
  def check_contradiction(world_id, entity, fact_text) do
    world_id = world_id!(world_id, "check_contradiction")

    case BeliefStore.query_beliefs(world_id, subject: :world, predicate: normalize_entity(entity)) do
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

  @doc """
  The predicate a fact's belief about `entity` is held under: the entity
  lowercased, its spaces made underscores, as `sync_facts_to_beliefs/2`
  keys it. `:none` when no atom of that name exists, so that no belief can be
  held under it.
  """
  @spec belief_predicate(String.t()) :: {:ok, atom()} | :none
  def belief_predicate(entity) when is_binary(entity) do
    name = entity |> String.trim() |> String.downcase() |> String.replace(" ", "_")

    try do
      {:ok, String.to_existing_atom(name)}
    rescue
      ArgumentError -> :none
    end
  end

  @doc """
  How what is known answers a polar question: a statement that states the
  question's proposition (`:confirming`) or states its negation
  (`:disconfirming`), with that statement, or nil when no statement does.

  `question_tags` are the question's words with their UD part-of-speech tags
  (`Brain.ML.POSTagger.predict/2`), and `statements` the texts known about it
  (`Brain.Analysis.ChunkAnalysis.statements/2`). A polar question opens with
  an auxiliary ("Is Paris the capital of France?"); its proposition is said by
  putting the auxiliary back after the subject. A statement divided in two by
  that same auxiliary (`A aux B`) states the proposition when the question's
  other words are A then B ("Paris is the capital of France"), or B then A
  when A and B each name one thing, so that the statement says they are the
  same thing, which is said either way round ("The capital of France is
  Paris"). A phrase names one thing when it opens with the definite article
  (`Brain.Lexicon.ClosedClass.definite_article?/1`) or is a name, every word
  a proper noun. Words are compared whole, ignoring case and punctuation:
  there is no partial match, synonym or paraphrase, so "France's capital is
  Paris" states nothing about the question.

  A statement that states it with a negation ("Paris is not the capital of
  Germany") disconfirms the proposition, one without confirms it. A negated
  question ("Isn't Paris the capital of France?") is answered by no
  statement: which answer it expects is not read from its words. When
  statements both confirm and disconfirm it, nothing decides between them,
  and the alignment is nil.

  Raises when the POS tagger cannot be loaded.
  """
  @spec polar_alignment(String.t(), [{String.t(), String.t()}], [String.t()]) ::
          %{alignment: :confirming | :disconfirming, statement: String.t()} | nil
  def polar_alignment(question, question_tags, statements)
      when is_binary(question) and is_list(question_tags) and is_list(statements) do
    case words(question_tags) do
      [{auxiliary, "AUX"} | [_ | _] = rest] ->
        if LinguisticData.has_negation?(question) do
          nil
        else
          proposition = Enum.map(rest, &lowercase/1)
          model = pos_model!()

          statements
          |> Enum.flat_map(fn statement ->
            case statement_alignment(statement, auxiliary, proposition, model) do
              nil -> []
              alignment -> [{alignment, statement}]
            end
          end)
          |> decided_alignment()
        end

      _not_polar ->
        nil
    end
  end

  defp decided_alignment(alignments) do
    case alignments |> Enum.map(&elem(&1, 0)) |> Enum.uniq() do
      [alignment] -> %{alignment: alignment, statement: alignments |> hd() |> elem(1)}
      _none_or_both -> nil
    end
  end

  defp statement_alignment(statement, auxiliary, proposition, model) when is_binary(statement) do
    tagged =
      statement
      |> Tokenizer.expand_contractions()
      |> Tokenizer.tokenize_words()
      |> POSTagger.predict(model)
      |> words()

    {negations, core} = Enum.split_with(tagged, fn {word, _tag} -> LinguisticData.negation?(word) end)
    auxiliary = String.downcase(auxiliary)
    last = length(core) - 1

    stated? =
      core
      |> Enum.with_index()
      |> Enum.filter(fn {{word, tag}, index} ->
        tag == "AUX" and String.downcase(word) == auxiliary and index > 0 and index < last
      end)
      |> Enum.any?(fn {_auxiliary, index} ->
        {before, [_auxiliary | after_auxiliary]} = Enum.split(core, index)
        words_before = Enum.map(before, &lowercase/1)
        words_after = Enum.map(after_auxiliary, &lowercase/1)

        proposition == words_before ++ words_after or
          (proposition == words_after ++ words_before and names_one?(before) and names_one?(after_auxiliary))
      end)

    cond do
      not stated? -> nil
      negations == [] -> :confirming
      true -> :disconfirming
    end
  end

  defp names_one?([{first, _tag} | _] = phrase) do
    ClosedClass.definite_article?(first) or Enum.all?(phrase, fn {_word, tag} -> tag == "PROPN" end)
  end

  defp words(tagged), do: Enum.reject(tagged, fn {_word, tag} -> tag == "PUNCT" end)

  defp lowercase({word, _tag}), do: String.downcase(word)

  defp pos_model! do
    case POSTagger.get_model() do
      {:ok, model} ->
        model

      {:error, reason} ->
        raise "FactDatabase.Integration: the POS tagger could not be loaded to read a statement's " <>
                "parts of speech, which decide whether it answers a polar question: #{inspect(reason)}"
    end
  end

  defp store_learned_fact(%Fact{} = fact) do
    if FactDatabase.curated?(fact.id) do
      Logger.warning("Rejected attempt to overwrite curated fact via Integration",
        fact_id: fact.id
      )

      {:error, :curated_fact_immutable}
    else
      Brain.AtlasIntegration.persist_learned_fact(fact)
      FactDatabase.reload()
    end
  end

  defp register_fact_with_jtms(world_id, fact_id, entity, fact_text, confidence) do
    if Process.whereis(JTMS) do
      node_type =
        if confidence >= 0.9 do
          :premise
        else
          :assumption
        end

      case JTMS.create_node(world_id, "fact:#{fact_id}",
             node_type: node_type,
             metadata: %{entity: entity, fact: fact_text, fact_id: fact_id}
           ) do
        {:ok, node_id} ->
          if node_type == :assumption do
            JTMS.enable_assumption(world_id, node_id)
          end

          node_id

        _ ->
          nil
      end
    else
      nil
    end
  end

  defp create_fact_belief(world_id, entity, fact_text, confidence, verification_source, node_id) do
    belief =
      Belief.new(:world, normalize_entity(entity), fact_text,
        source: :learned,
        confidence: confidence,
        provenance: ["fact_database", verification_source],
        node_id: node_id,
        metadata: %{category: "learned"}
      )

    case BeliefStore.add_belief(world_id, belief) do
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

  defp contradicts?(text1, text2),
    do: Brain.Knowledge.ContradictionDetector.contradicts?(text1, text2)

  defp normalize_entity(entity) when is_binary(entity) do
    normalized = entity |> String.downcase() |> String.replace(" ", "_")
    String.to_existing_atom(normalized)
  rescue
    ArgumentError -> :unknown
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

  defp world_id!(world_id, _function) when is_binary(world_id) and world_id != "", do: world_id

  defp world_id!(other, function) do
    raise ArgumentError,
          "FactDatabase.Integration.#{function} needs the world whose beliefs it uses, got: #{inspect(other)}"
  end
end
