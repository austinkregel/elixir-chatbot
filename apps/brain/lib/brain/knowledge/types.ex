defmodule Brain.Knowledge.Types do
  @moduledoc """
  Type definitions for the Knowledge Expansion System.

  Provides structs for representing research goals, findings, source information,
  review candidates, learning sessions, and scientific investigation types.

  ## Scientific Method Model

  The knowledge expansion system follows the scientific method:

  1. **Observation** → User inputs, training data
  2. **Hypothesis** → Testable claims derived from questions
  3. **Investigation** → Systematic evidence gathering
  4. **Evidence** → Findings from sources (Finding struct)
  5. **Falsification** → Contradicting evidence falsifies hypotheses
  6. **Support** → Agreeing evidence supports hypotheses
  7. **Accumulation** → Knowledge builds through multiple investigations

  Key principle: We cannot prove a hypothesis true, only support it with evidence
  or falsify it with contradicting evidence.
  """

  defmodule SourceInfo do
    @moduledoc """
    Information about a content source including reliability and bias metrics.
    """

    @type bias_rating ::
            :left | :center_left | :center | :center_right | :right | :unknown

    @type trust_tier :: :verified | :neutral | :untrusted | :blocked

    @type t :: %__MODULE__{
            url: String.t(),
            domain: String.t(),
            title: String.t() | nil,
            fetched_at: DateTime.t() | nil,
            reliability_score: float(),
            bias_rating: bias_rating(),
            trust_tier: trust_tier()
          }

    @enforce_keys [:url, :domain]
    defstruct [
      :url,
      :domain,
      :title,
      :fetched_at,
      reliability_score: 0.5,
      bias_rating: :unknown,
      trust_tier: :neutral
    ]

    @doc """
    Creates a new SourceInfo from a URL.
    """
    def new(url, opts \\ []) when is_binary(url) do
      domain = extract_domain(url)

      %__MODULE__{
        url: url,
        domain: domain,
        title: Keyword.get(opts, :title),
        fetched_at: Keyword.get(opts, :fetched_at, DateTime.utc_now()),
        reliability_score: Keyword.get(opts, :reliability_score, 0.5),
        bias_rating: Keyword.get(opts, :bias_rating, :unknown),
        trust_tier: Keyword.get(opts, :trust_tier, :neutral)
      }
    end

    @doc """
    Extracts the domain from a URL.
    """
    def extract_domain(url) when is_binary(url) do
      case URI.parse(url) do
        %URI{host: host} when is_binary(host) ->
          # Remove www. prefix if present
          host
          |> String.replace_leading("www.", "")
          |> String.downcase()

        _ ->
          # Fallback for malformed URLs
          url
          |> String.downcase()
          |> String.replace_leading("http://", "")
          |> String.replace_leading("https://", "")
          |> String.split("/")
          |> List.first()
          |> String.replace_leading("www.", "")
      end
    end
  end

  defmodule Finding do
    @moduledoc """
    A factual claim extracted from a source.
    """

    alias Brain.Knowledge.Types.SourceInfo

    @type t :: %__MODULE__{
            id: String.t(),
            claim: String.t(),
            entity: String.t(),
            entity_type: String.t() | nil,
            source: SourceInfo.t(),
            raw_context: String.t(),
            extracted_at: DateTime.t(),
            confidence: float(),
            corroboration_group: String.t() | nil,
            embedding: [float()] | nil
          }

    @enforce_keys [:id, :claim, :entity, :source]
    defstruct [
      :id,
      :claim,
      :entity,
      :entity_type,
      :source,
      :corroboration_group,
      :embedding,
      raw_context: "",
      extracted_at: nil,
      confidence: 0.5
    ]

    @doc """
    Creates a new Finding with a generated ID.
    """
    def new(claim, entity, source, opts \\ []) do
      %__MODULE__{
        id: generate_id(),
        claim: claim,
        entity: entity,
        entity_type: Keyword.get(opts, :entity_type),
        source: source,
        raw_context: Keyword.get(opts, :raw_context, ""),
        extracted_at: Keyword.get(opts, :extracted_at, DateTime.utc_now()),
        confidence: Keyword.get(opts, :confidence, 0.5),
        corroboration_group: Keyword.get(opts, :corroboration_group),
        embedding: Keyword.get(opts, :embedding)
      }
    end

    defp generate_id do
      :crypto.strong_rand_bytes(12) |> Base.url_encode64(padding: false)
    end
  end

  defmodule Hypothesis do
    @moduledoc """
    A testable claim derived from a research question.

    Following the scientific method, a hypothesis must be:
    - **Falsifiable**: Can be proven false with evidence
    - **Testable**: Evidence can be gathered to evaluate it
    - **Specific**: Clear enough to test

    A hypothesis cannot be "proven true" - it can only be supported
    by evidence or falsified by contradicting evidence.

    ## Predictions

    Each hypothesis includes a prediction - an "If/Then" statement that
    describes the expected results if the hypothesis is true:
    
    > "If the hypothesis is true, then the results of the experiment will be..."

    If predictions are confirmed, the hypothesis is supported.
    If predictions are not supported, the hypothesis is falsified.

    ## Replication

    Repeating the experiment (finding multiple sources) increases confidence.
    We should not expect exactly the same answer each time - variation is normal.
    Replication enables us to see variation and obtain an average result.

    ## States

    - `:untested` - No evidence gathered yet
    - `:testing` - Currently gathering evidence
    - `:supported` - Evidence supports the hypothesis (not proven!)
    - `:falsified` - Contradicting evidence disproves the hypothesis
    - `:inconclusive` - Mixed or insufficient evidence

    ## Example

        hypothesis = Hypothesis.new(
          "Paris is the capital of France",
          derived_from: "What is the capital of France?",
          prediction: "If Paris is the capital, then authoritative sources will confirm this."
        )
    """

    alias Brain.Knowledge.Types.Finding

    @type status :: :untested | :testing | :supported | :falsified | :inconclusive
    @type confidence_level :: :none | :low | :moderate | :high | :very_high

    @type t :: %__MODULE__{
            id: String.t(),
            claim: String.t(),
            entity: String.t() | nil,
            derived_from: String.t() | nil,
            prediction: String.t() | nil,
            status: status(),
            supporting_evidence: [Finding.t()],
            contradicting_evidence: [Finding.t()],
            confidence: float(),
            confidence_level: confidence_level(),
            source_count: non_neg_integer(),
            replication_count: non_neg_integer(),
            tested_at: DateTime.t() | nil,
            created_at: DateTime.t()
          }

    @enforce_keys [:id, :claim]
    defstruct [
      :id,
      :claim,
      :entity,
      :derived_from,
      :prediction,
      :tested_at,
      :created_at,
      status: :untested,
      supporting_evidence: [],
      contradicting_evidence: [],
      confidence: 0.0,
      confidence_level: :none,
      source_count: 0,
      replication_count: 0
    ]

    @doc """
    Creates a new hypothesis from a claim.

    ## Options
      - :entity - The entity this hypothesis is about
      - :derived_from - The question that generated this hypothesis
      - :prediction - The expected outcome if hypothesis is true (If/Then)
    """
    def new(claim, opts \\ []) when is_binary(claim) do
      prediction = Keyword.get(opts, :prediction) || generate_prediction(claim)

      %__MODULE__{
        id: generate_id(),
        claim: claim,
        entity: Keyword.get(opts, :entity),
        derived_from: Keyword.get(opts, :derived_from),
        prediction: prediction,
        status: :untested,
        created_at: DateTime.utc_now()
      }
    end

    # Generate a default prediction from the claim
    defp generate_prediction(claim) do
      "If #{claim} is true, then independent sources will confirm this claim."
    end

    @doc """
    Adds supporting evidence to a hypothesis.

    Supporting evidence increases confidence but does NOT prove the hypothesis.
    """
    def add_supporting_evidence(%__MODULE__{} = hypothesis, %Finding{} = finding) do
      # Track replication: evidence from same domain counts as replication
      is_replication = Enum.any?(hypothesis.supporting_evidence, fn existing ->
        existing.source.domain == finding.source.domain
      end)

      updated = %{hypothesis |
        supporting_evidence: [finding | hypothesis.supporting_evidence],
        source_count: hypothesis.source_count + 1,
        replication_count: if(is_replication, do: hypothesis.replication_count + 1, else: hypothesis.replication_count),
        status: :testing
      }
      recalculate_confidence(updated)
    end

    @doc """
    Adds contradicting evidence to a hypothesis.

    Contradicting evidence from reliable sources can falsify the hypothesis.
    """
    def add_contradicting_evidence(%__MODULE__{} = hypothesis, %Finding{} = finding) do
      updated = %{hypothesis |
        contradicting_evidence: [finding | hypothesis.contradicting_evidence],
        source_count: hypothesis.source_count + 1,
        status: :testing
      }
      recalculate_confidence(updated)
    end

    @doc """
    Evaluates the hypothesis based on accumulated evidence.

    Returns the hypothesis with updated status:
    - `:supported` if supporting evidence outweighs contradicting
    - `:falsified` if reliable contradicting evidence exists
    - `:inconclusive` if evidence is mixed or insufficient
    """
    def evaluate(%__MODULE__{} = hypothesis) do
      supporting_count = length(hypothesis.supporting_evidence)
      contradicting_count = length(hypothesis.contradicting_evidence)

      # Calculate average reliability of contradicting evidence
      contradicting_reliability = average_reliability(hypothesis.contradicting_evidence)

      cond do
        # Falsified: reliable contradicting evidence exists
        contradicting_count > 0 and contradicting_reliability >= 0.6 ->
          %{hypothesis | status: :falsified, tested_at: DateTime.utc_now()}

        # Supported: multiple supporting sources, no contradictions
        supporting_count >= 2 and contradicting_count == 0 ->
          %{hypothesis | status: :supported, tested_at: DateTime.utc_now()}

        # Mixed evidence
        supporting_count > 0 and contradicting_count > 0 ->
          if supporting_count > contradicting_count * 2 do
            %{hypothesis | status: :supported, tested_at: DateTime.utc_now()}
          else
            %{hypothesis | status: :inconclusive, tested_at: DateTime.utc_now()}
          end

        # Insufficient evidence
        supporting_count < 2 ->
          %{hypothesis | status: :inconclusive, tested_at: DateTime.utc_now()}

        true ->
          %{hypothesis | status: :inconclusive, tested_at: DateTime.utc_now()}
      end
    end

    @doc """
    Returns true if the hypothesis can be promoted to a fact.

    A hypothesis can become a fact only if:
    1. It is supported (not falsified)
    2. It has high confidence (>= 0.7)
    3. It has multiple independent sources (>= 2)
    """
    def promotable?(%__MODULE__{} = hypothesis) do
      hypothesis.status == :supported and
        hypothesis.confidence >= 0.7 and
        count_unique_sources(hypothesis.supporting_evidence) >= 2
    end

    # Private functions

    defp recalculate_confidence(%__MODULE__{} = hypothesis) do
      supporting_count = length(hypothesis.supporting_evidence)
      contradicting_count = length(hypothesis.contradicting_evidence)
      total = supporting_count + contradicting_count

      if total == 0 do
        %{hypothesis | confidence: 0.0, confidence_level: :none}
      else
        # Base confidence from pass rate (what percentage of evidence supports)
        pass_rate = supporting_count / total

        # Scale by source reliability
        reliability_factor = average_reliability(hypothesis.supporting_evidence)

        # Bonus for having multiple independent sources
        unique_sources = count_unique_sources(hypothesis.supporting_evidence)
        source_diversity_bonus = if unique_sources >= 2, do: 0.1, else: 0.0

        # Penalty for low sample size (need at least 5 samples for reliable results)
        sample_size_factor = min(total / 5.0, 1.0)

        # Combined confidence:
        # - Pass rate is the primary factor (60% weight)
        # - Reliability adjusts quality (20% weight)
        # - Sample size matters (10% weight)
        # - Source diversity bonus (10% weight)
        confidence =
          (pass_rate * 0.6 +
           reliability_factor * 0.2 +
           sample_size_factor * 0.1 +
           source_diversity_bonus)
          |> max(0.0)
          |> min(1.0)

        level = confidence_to_level(confidence)

        %{hypothesis | confidence: confidence, confidence_level: level}
      end
    end

    defp confidence_to_level(confidence) do
      cond do
        confidence >= 0.85 -> :very_high
        confidence >= 0.70 -> :high
        confidence >= 0.50 -> :moderate
        confidence >= 0.25 -> :low
        true -> :none
      end
    end

    defp average_reliability(findings) when is_list(findings) do
      if findings == [] do
        0.0
      else
        findings
        |> Enum.map(fn f -> f.source.reliability_score end)
        |> Enum.sum()
        |> Kernel./(length(findings))
      end
    end

    defp count_unique_sources(findings) when is_list(findings) do
      findings
      |> Enum.map(fn f -> f.source.domain end)
      |> Enum.uniq()
      |> length()
    end

    defp generate_id do
      :crypto.strong_rand_bytes(12) |> Base.url_encode64(padding: false)
    end
  end

  defmodule Investigation do
    @moduledoc """
    Represents a scientific investigation testing one or more hypotheses.

    An investigation follows the scientific method:
    1. Formulate hypotheses from questions
    2. Gather evidence from independent sources
    3. Evaluate hypotheses against evidence
    4. Report conclusions (supported, falsified, or inconclusive)

    ## Experimental Variables

    From the scientific method, experiments involve three types of variables:

    - **Independent Variable**: What we vary (the sources we query)
    - **Dependent Variable**: What we measure (the findings/claims extracted)
    - **Constants**: What we hold fixed (NLP pipeline, corroboration rules)

    ## Control Treatment

    A control treatment provides a baseline for comparison. In our context,
    this could be:
    - Existing facts in the database (do new findings agree?)
    - Known reliable sources (Wikipedia, encyclopedias)

    ## Replication

    We require multiple independent sources (replication) to increase
    confidence. Variation between sources is normal - replication helps
    us see this variation and obtain a consensus.

    ## Key Principles

    - **Falsifiability**: Hypotheses can be disproven by contradicting evidence
    - **Cannot Prove True**: Only support with evidence, never absolute proof
    - **Accumulation**: Knowledge builds through many investigations
    """

    alias Brain.Knowledge.Types.{Hypothesis, Finding}

    @type status :: :planning | :gathering_evidence | :evaluating | :concluded
    @type conclusion :: :hypotheses_supported | :hypotheses_falsified | :inconclusive | :mixed

    @type t :: %__MODULE__{
            id: String.t(),
            topic: String.t(),
            hypotheses: [Hypothesis.t()],
            evidence: [Finding.t()],
            control_evidence: [Finding.t()],
            independent_variable: String.t(),
            dependent_variable: String.t(),
            constants: [String.t()],
            status: status(),
            conclusion: conclusion() | nil,
            started_at: DateTime.t(),
            concluded_at: DateTime.t() | nil,
            methodology_notes: String.t() | nil
          }

    @enforce_keys [:id, :topic]
    defstruct [
      :id,
      :topic,
      :concluded_at,
      :methodology_notes,
      hypotheses: [],
      evidence: [],
      control_evidence: [],
      independent_variable: "source",
      dependent_variable: "claim",
      constants: ["nlp_pipeline", "corroboration_threshold", "similarity_threshold"],
      status: :planning,
      conclusion: nil,
      started_at: nil
    ]

    @doc """
    Creates a new investigation for a topic.

    ## Options
      - :hypotheses - Pre-formulated hypotheses
      - :independent_variable - What we're varying (default: "source")
      - :dependent_variable - What we're measuring (default: "claim")
      - :constants - What we hold fixed (default: NLP pipeline settings)
    """
    def new(topic, opts \\ []) when is_binary(topic) do
      %__MODULE__{
        id: generate_id(),
        topic: topic,
        hypotheses: Keyword.get(opts, :hypotheses, []),
        independent_variable: Keyword.get(opts, :independent_variable, "source"),
        dependent_variable: Keyword.get(opts, :dependent_variable, "claim"),
        constants: Keyword.get(opts, :constants, ["nlp_pipeline", "corroboration_threshold"]),
        started_at: DateTime.utc_now(),
        status: :planning
      }
    end

    @doc """
    Adds a hypothesis to the investigation.
    """
    def add_hypothesis(%__MODULE__{} = investigation, %Hypothesis{} = hypothesis) do
      %{investigation | hypotheses: investigation.hypotheses ++ [hypothesis]}
    end

    @doc """
    Formulates hypotheses from a list of questions.
    """
    def formulate_hypotheses(%__MODULE__{} = investigation, questions) when is_list(questions) do
      hypotheses =
        questions
        |> Enum.map(fn question ->
          Hypothesis.new(
            question_to_claim(question),
            derived_from: question,
            entity: extract_entity_from_question(question)
          )
        end)

      %{investigation | hypotheses: investigation.hypotheses ++ hypotheses}
    end

    @doc """
    Sets the control treatment - baseline facts to compare against.

    Control evidence provides a baseline for comparison:
    - Existing facts in the database
    - Known reliable sources (encyclopedias, etc.)
    """
    def set_control(%__MODULE__{} = investigation, control_findings) when is_list(control_findings) do
      %{investigation | control_evidence: control_findings}
    end

    @doc """
    Records evidence and associates it with relevant hypotheses.
    """
    def record_evidence(%__MODULE__{} = investigation, findings) when is_list(findings) do
      updated_evidence = investigation.evidence ++ findings

      # Associate each finding with relevant hypotheses
      updated_hypotheses =
        investigation.hypotheses
        |> Enum.map(fn hypothesis ->
          associate_evidence(hypothesis, findings)
        end)

      %{investigation |
        evidence: updated_evidence,
        hypotheses: updated_hypotheses,
        status: :gathering_evidence
      }
    end

    @doc """
    Evaluates all hypotheses and concludes the investigation.
    """
    def conclude(%__MODULE__{} = investigation) do
      # Evaluate each hypothesis
      evaluated =
        investigation.hypotheses
        |> Enum.map(&Hypothesis.evaluate/1)

      # Determine overall conclusion
      conclusion = determine_conclusion(evaluated)

      %{investigation |
        hypotheses: evaluated,
        status: :concluded,
        conclusion: conclusion,
        concluded_at: DateTime.utc_now()
      }
    end

    @doc """
    Returns hypotheses that can be promoted to facts.
    """
    def promotable_hypotheses(%__MODULE__{} = investigation) do
      investigation.hypotheses
      |> Enum.filter(&Hypothesis.promotable?/1)
    end

    @doc """
    Returns a summary of the investigation results.
    """
    def summary(%__MODULE__{} = investigation) do
      supported = Enum.count(investigation.hypotheses, &(&1.status == :supported))
      falsified = Enum.count(investigation.hypotheses, &(&1.status == :falsified))
      inconclusive = Enum.count(investigation.hypotheses, &(&1.status == :inconclusive))

      # Count unique sources (independent variables)
      unique_sources =
        investigation.evidence
        |> Enum.map(& &1.source.domain)
        |> Enum.uniq()
        |> length()

      # Count total replications across hypotheses
      total_replications =
        investigation.hypotheses
        |> Enum.map(& &1.replication_count)
        |> Enum.sum()

      %{
        topic: investigation.topic,
        total_hypotheses: length(investigation.hypotheses),
        supported: supported,
        falsified: falsified,
        inconclusive: inconclusive,
        evidence_count: length(investigation.evidence),
        control_evidence_count: length(investigation.control_evidence),
        unique_sources: unique_sources,
        replications: total_replications,
        independent_variable: investigation.independent_variable,
        dependent_variable: investigation.dependent_variable,
        constants: investigation.constants,
        conclusion: investigation.conclusion,
        promotable: length(promotable_hypotheses(investigation))
      }
    end

    # Private functions

    defp question_to_claim(question) when is_binary(question) do
      # Transform question into a claim statement
      # This is a simplified heuristic - the actual claim will be refined by evidence
      question
      |> String.trim_trailing("?")
      |> String.trim()
    end

    defp extract_entity_from_question(question) do
      # Use tokenizer to extract likely entity (nouns/proper nouns)
      # Simplified: return the question for now, will be refined by pipeline
      question
    end

    defp associate_evidence(%Hypothesis{} = hypothesis, findings) do
      alias Brain.ML.Tokenizer

      hypothesis_tokens =
        hypothesis.claim
        |> Tokenizer.tokenize_words()
        |> MapSet.new()

      Enum.reduce(findings, hypothesis, fn finding, hyp ->
        finding_tokens =
          finding.claim
          |> Tokenizer.tokenize_words()
          |> MapSet.new()

        # Check token overlap to determine relevance
        overlap = MapSet.intersection(hypothesis_tokens, finding_tokens) |> MapSet.size()
        min_size = min(MapSet.size(hypothesis_tokens), MapSet.size(finding_tokens))

        relevance = if min_size > 0, do: overlap / min_size, else: 0

        if relevance >= 0.3 do
          # Check if this evidence supports or contradicts
          if evidence_contradicts?(hyp.claim, finding.claim) do
            Hypothesis.add_contradicting_evidence(hyp, finding)
          else
            Hypothesis.add_supporting_evidence(hyp, finding)
          end
        else
          hyp
        end
      end)
    end

    defp evidence_contradicts?(claim, finding_claim) do
      # Simplified contradiction detection using negation patterns
      c1 = String.downcase(claim)
      c2 = String.downcase(finding_claim)

      negation_words = ["not", "no", "never", "none", "cannot", "isn't", "aren't", "wasn't", "weren't"]

      c1_negated = Enum.any?(negation_words, &String.contains?(c1, &1))
      c2_negated = Enum.any?(negation_words, &String.contains?(c2, &1))

      # XOR: one has negation, other doesn't
      c1_negated != c2_negated
    end

    defp determine_conclusion(hypotheses) do
      supported = Enum.count(hypotheses, &(&1.status == :supported))
      falsified = Enum.count(hypotheses, &(&1.status == :falsified))
      total = length(hypotheses)

      cond do
        total == 0 ->
          :inconclusive

        falsified == total ->
          :hypotheses_falsified

        supported == total ->
          :hypotheses_supported

        supported > falsified ->
          :mixed

        true ->
          :inconclusive
      end
    end

    defp generate_id do
      :crypto.strong_rand_bytes(12) |> Base.url_encode64(padding: false)
    end
  end

  defmodule ResearchGoal do
    @moduledoc """
    A research objective for the Learning Center to pursue.

    ## Scientific Method Integration

    Research goals now support the scientific method by:
    - Generating hypotheses from questions
    - Creating investigations to test those hypotheses
    - Tracking the scientific outcome (supported/falsified)
    """

    alias Brain.Knowledge.Types.{Hypothesis, Investigation}

    @type priority :: :low | :normal | :high
    @type status :: :pending | :in_progress | :completed | :failed

    @type t :: %__MODULE__{
            id: String.t(),
            topic: String.t(),
            questions: [String.t()],
            constraints: map(),
            priority: priority(),
            created_at: DateTime.t(),
            status: status()
          }

    @enforce_keys [:id, :topic]
    defstruct [
      :id,
      :topic,
      :created_at,
      questions: [],
      constraints: %{},
      priority: :normal,
      status: :pending
    ]

    @doc """
    Creates a new ResearchGoal.

    ## Options
      - :questions - List of specific questions to answer
      - :constraints - Map of constraints (e.g., %{min_sources: 2, max_age_days: 30})
      - :priority - :low | :normal | :high
    """
    def new(topic, opts \\ []) when is_binary(topic) do
      %__MODULE__{
        id: generate_id(),
        topic: topic,
        questions: Keyword.get(opts, :questions, []),
        constraints: Keyword.get(opts, :constraints, %{}),
        priority: Keyword.get(opts, :priority, :normal),
        created_at: DateTime.utc_now(),
        status: :pending
      }
    end

    @doc """
    Updates the status of a goal.
    """
    def update_status(%__MODULE__{} = goal, new_status)
        when new_status in [:pending, :in_progress, :completed, :failed] do
      %{goal | status: new_status}
    end

    @doc """
    Generates hypotheses from the goal's questions.

    Each question is transformed into a testable hypothesis.
    If no questions exist, a hypothesis is generated from the topic.
    """
    def generate_hypotheses(%__MODULE__{} = goal) do
      if goal.questions == [] do
        # Generate default hypotheses from topic
        [
          Hypothesis.new(goal.topic,
            entity: goal.topic,
            derived_from: "What is #{goal.topic}?"
          )
        ]
      else
        goal.questions
        |> Enum.map(fn question ->
          Hypothesis.new(
            question_to_claim(question),
            entity: extract_entity(question, goal.topic),
            derived_from: question
          )
        end)
      end
    end

    @doc """
    Creates a scientific investigation from this goal.

    The investigation will:
    1. Formulate hypotheses from questions
    2. Be ready to gather evidence
    3. Track the scientific outcome
    """
    def to_investigation(%__MODULE__{} = goal) do
      hypotheses = generate_hypotheses(goal)

      Investigation.new(goal.topic,
        hypotheses: hypotheses
      )
    end

    # Private helpers

    defp question_to_claim(question) do
      # Transform question to claim statement
      # Remove question mark and "what is", "where is" etc.
      question
      |> String.trim_trailing("?")
      |> String.trim()
      |> remove_question_prefix()
    end

    defp remove_question_prefix(text) do
      text
      |> String.replace(~r/^(what is |what are |where is |who is |when did |how does )/i, "")
      |> String.trim()
    end

    defp extract_entity(question, default_topic) do
      # Try to extract entity from question using simple heuristics
      # For more sophisticated extraction, use the NLP pipeline
      words =
        question
        |> String.downcase()
        |> String.replace(~r/[^\w\s]/, "")
        |> String.split()

      # Filter out common question words
      stop_words = ~w(what is are where who when how does did the a an of in to)

      content_words =
        words
        |> Enum.reject(&(&1 in stop_words))

      if content_words != [] do
        Enum.join(content_words, " ")
      else
        default_topic
      end
    end

    defp generate_id do
      :crypto.strong_rand_bytes(12) |> Base.url_encode64(padding: false)
    end
  end

  defmodule ReviewCandidate do
    @moduledoc """
    A finding that has been vetted and is ready for admin review.
    """

    alias Brain.Knowledge.Types.{Finding, SourceInfo}

    @type status :: :pending | :approved | :rejected | :deferred

    @type t :: %__MODULE__{
            id: String.t(),
            finding: Finding.t(),
            corroborating_sources: [SourceInfo.t()],
            conflicting_findings: [Finding.t()],
            existing_contradictions: [map()],
            aggregate_confidence: float(),
            status: status(),
            reviewed_at: DateTime.t() | nil,
            reviewer_notes: String.t() | nil,
            session_id: String.t() | nil
          }

    @enforce_keys [:id, :finding]
    defstruct [
      :id,
      :finding,
      :reviewed_at,
      :reviewer_notes,
      :session_id,
      corroborating_sources: [],
      conflicting_findings: [],
      existing_contradictions: [],
      aggregate_confidence: 0.5,
      status: :pending
    ]

    @doc """
    Creates a new ReviewCandidate from a finding.
    """
    def new(%Finding{} = finding, opts \\ []) do
      %__MODULE__{
        id: generate_id(),
        finding: finding,
        corroborating_sources: Keyword.get(opts, :corroborating_sources, []),
        conflicting_findings: Keyword.get(opts, :conflicting_findings, []),
        existing_contradictions: Keyword.get(opts, :existing_contradictions, []),
        aggregate_confidence: Keyword.get(opts, :aggregate_confidence, finding.confidence),
        status: :pending,
        session_id: Keyword.get(opts, :session_id)
      }
    end

    @doc """
    Marks a candidate as approved.
    """
    def approve(%__MODULE__{} = candidate, notes \\ nil) do
      %{candidate | status: :approved, reviewed_at: DateTime.utc_now(), reviewer_notes: notes}
    end

    @doc """
    Marks a candidate as rejected.
    """
    def reject(%__MODULE__{} = candidate, notes \\ nil) do
      %{candidate | status: :rejected, reviewed_at: DateTime.utc_now(), reviewer_notes: notes}
    end

    @doc """
    Marks a candidate as deferred for later review.
    """
    def defer(%__MODULE__{} = candidate, notes \\ nil) do
      %{candidate | status: :deferred, reviewed_at: DateTime.utc_now(), reviewer_notes: notes}
    end

    defp generate_id do
      :crypto.strong_rand_bytes(12) |> Base.url_encode64(padding: false)
    end
  end

  defmodule LearningSession do
    @moduledoc """
    Represents an active or completed learning session.

    ## Scientific Method Integration

    A LearningSession now tracks the full scientific investigation lifecycle:
    - **Investigations**: Scientific investigations with hypotheses
    - **Hypotheses Tested**: Total hypotheses evaluated
    - **Hypotheses Supported**: Hypotheses backed by evidence
    - **Hypotheses Falsified**: Hypotheses disproven by contradicting evidence

    This allows tracking the accumulation of scientific knowledge over time.
    """

    alias Brain.Knowledge.Types.{ResearchGoal, Investigation}

    @type status :: :active | :completed | :cancelled

    @type t :: %__MODULE__{
            id: String.t(),
            goals: [ResearchGoal.t()],
            investigations: [Investigation.t()],
            started_at: DateTime.t(),
            completed_at: DateTime.t() | nil,
            findings_count: non_neg_integer(),
            approved_count: non_neg_integer(),
            rejected_count: non_neg_integer(),
            hypotheses_tested: non_neg_integer(),
            hypotheses_supported: non_neg_integer(),
            hypotheses_falsified: non_neg_integer(),
            status: status(),
            topic: String.t() | nil
          }

    @enforce_keys [:id, :started_at]
    defstruct [
      :id,
      :started_at,
      :completed_at,
      :topic,
      goals: [],
      investigations: [],
      findings_count: 0,
      approved_count: 0,
      rejected_count: 0,
      hypotheses_tested: 0,
      hypotheses_supported: 0,
      hypotheses_falsified: 0,
      status: :active
    ]

    @doc """
    Creates a new LearningSession.
    """
    def new(opts \\ []) do
      %__MODULE__{
        id: generate_id(),
        goals: Keyword.get(opts, :goals, []),
        investigations: [],
        started_at: DateTime.utc_now(),
        status: :active,
        topic: Keyword.get(opts, :topic)
      }
    end

    @doc """
    Adds a goal to the session.
    """
    def add_goal(%__MODULE__{} = session, %ResearchGoal{} = goal) do
      %{session | goals: session.goals ++ [goal]}
    end

    @doc """
    Adds a completed investigation to the session.

    Updates hypothesis statistics based on the investigation results.
    """
    def add_investigation(%__MODULE__{} = session, %Investigation{} = investigation) do
      # Count hypothesis outcomes
      supported = Enum.count(investigation.hypotheses, &(&1.status == :supported))
      falsified = Enum.count(investigation.hypotheses, &(&1.status == :falsified))
      tested = length(investigation.hypotheses)

      %{session |
        investigations: session.investigations ++ [investigation],
        hypotheses_tested: session.hypotheses_tested + tested,
        hypotheses_supported: session.hypotheses_supported + supported,
        hypotheses_falsified: session.hypotheses_falsified + falsified
      }
    end

    @doc """
    Increments the findings count.
    """
    def record_findings(%__MODULE__{} = session, count) when is_integer(count) and count >= 0 do
      %{session | findings_count: session.findings_count + count}
    end

    @doc """
    Records an approval.
    """
    def record_approval(%__MODULE__{} = session) do
      %{session | approved_count: session.approved_count + 1}
    end

    @doc """
    Records a rejection.
    """
    def record_rejection(%__MODULE__{} = session) do
      %{session | rejected_count: session.rejected_count + 1}
    end

    @doc """
    Marks the session as completed.
    """
    def complete(%__MODULE__{} = session) do
      %{session | status: :completed, completed_at: DateTime.utc_now()}
    end

    @doc """
    Marks the session as cancelled.
    """
    def cancel(%__MODULE__{} = session) do
      %{session | status: :cancelled, completed_at: DateTime.utc_now()}
    end

    @doc """
    Returns a summary of the session's scientific outcomes.
    """
    def scientific_summary(%__MODULE__{} = session) do
      %{
        topic: session.topic,
        investigations_completed: length(session.investigations),
        hypotheses_tested: session.hypotheses_tested,
        hypotheses_supported: session.hypotheses_supported,
        hypotheses_falsified: session.hypotheses_falsified,
        support_rate: if(session.hypotheses_tested > 0,
          do: session.hypotheses_supported / session.hypotheses_tested,
          else: 0.0
        ),
        facts_approved: session.approved_count,
        facts_rejected: session.rejected_count
      }
    end

    defp generate_id do
      :crypto.strong_rand_bytes(12) |> Base.url_encode64(padding: false)
    end
  end

  defmodule SourceProfile do
    @moduledoc """
    Extended profile for a source domain including historical data.
    Used internally by SourceReliability GenServer.
    """

    @type t :: %__MODULE__{
            domain: String.t(),
            factual_accuracy: float(),
            bias_rating: atom(),
            trust_tier: atom(),
            notes: String.t() | nil,
            admin_decisions: [map()],
            last_updated: DateTime.t()
          }

    @enforce_keys [:domain]
    defstruct [
      :domain,
      :notes,
      :last_updated,
      factual_accuracy: 0.5,
      bias_rating: :unknown,
      trust_tier: :neutral,
      admin_decisions: []
    ]

    @doc """
    Creates a new SourceProfile.
    """
    def new(domain, opts \\ []) when is_binary(domain) do
      %__MODULE__{
        domain: String.downcase(domain),
        factual_accuracy: Keyword.get(opts, :factual_accuracy, 0.5),
        bias_rating: Keyword.get(opts, :bias_rating, :unknown),
        trust_tier: Keyword.get(opts, :trust_tier, :neutral),
        notes: Keyword.get(opts, :notes),
        admin_decisions: [],
        last_updated: DateTime.utc_now()
      }
    end

    @doc """
    Records an admin decision (approval or rejection) for this source.
    """
    def record_decision(%__MODULE__{} = profile, decision, opts \\ [])
        when decision in [:approved, :rejected] do
      entry = %{
        decision: decision,
        timestamp: DateTime.utc_now(),
        candidate_id: Keyword.get(opts, :candidate_id),
        notes: Keyword.get(opts, :notes)
      }

      updated_decisions = [entry | profile.admin_decisions] |> Enum.take(100)
      %{profile | admin_decisions: updated_decisions, last_updated: DateTime.utc_now()}
    end

    @doc """
    Calculates the current reliability score based on base accuracy and admin feedback.
    """
    def calculate_reliability(%__MODULE__{} = profile) do
      base = profile.factual_accuracy
      feedback_adjustment = calculate_feedback_adjustment(profile.admin_decisions)

      # Weighted combination, capped at 0.0-1.0
      (base * 0.7 + feedback_adjustment * 0.3)
      |> max(0.0)
      |> min(1.0)
    end

    defp calculate_feedback_adjustment(decisions) when is_list(decisions) do
      if decisions == [] do
        0.5
      else
        # Weight recent decisions more heavily
        {weighted_sum, total_weight} =
          decisions
          |> Enum.with_index()
          |> Enum.reduce({0.0, 0.0}, fn {decision, idx}, {sum, weight} ->
            # Exponential decay: more recent decisions have higher weight
            decay = :math.pow(0.9, idx)
            value = if decision.decision == :approved, do: 1.0, else: 0.0
            {sum + value * decay, weight + decay}
          end)

        if total_weight > 0, do: weighted_sum / total_weight, else: 0.5
      end
    end
  end
end
