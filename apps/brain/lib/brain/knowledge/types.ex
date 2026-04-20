defmodule Brain.Knowledge.Types do
  @moduledoc "Type definitions for the Knowledge Expansion System.\n\nProvides structs for representing research goals, findings, source information,\nreview candidates, learning sessions, and scientific investigation types.\n\n## Scientific Method Model\n\nThe knowledge expansion system follows the scientific method:\n\n1. **Observation** → User inputs, training data\n2. **Hypothesis** → Testable claims derived from questions\n3. **Investigation** → Systematic evidence gathering\n4. **Evidence** → Findings from sources (Finding struct)\n5. **Falsification** → Contradicting evidence falsifies hypotheses\n6. **Support** → Agreeing evidence supports hypotheses\n7. **Accumulation** → Knowledge builds through multiple investigations\n\nKey principle: We cannot prove a hypothesis true, only support it with evidence\nor falsify it with contradicting evidence.\n"

  alias Brain.Knowledge.Types
  alias Brain.LinguisticData

  defmodule SourceInfo do
    @moduledoc "Information about a content source including reliability and bias metrics.\n"

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

    @doc "Creates a new SourceInfo from a URL.\n"
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

    @doc "Extracts the domain from a URL.\n"
    def extract_domain(url) when is_binary(url) do
      case URI.parse(url) do
        %URI{host: host} when is_binary(host) ->
          host
          |> String.replace_leading("www.", "")
          |> String.downcase()

        _ ->
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
    @moduledoc "A factual claim extracted from a source.\n"

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
            embedding: [float()] | nil,
            comprehension_profile_id: String.t() | nil
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
      :comprehension_profile_id,
      raw_context: "",
      extracted_at: nil,
      confidence: 0.5
    ]

    @doc "Creates a new Finding with a generated ID.\n"
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
        embedding: Keyword.get(opts, :embedding),
        comprehension_profile_id: Keyword.get(opts, :comprehension_profile_id)
      }
    end

    defp generate_id do
      :crypto.strong_rand_bytes(12) |> Base.url_encode64(padding: false)
    end
  end

  defmodule Hypothesis do
    @moduledoc "A testable claim derived from a research question.\n\nFollowing the scientific method, a hypothesis must be:\n- **Falsifiable**: Can be proven false with evidence\n- **Testable**: Evidence can be gathered to evaluate it\n- **Specific**: Clear enough to test\n\nA hypothesis cannot be \"proven true\" - it can only be supported\nby evidence or falsified by contradicting evidence.\n\n## Predictions\n\nEach hypothesis includes a prediction - an \"If/Then\" statement that\ndescribes the expected results if the hypothesis is true:\n\n> \"If the hypothesis is true, then the results of the experiment will be...\"\n\nIf predictions are confirmed, the hypothesis is supported.\nIf predictions are not supported, the hypothesis is falsified.\n\n## Replication\n\nRepeating the experiment (finding multiple sources) increases confidence.\nWe should not expect exactly the same answer each time - variation is normal.\nReplication enables us to see variation and obtain an average result.\n\n## States\n\n- `:untested` - No evidence gathered yet\n- `:testing` - Currently gathering evidence\n- `:supported` - Evidence supports the hypothesis (not proven!)\n- `:falsified` - Contradicting evidence disproves the hypothesis\n- `:inconclusive` - Mixed or insufficient evidence\n\n## Example\n\n    hypothesis = Hypothesis.new(\n      \"Paris is the capital of France\",\n      derived_from: \"What is the capital of France?\",\n      prediction: \"If Paris is the capital, then authoritative sources will confirm this.\"\n    )\n"

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

    @doc "Creates a new hypothesis from a claim.\n\n## Options\n  - :entity - The entity this hypothesis is about\n  - :derived_from - The question that generated this hypothesis\n  - :prediction - The expected outcome if hypothesis is true (If/Then)\n"
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

    defp generate_prediction(claim) do
      "If #{claim} is true, then independent sources will confirm this claim."
    end

    @doc "Adds supporting evidence to a hypothesis.\n\nSupporting evidence increases confidence but does NOT prove the hypothesis.\n"
    def add_supporting_evidence(%__MODULE__{} = hypothesis, %Finding{} = finding) do
      is_replication =
        Enum.any?(hypothesis.supporting_evidence, fn existing ->
          existing.source.domain == finding.source.domain
        end)

      updated = %{
        hypothesis
        | supporting_evidence: [finding | hypothesis.supporting_evidence],
          source_count: hypothesis.source_count + 1,
          replication_count:
            if(is_replication) do
              hypothesis.replication_count + 1
            else
              hypothesis.replication_count
            end,
          status: :testing
      }

      recalculate_confidence(updated)
    end

    @doc "Adds contradicting evidence to a hypothesis.\n\nContradicting evidence from reliable sources can falsify the hypothesis.\n"
    def add_contradicting_evidence(%__MODULE__{} = hypothesis, %Finding{} = finding) do
      updated = %{
        hypothesis
        | contradicting_evidence: [finding | hypothesis.contradicting_evidence],
          source_count: hypothesis.source_count + 1,
          status: :testing
      }

      recalculate_confidence(updated)
    end

    @doc "Evaluates the hypothesis based on accumulated evidence.\n\nReturns the hypothesis with updated status:\n- `:supported` if supporting evidence outweighs contradicting\n- `:falsified` if reliable contradicting evidence exists\n- `:inconclusive` if evidence is mixed or insufficient\n"
    def evaluate(%__MODULE__{} = hypothesis) do
      supporting_count = length(hypothesis.supporting_evidence)
      contradicting_count = length(hypothesis.contradicting_evidence)
      contradicting_reliability = average_reliability(hypothesis.contradicting_evidence)

      cond do
        contradicting_count > 0 and contradicting_reliability >= 0.6 ->
          %{hypothesis | status: :falsified, tested_at: DateTime.utc_now()}

        supporting_count >= 2 and contradicting_count == 0 ->
          %{hypothesis | status: :supported, tested_at: DateTime.utc_now()}

        supporting_count > 0 and contradicting_count > 0 ->
          if supporting_count > contradicting_count * 2 do
            %{hypothesis | status: :supported, tested_at: DateTime.utc_now()}
          else
            %{hypothesis | status: :inconclusive, tested_at: DateTime.utc_now()}
          end

        supporting_count < 2 ->
          %{hypothesis | status: :inconclusive, tested_at: DateTime.utc_now()}

        true ->
          %{hypothesis | status: :inconclusive, tested_at: DateTime.utc_now()}
      end
    end

    @doc "Returns true if the hypothesis can be promoted to a fact.\n\nA hypothesis can become a fact only if:\n1. It is supported (not falsified)\n2. It has high confidence (>= 0.7)\n3. It has multiple independent sources (>= 2)\n"
    def promotable?(%__MODULE__{} = hypothesis) do
      hypothesis.status == :supported and
        hypothesis.confidence >= 0.7 and
        count_unique_sources(hypothesis.supporting_evidence) >= 2
    end

    defp recalculate_confidence(%__MODULE__{} = hypothesis) do
      supporting_count = length(hypothesis.supporting_evidence)
      contradicting_count = length(hypothesis.contradicting_evidence)
      total = supporting_count + contradicting_count

      if total == 0 do
        %{hypothesis | confidence: 0.0, confidence_level: :none}
      else
        pass_rate = supporting_count / total
        reliability_factor = average_reliability(hypothesis.supporting_evidence)
        unique_sources = count_unique_sources(hypothesis.supporting_evidence)

        source_diversity_bonus =
          if unique_sources >= 2 do
            0.1
          else
            0.0
          end

        sample_size_factor = min(total / 5.0, 1.0)

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
        confidence >= 0.7 -> :high
        confidence >= 0.5 -> :moderate
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
    @moduledoc "Represents a scientific investigation testing one or more hypotheses.\n\nAn investigation follows the scientific method:\n1. Formulate hypotheses from questions\n2. Gather evidence from independent sources\n3. Evaluate hypotheses against evidence\n4. Report conclusions (supported, falsified, or inconclusive)\n\n## Experimental Variables\n\nFrom the scientific method, experiments involve three types of variables:\n\n- **Independent Variable**: What we vary (the sources we query)\n- **Dependent Variable**: What we measure (the findings/claims extracted)\n- **Constants**: What we hold fixed (NLP pipeline, corroboration rules)\n\n## Control Treatment\n\nA control treatment provides a baseline for comparison. In our context,\nthis could be:\n- Existing facts in the database (do new findings agree?)\n- Known reliable sources (Wikipedia, encyclopedias)\n\n## Replication\n\nWe require multiple independent sources (replication) to increase\nconfidence. Variation between sources is normal - replication helps\nus see this variation and obtain a consensus.\n\n## Key Principles\n\n- **Falsifiability**: Hypotheses can be disproven by contradicting evidence\n- **Cannot Prove True**: Only support with evidence, never absolute proof\n- **Accumulation**: Knowledge builds through many investigations\n"

    alias Types.{Hypothesis, Finding}

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

    @doc "Creates a new investigation for a topic.\n\n## Options\n  - :hypotheses - Pre-formulated hypotheses\n  - :independent_variable - What we're varying (default: \"source\")\n  - :dependent_variable - What we're measuring (default: \"claim\")\n  - :constants - What we hold fixed (default: NLP pipeline settings)\n"
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

    @doc "Adds a hypothesis to the investigation.\n"
    def add_hypothesis(%__MODULE__{} = investigation, %Hypothesis{} = hypothesis) do
      %{investigation | hypotheses: investigation.hypotheses ++ [hypothesis]}
    end

    @doc "Formulates hypotheses from a list of questions.\n"
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

    @doc "Sets the control treatment - baseline facts to compare against.\n\nControl evidence provides a baseline for comparison:\n- Existing facts in the database\n- Known reliable sources (encyclopedias, etc.)\n"
    def set_control(%__MODULE__{} = investigation, control_findings)
        when is_list(control_findings) do
      %{investigation | control_evidence: control_findings}
    end

    @doc "Records evidence and associates it with relevant hypotheses.\n"
    def record_evidence(%__MODULE__{} = investigation, findings) when is_list(findings) do
      updated_evidence = investigation.evidence ++ findings

      updated_hypotheses =
        investigation.hypotheses
        |> Enum.map(fn hypothesis ->
          associate_evidence(hypothesis, findings)
        end)

      %{
        investigation
        | evidence: updated_evidence,
          hypotheses: updated_hypotheses,
          status: :gathering_evidence
      }
    end

    @doc "Evaluates all hypotheses and concludes the investigation.\n"
    def conclude(%__MODULE__{} = investigation) do
      evaluated =
        investigation.hypotheses
        |> Enum.map(&Hypothesis.evaluate/1)

      conclusion = determine_conclusion(evaluated)

      %{
        investigation
        | hypotheses: evaluated,
          status: :concluded,
          conclusion: conclusion,
          concluded_at: DateTime.utc_now()
      }
    end

    @doc "Returns hypotheses that can be promoted to facts.\n"
    def promotable_hypotheses(%__MODULE__{} = investigation) do
      investigation.hypotheses
      |> Enum.filter(&Hypothesis.promotable?/1)
    end

    @doc "Returns a summary of the investigation results.\n"
    def summary(%__MODULE__{} = investigation) do
      supported = Enum.count(investigation.hypotheses, &(&1.status == :supported))
      falsified = Enum.count(investigation.hypotheses, &(&1.status == :falsified))
      inconclusive = Enum.count(investigation.hypotheses, &(&1.status == :inconclusive))

      unique_sources =
        investigation.evidence
        |> Enum.map(& &1.source.domain)
        |> Enum.uniq()
        |> length()

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

    defp question_to_claim(question) when is_binary(question) do
      question
      |> String.trim_trailing("?")
      |> String.trim()
    end

    defp extract_entity_from_question(question) do
      question
    end

    defp associate_evidence(%Hypothesis{} = hypothesis, findings) do
      Enum.reduce(findings, hypothesis, fn finding, hyp ->
        # Skip findings with empty claims
        if finding.claim == nil or finding.claim == "" do
          hyp
        else
          # Use TF-IDF cosine similarity instead of token overlap
          base_similarity = text_similarity(hyp.claim, finding.claim)

          # Entity-match boost: if finding entity matches hypothesis entity, add 0.15
          entity_boost =
            if hyp.entity && finding.entity &&
                 String.downcase(to_string(hyp.entity)) ==
                   String.downcase(to_string(finding.entity)) do
              0.15
            else
              0.0
            end

          relevance = min(base_similarity + entity_boost, 1.0)

          if relevance >= 0.5 do
            if evidence_contradicts?(hyp.claim, finding.claim) do
              Hypothesis.add_contradicting_evidence(hyp, finding)
            else
              Hypothesis.add_supporting_evidence(hyp, finding)
            end
          else
            hyp
          end
        end
      end)
    end

    defp text_similarity(text_a, text_b) when is_binary(text_a) and is_binary(text_b) do
      alias Brain.Memory.Embedder

      if Embedder.ready?() do
        with {:ok, vec_a} <- Embedder.embed(text_a),
             {:ok, vec_b} <- Embedder.embed(text_b) do
          Embedder.cosine_similarity(vec_a, vec_b)
        else
          _ -> token_overlap_similarity(text_a, text_b)
        end
      else
        token_overlap_similarity(text_a, text_b)
      end
    end

    defp token_overlap_similarity(text_a, text_b) do
      tokens_a = text_a |> Brain.ML.Tokenizer.tokenize_words() |> MapSet.new()
      tokens_b = text_b |> Brain.ML.Tokenizer.tokenize_words() |> MapSet.new()
      overlap = MapSet.intersection(tokens_a, tokens_b) |> MapSet.size()
      union = MapSet.union(tokens_a, tokens_b) |> MapSet.size()

      if union > 0, do: overlap / union, else: 0.0
    end

    defp evidence_contradicts?(claim, finding_claim) do
      LinguisticData.has_negation?(claim) != LinguisticData.has_negation?(finding_claim)
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
    @moduledoc "A research objective for the Learning Center to pursue.\n\n## Scientific Method Integration\n\nResearch goals now support the scientific method by:\n- Generating hypotheses from questions\n- Creating investigations to test those hypotheses\n- Tracking the scientific outcome (supported/falsified)\n"

    alias Types.{Hypothesis, Investigation}

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

    @doc "Creates a new ResearchGoal.\n\n## Options\n  - :questions - List of specific questions to answer\n  - :constraints - Map of constraints (e.g., %{min_sources: 2, max_age_days: 30})\n  - :priority - :low | :normal | :high\n"
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

    @doc "Updates the status of a goal.\n"
    def update_status(%__MODULE__{} = goal, new_status)
        when new_status in [:pending, :in_progress, :completed, :failed] do
      %{goal | status: new_status}
    end

    @doc "Generates hypotheses from the goal's questions.\n\nEach question is transformed into a testable hypothesis.\nIf no questions exist, a hypothesis is generated from the topic.\nReturns empty list for system IDs (containing ':') with no questions.\n"
    def generate_hypotheses(%__MODULE__{} = goal) do
      if goal.questions == [] do
        # System IDs (e.g. "task_training:commonsense") are not real entities
        if system_id?(goal.topic) do
          []
        else
          entity = goal.topic

          [
            Hypothesis.new(
              "#{entity} is a notable entity",
              entity: entity,
              derived_from: "What is #{entity}?",
              prediction:
                "If #{entity} is a notable entity, then sources discussing #{entity} should describe its defining characteristics."
            ),
            Hypothesis.new(
              "#{entity} can be verified by independent sources",
              entity: entity,
              derived_from: "Can #{entity} be independently verified?",
              prediction:
                "If #{entity} can be verified, then multiple independent sources will contain consistent information about #{entity}."
            )
          ]
        end
      else
        goal.questions
        |> Enum.map(fn question ->
          claim = question_to_claim(question)
          entity = extract_entity(question, goal.topic)

          Hypothesis.new(
            claim,
            entity: entity,
            derived_from: question,
            prediction:
              "If #{claim}, then sources discussing #{entity} should confirm this specific aspect."
          )
        end)
      end
    end

    defp system_id?(topic) when is_binary(topic) do
      Brain.ML.Tokenizer.tokenize_words(topic)
      |> Enum.any?(&String.contains?(&1, ":"))
    end

    defp system_id?(_), do: false

    @doc "Creates a scientific investigation from this goal.\n\nThe investigation will:\n1. Formulate hypotheses from questions\n2. Be ready to gather evidence\n3. Track the scientific outcome\n"
    def to_investigation(%__MODULE__{} = goal) do
      hypotheses = generate_hypotheses(goal)

      Investigation.new(goal.topic, hypotheses: hypotheses)
    end

    defp question_to_claim(question) do
      question
      |> String.trim_trailing("?")
      |> String.trim()
      |> remove_question_prefix()
    end

    @question_prefixes MapSet.new(~w(what where who when how is are was were does did))

    defp remove_question_prefix(text) do
      tokens = Brain.ML.Tokenizer.tokenize_normalized(text)

      tokens
      |> Enum.drop_while(fn token -> MapSet.member?(@question_prefixes, token) end)
      |> Enum.join(" ")
      |> String.trim()
    end

    defp extract_entity(question, default_topic) do
      stop_words = MapSet.new(~w(what is are where who when how does did the a an of in to))

      content_words =
        question
        |> Brain.ML.Tokenizer.tokenize_normalized()
        |> Enum.reject(&MapSet.member?(stop_words, &1))

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
    @moduledoc "A finding that has been vetted and is ready for admin review.\n"

    alias Types.{Finding, SourceInfo}

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

    @doc "Creates a new ReviewCandidate from a finding.\n"
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

    @doc "Marks a candidate as approved.\n"
    def approve(%__MODULE__{} = candidate, notes \\ nil) do
      %{candidate | status: :approved, reviewed_at: DateTime.utc_now(), reviewer_notes: notes}
    end

    @doc "Marks a candidate as rejected.\n"
    def reject(%__MODULE__{} = candidate, notes \\ nil) do
      %{candidate | status: :rejected, reviewed_at: DateTime.utc_now(), reviewer_notes: notes}
    end

    @doc "Marks a candidate as deferred for later review.\n"
    def defer(%__MODULE__{} = candidate, notes \\ nil) do
      %{candidate | status: :deferred, reviewed_at: DateTime.utc_now(), reviewer_notes: notes}
    end

    defp generate_id do
      :crypto.strong_rand_bytes(12) |> Base.url_encode64(padding: false)
    end
  end

  defmodule LearningSession do
    @moduledoc "Represents an active or completed learning session.\n\n## Scientific Method Integration\n\nA LearningSession now tracks the full scientific investigation lifecycle:\n- **Investigations**: Scientific investigations with hypotheses\n- **Hypotheses Tested**: Total hypotheses evaluated\n- **Hypotheses Supported**: Hypotheses backed by evidence\n- **Hypotheses Falsified**: Hypotheses disproven by contradicting evidence\n\nThis allows tracking the accumulation of scientific knowledge over time.\n"

    alias Types.{ResearchGoal, Investigation}

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

    @doc "Creates a new LearningSession.\n"
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

    @doc "Adds a goal to the session.\n"
    def add_goal(%__MODULE__{} = session, %ResearchGoal{} = goal) do
      %{session | goals: session.goals ++ [goal]}
    end

    @doc "Adds a completed investigation to the session.\n\nUpdates hypothesis statistics based on the investigation results.\n"
    def add_investigation(%__MODULE__{} = session, %Investigation{} = investigation) do
      supported = Enum.count(investigation.hypotheses, &(&1.status == :supported))
      falsified = Enum.count(investigation.hypotheses, &(&1.status == :falsified))
      tested = length(investigation.hypotheses)

      %{
        session
        | investigations: session.investigations ++ [investigation],
          hypotheses_tested: session.hypotheses_tested + tested,
          hypotheses_supported: session.hypotheses_supported + supported,
          hypotheses_falsified: session.hypotheses_falsified + falsified
      }
    end

    @doc "Increments the findings count.\n"
    def record_findings(%__MODULE__{} = session, count) when is_integer(count) and count >= 0 do
      %{session | findings_count: session.findings_count + count}
    end

    @doc "Records an approval.\n"
    def record_approval(%__MODULE__{} = session) do
      %{session | approved_count: session.approved_count + 1}
    end

    @doc "Records a rejection.\n"
    def record_rejection(%__MODULE__{} = session) do
      %{session | rejected_count: session.rejected_count + 1}
    end

    @doc "Marks the session as completed.\n"
    def complete(%__MODULE__{} = session) do
      %{session | status: :completed, completed_at: DateTime.utc_now()}
    end

    @doc "Marks the session as cancelled.\n"
    def cancel(%__MODULE__{} = session) do
      %{session | status: :cancelled, completed_at: DateTime.utc_now()}
    end

    @doc "Returns a summary of the session's scientific outcomes.\n"
    def scientific_summary(%__MODULE__{} = session) do
      %{
        topic: session.topic,
        investigations_completed: length(session.investigations),
        hypotheses_tested: session.hypotheses_tested,
        hypotheses_supported: session.hypotheses_supported,
        hypotheses_falsified: session.hypotheses_falsified,
        support_rate:
          if(session.hypotheses_tested > 0) do
            session.hypotheses_supported / session.hypotheses_tested
          else
            0.0
          end,
        facts_approved: session.approved_count,
        facts_rejected: session.rejected_count
      }
    end

    defp generate_id do
      :crypto.strong_rand_bytes(12) |> Base.url_encode64(padding: false)
    end
  end

  defmodule SourceProfile do
    @moduledoc "Extended profile for a source domain including historical data.\nUsed internally by SourceReliability GenServer.\n"

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

    @doc "Creates a new SourceProfile.\n"
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

    @doc "Records an admin decision (approval or rejection) for this source.\n"
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

    @doc "Calculates the current reliability score based on base accuracy and admin feedback.\n"
    def calculate_reliability(%__MODULE__{} = profile) do
      base = profile.factual_accuracy
      feedback_adjustment = calculate_feedback_adjustment(profile.admin_decisions)

      (base * 0.7 + feedback_adjustment * 0.3)
      |> max(0.0)
      |> min(1.0)
    end

    defp calculate_feedback_adjustment(decisions) when is_list(decisions) do
      if decisions == [] do
        0.5
      else
        {weighted_sum, total_weight} =
          decisions
          |> Enum.with_index()
          |> Enum.reduce({0.0, 0.0}, fn {decision, idx}, {sum, weight} ->
            decay = :math.pow(0.9, idx)

            value =
              if decision.decision == :approved do
                1.0
              else
                0.0
              end

            {sum + value * decay, weight + decay}
          end)

        if total_weight > 0 do
          weighted_sum / total_weight
        else
          0.5
        end
      end
    end
  end
end
