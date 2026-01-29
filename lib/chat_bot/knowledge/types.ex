defmodule ChatBot.Knowledge.Types do
  @moduledoc """
  Type definitions for the Knowledge Expansion System.

  Provides structs for representing research goals, findings, source information,
  review candidates, and learning sessions.
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

    alias ChatBot.Knowledge.Types.SourceInfo

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

  defmodule ResearchGoal do
    @moduledoc """
    A research objective for the Learning Center to pursue.
    """

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

    defp generate_id do
      :crypto.strong_rand_bytes(12) |> Base.url_encode64(padding: false)
    end
  end

  defmodule ReviewCandidate do
    @moduledoc """
    A finding that has been vetted and is ready for admin review.
    """

    alias ChatBot.Knowledge.Types.{Finding, SourceInfo}

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
    """

    alias ChatBot.Knowledge.Types.ResearchGoal

    @type status :: :active | :completed | :cancelled

    @type t :: %__MODULE__{
            id: String.t(),
            goals: [ResearchGoal.t()],
            started_at: DateTime.t(),
            completed_at: DateTime.t() | nil,
            findings_count: non_neg_integer(),
            approved_count: non_neg_integer(),
            rejected_count: non_neg_integer(),
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
      findings_count: 0,
      approved_count: 0,
      rejected_count: 0,
      status: :active
    ]

    @doc """
    Creates a new LearningSession.
    """
    def new(opts \\ []) do
      %__MODULE__{
        id: generate_id(),
        goals: Keyword.get(opts, :goals, []),
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
