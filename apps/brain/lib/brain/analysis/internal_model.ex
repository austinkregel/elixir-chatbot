defmodule Brain.Analysis.InternalModel do
  @moduledoc "Core data structures for the text analysis pipeline.\n\nThe InternalModel represents the complete analysis of user input,\ncombining results from all analysis stages into a unified model\nthat downstream components can use to generate contextual responses.\n"

  alias Brain.Analysis.{Chunk, ChunkAnalysis, DiscourseResult, SpeechActResult, SlotResult}

  @type response_strategy ::
          :can_respond
          | :hedged_response
          | :needs_clarification
          | :cannot_respond
          | :defer_to_user
          | :partial_response_with_clarification

  @type t :: %__MODULE__{
          raw_input: String.t(),
          chunks: list(Chunk.t()),
          analyses: list(ChunkAnalysis.t()),
          overall_strategy: response_strategy(),
          suggested_prompts: list(String.t()),
          metadata: map(),
          created_at: integer()
        }

  defstruct [
    :raw_input,
    chunks: [],
    analyses: [],
    overall_strategy: :can_respond,
    suggested_prompts: [],
    metadata: %{},
    created_at: nil
  ]

  @doc "Creates a new InternalModel from raw input.\n"
  def new(raw_input) when is_binary(raw_input) do
    %__MODULE__{
      raw_input: raw_input,
      created_at: System.system_time(:millisecond)
    }
  end

  @doc "Adds chunks to the model.\n"
  def with_chunks(%__MODULE__{} = model, chunks) when is_list(chunks) do
    %{model | chunks: chunks}
  end

  @doc "Adds chunk analyses to the model.\n"
  def with_analyses(%__MODULE__{} = model, analyses) when is_list(analyses) do
    %{model | analyses: analyses}
  end

  @doc "Determines the overall response strategy based on chunk analyses.\n"
  def determine_strategy(%__MODULE__{analyses: analyses} = model) do
    strategies = Enum.map(analyses, & &1.response_strategy)

    overall =
      cond do
        Enum.all?(strategies, &(&1 == :can_respond)) ->
          :can_respond

        Enum.all?(strategies, &(&1 in [:can_respond, :hedged_response])) and
            Enum.any?(strategies, &(&1 == :hedged_response)) ->
          :hedged_response

        Enum.all?(strategies, &(&1 == :cannot_respond)) ->
          :cannot_respond

        Enum.all?(strategies, &(&1 == :defer_to_user)) ->
          :defer_to_user

        Enum.any?(strategies, &(&1 == :needs_clarification)) ->
          if Enum.any?(strategies, &(&1 in [:can_respond, :hedged_response])) do
            :partial_response_with_clarification
          else
            :needs_clarification
          end

        true ->
          :can_respond
      end

    prompts =
      analyses
      |> Enum.filter(&(&1.response_strategy == :needs_clarification))
      |> Enum.flat_map(& &1.clarification_prompts)
      |> Enum.uniq()

    %{model | overall_strategy: overall, suggested_prompts: prompts}
  end

  @doc "Returns analyses that can be responded to (including hedged responses).\n"
  def respondable_analyses(%__MODULE__{analyses: analyses}) do
    Enum.filter(analyses, &(&1.response_strategy in [:can_respond, :hedged_response]))
  end

  @doc "Returns analyses that need clarification.\n"
  def clarification_analyses(%__MODULE__{analyses: analyses}) do
    Enum.filter(analyses, &(&1.response_strategy == :needs_clarification))
  end

  @doc "Checks if the bot is being addressed in any chunk.\n"
  def bot_addressed?(%__MODULE__{analyses: analyses}) do
    Enum.any?(analyses, fn analysis ->
      analysis.discourse.addressee == :bot
    end)
  end
end

defmodule Brain.Analysis.Chunk do
  @moduledoc "Represents a single semantic chunk (utterance) extracted from user input.\n"

  @type t :: %__MODULE__{
          text: String.t(),
          index: non_neg_integer(),
          start_pos: non_neg_integer(),
          end_pos: non_neg_integer(),
          is_quoted: boolean(),
          discourse_markers: list(String.t())
        }

  defstruct [:text, :index, :start_pos, :end_pos, is_quoted: false, discourse_markers: []]

  @doc "Creates a new chunk.\n"
  def new(text, index, start_pos, end_pos, opts \\ []) do
    %__MODULE__{
      text: text,
      index: index,
      start_pos: start_pos,
      end_pos: end_pos,
      is_quoted: Keyword.get(opts, :is_quoted, false),
      discourse_markers: Keyword.get(opts, :discourse_markers, [])
    }
  end
end

defmodule Brain.Analysis.ChunkAnalysis do
  @moduledoc "Complete analysis for a single chunk, combining results from all analyzers.\n"

  alias Brain.Analysis.{DiscourseResult, SpeechActResult, SlotResult, SlotDetector, ChunkProfile}
  alias Brain.Analysis.Types.Event
  alias Brain.Analysis.InternalModel

  @type sentiment :: %{label: atom(), confidence: float()} | nil

  @type fact_verification ::
          {:verified, float()}
          | {:contradicted, list(map())}
          | {:uncertain, atom()}
          | nil

  @type epistemic_status :: :unchecked | :verified | :contradicted | :uncertain

  @type pass :: 1 | 2

  @type t :: %__MODULE__{
          chunk_index: non_neg_integer(),
          text: String.t(),
          discourse: DiscourseResult.t(),
          speech_act: SpeechActResult.t(),
          intent: String.t() | nil,
          profile: ChunkProfile.t() | nil,
          feature_vector: list(float()),
          entities: list(map()),
          slots: SlotResult.t(),
          missing_context: list(atom()),
          response_strategy: InternalModel.response_strategy(),
          clarification_prompts: list(String.t()),
          confidence: float(),
          events: list(Event.t()),
          sentiment: sentiment(),
          fact_verification: fact_verification(),
          related_beliefs: list(map()),
          epistemic_status: epistemic_status(),
          pass: pass()
        }

  defstruct [
    :chunk_index,
    :text,
    :discourse,
    :speech_act,
    intent: nil,
    profile: nil,
    feature_vector: [],
    entities: [],
    slots: nil,
    missing_context: [],
    response_strategy: :can_respond,
    clarification_prompts: [],
    confidence: 0.0,
    events: [],
    sentiment: nil,
    fact_verification: nil,
    related_beliefs: [],
    epistemic_status: :unchecked,
    event_frames: [],
    srl_frames: [],
    pos_tags: [],
    accumulated_context: nil,
    pass: 1
  ]

  @doc "Creates a new chunk analysis.\n"
  def new(chunk_index, text) do
    %__MODULE__{
      chunk_index: chunk_index,
      text: text
    }
  end

  @doc "Adds extracted events to the chunk analysis.\n\n## Examples\n\n    analysis = ChunkAnalysis.new(0, \"I want coffee\")\n    events = [%Event{action: %{verb: \"want\", ...}, ...}]\n    analysis = ChunkAnalysis.with_events(analysis, events)\n"
  def with_events(%__MODULE__{} = analysis, events) when is_list(events) do
    %{analysis | events: events}
  end

  @doc "Returns the primary event from the analysis (highest confidence).\n"
  def primary_event(%__MODULE__{events: []}) do
    nil
  end

  def primary_event(%__MODULE__{events: events}) do
    Enum.max_by(events, & &1.confidence, fn -> nil end)
  end

  @doc "Checks if the chunk has any extracted events.\n"
  def has_events?(%__MODULE__{events: events}) do
    events != []
  end

  @doc "Determines response strategy based on analysis results.\n"
  def determine_response_strategy(%__MODULE__{} = analysis) do
    cond do
      analysis.discourse.addressee in [:user, :third_party] ->
        %{analysis | response_strategy: :defer_to_user}

      analysis.discourse.addressee not in [:bot, :ambiguous, :unknown] ->
        %{analysis | response_strategy: :defer_to_user}

      analysis.missing_context != [] ->
        prompts = generate_clarification_prompts(analysis.missing_context, analysis.intent)

        %{
          analysis
          | response_strategy: :needs_clarification,
            clarification_prompts: prompts
        }

      uncertain_intent?(analysis) ->
        %{analysis | response_strategy: :hedged_response}

      true ->
        %{analysis | response_strategy: :can_respond}
    end
  end

  defp uncertain_intent?(analysis) do
    intent_conf = get_intent_confidence(analysis)
    acc = analysis.accumulated_context

    low_intent = is_number(intent_conf) and intent_conf < 0.4

    should_hedge =
      if acc != nil and is_struct(acc, Brain.Analysis.ContextAccumulator) do
        Brain.Analysis.ContextAccumulator.should_hedge?(acc)
      else
        false
      end

    low_intent or should_hedge
  end

  defp get_intent_confidence(%{speech_act: %{intent_confidence: c}}) when is_number(c), do: c
  defp get_intent_confidence(_), do: nil

  defp generate_clarification_prompts(missing_context, intent) do
    SlotDetector.get_clarification_prompts(missing_context, intent)
  end
end

defmodule Brain.Analysis.DiscourseResult do
  @moduledoc "Result of discourse analysis - who is being addressed.\n"

  @type addressee :: :bot | :user | :third_party | :ambiguous | :unknown

  @type t :: %__MODULE__{
          addressee: addressee(),
          confidence: float(),
          indicators: list(String.t()),
          participants: list(atom()),
          direct_address_detected: boolean()
        }

  defstruct addressee: :unknown,
            confidence: 0.0,
            indicators: [],
            participants: [:user, :bot],
            direct_address_detected: false

  @doc "Creates a new discourse result.\n"
  def new(addressee, confidence, indicators \\ []) do
    %__MODULE__{
      addressee: addressee,
      confidence: confidence,
      indicators: indicators,
      direct_address_detected: "direct_address" in indicators
    }
  end
end

defmodule Brain.Analysis.SpeechActResult do
  @moduledoc "Result of speech act classification.\n\nBased on Searle's taxonomy:\n- Assertives: statements, claims, reports\n- Directives: requests, commands, questions\n- Commissives: promises, offers\n- Expressives: thanks, apologies, greetings\n- Declaratives: performatives\n"

  @type category :: :assertive | :directive | :commissive | :expressive | :declarative | :unknown

  @type sub_type ::
          :statement
          | :claim
          | :report
          | :question_factual
          | :question_opinion
          | :question_rhetorical
          | :request_action
          | :request_information
          | :command
          | :promise
          | :offer
          | :thanks
          | :apology
          | :greeting
          | :farewell
          | :performative
          | :backchannel
          | :compliment
          | :acknowledgment
          | :continuation
          | :unknown

  @type t :: %__MODULE__{
          category: category(),
          sub_type: sub_type(),
          confidence: float(),
          indicators: list(String.t()),
          is_question: boolean(),
          is_imperative: boolean(),
          intent_confidence: float() | nil,
          raw_analyses: map() | nil
        }

  defstruct category: :unknown,
            sub_type: :unknown,
            confidence: 0.0,
            indicators: [],
            is_question: false,
            is_imperative: false,
            intent_confidence: nil,
            raw_analyses: nil

  @doc "Creates a new speech act result.\n\nThe `:raw_analyses` option carries the per-voter analysis maps so that\n`Brain.Analysis.SpeechActClassifier.refine_with_intent/3` can splice in a\nfreshly-classified intent voter without re-running the cheap voters.\n"
  def new(category, sub_type, confidence, opts \\ []) do
    %__MODULE__{
      category: category,
      sub_type: sub_type,
      confidence: confidence,
      indicators: Keyword.get(opts, :indicators, []),
      is_question: Keyword.get(opts, :is_question, false),
      is_imperative: Keyword.get(opts, :is_imperative, false),
      intent_confidence: Keyword.get(opts, :intent_confidence),
      raw_analyses: Keyword.get(opts, :raw_analyses)
    }
  end

  @doc "Checks if this speech act expects a response from the addressee.\n\nReturns:\n- `true` for speech acts that clearly expect a response (directives, questions, greetings)\n- `false` for speech acts where silence is appropriate (backchannels, continuations)\n- `:optional` for speech acts where response is situational (compliments, acknowledgments)\n"
  def expects_response?(%__MODULE__{category: :directive}) do
    true
  end

  def expects_response?(%__MODULE__{sub_type: :greeting}) do
    true
  end

  def expects_response?(%__MODULE__{sub_type: :question_factual}) do
    true
  end

  def expects_response?(%__MODULE__{sub_type: :question_opinion}) do
    true
  end

  def expects_response?(%__MODULE__{sub_type: :backchannel}) do
    false
  end

  def expects_response?(%__MODULE__{sub_type: :continuation}) do
    false
  end

  def expects_response?(%__MODULE__{sub_type: :compliment}) do
    :optional
  end

  def expects_response?(%__MODULE__{sub_type: :acknowledgment}) do
    :optional
  end

  def expects_response?(_) do
    :unknown
  end
end

defmodule Brain.Analysis.SlotResult do
  @moduledoc "Result of slot detection and context resolution.\n"

  @type slot_value :: %{
          value: any(),
          source: :explicit | :conversation | :user_profile | :default | :inferred,
          confidence: float()
        }

  @type t :: %__MODULE__{
          schema_name: String.t() | nil,
          filled_slots: %{String.t() => slot_value()},
          missing_required: list(String.t()),
          missing_optional: list(String.t()),
          all_required_filled: boolean()
        }

  defstruct schema_name: nil,
            filled_slots: %{},
            missing_required: [],
            missing_optional: [],
            all_required_filled: true

  @doc "Creates a new slot result.\n"
  def new(schema_name \\ nil) do
    %__MODULE__{schema_name: schema_name}
  end

  @doc "Adds a filled slot.\n"
  def fill_slot(%__MODULE__{} = result, slot_name, value, source, confidence \\ 1.0) do
    slot_value = %{value: value, source: source, confidence: confidence}

    updated_filled = Map.put(result.filled_slots, slot_name, slot_value)
    updated_missing_req = List.delete(result.missing_required, slot_name)
    updated_missing_opt = List.delete(result.missing_optional, slot_name)

    %{
      result
      | filled_slots: updated_filled,
        missing_required: updated_missing_req,
        missing_optional: updated_missing_opt,
        all_required_filled: updated_missing_req == []
    }
  end

  @doc "Gets the value of a slot if filled.\n"
  def get_slot_value(%__MODULE__{filled_slots: slots}, slot_name) do
    case Map.get(slots, slot_name) do
      nil -> nil
      %{value: value} -> value
    end
  end
end
