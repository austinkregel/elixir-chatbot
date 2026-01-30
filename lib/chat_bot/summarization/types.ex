defmodule ChatBot.Summarization.Types do
  @moduledoc """
  Type definitions for the dialogue summarization pipeline.

  Each struct represents an intermediate representation in the pipeline:
  - Turn: A single speaker's utterance in dialogue
  - Event: A semantic event extracted from a turn
  - Fact: A ranked, resolved fact ready for summary
  - Summary: The final generated summary
  """

  # ============================================================================
  # Turn - Output of Turn Segmenter
  # ============================================================================

  defmodule Turn do
    @moduledoc """
    A single turn in a dialogue - one speaker's utterance.
    """
    @type t :: %__MODULE__{
      speaker: String.t(),
      text: String.t(),
      index: non_neg_integer(),
      tokens: [String.t()] | nil
    }

    @enforce_keys [:speaker, :text, :index]
    defstruct [
      :speaker,
      :text,
      :index,
      tokens: nil
    ]

    def new(speaker, text, index) do
      %__MODULE__{
        speaker: speaker,
        text: text,
        index: index
      }
    end
  end

  # ============================================================================
  # Event - Output of Event Extractor
  # ============================================================================

  defmodule Event do
    @moduledoc """
    A semantic event extracted from a dialogue turn.

    Based on speech act theory:
    - :assertive - Statements of fact
    - :commissive - Promises, plans, commitments
    - :expressive - Emotions, attitudes
    - :directive - Questions, requests, commands
    """
    @type event_type :: :assertive | :commissive | :expressive | :directive | :phatic
    @type event_subtype ::
      :announcement | :statement | :plan | :promise | :greeting |
      :farewell | :question | :request | :emotion | :state | :other

    @type t :: %__MODULE__{
      turn_index: non_neg_integer(),
      speaker: String.t(),
      type: event_type(),
      subtype: event_subtype(),
      subject: String.t() | nil,
      predicate: String.t(),
      object: String.t() | nil,
      tense: :past | :present | :future | nil,
      negated: boolean(),
      raw_text: String.t()
    }

    @enforce_keys [:turn_index, :speaker, :type, :predicate, :raw_text]
    defstruct [
      :turn_index,
      :speaker,
      :type,
      :predicate,
      :raw_text,
      subtype: :other,
      subject: nil,
      object: nil,
      tense: nil,
      negated: false
    ]

    def new(turn_index, speaker, type, predicate, raw_text, opts \\ []) do
      %__MODULE__{
        turn_index: turn_index,
        speaker: speaker,
        type: type,
        predicate: predicate,
        raw_text: raw_text,
        subtype: Keyword.get(opts, :subtype, :other),
        subject: Keyword.get(opts, :subject),
        object: Keyword.get(opts, :object),
        tense: Keyword.get(opts, :tense),
        negated: Keyword.get(opts, :negated, false)
      }
    end
  end

  # ============================================================================
  # Fact - Output of Fact Ranker (after reference resolution)
  # ============================================================================

  defmodule Fact do
    @moduledoc """
    A resolved and ranked fact ready for summary generation.

    All pronouns have been resolved to concrete referents.
    Importance score determines inclusion in final summary.
    """
    @type importance :: :high | :medium | :low | :skip

    @type t :: %__MODULE__{
      subject: String.t(),
      predicate: String.t(),
      object: String.t() | nil,
      participants: [String.t()],
      tense: :past | :present | :future | nil,
      importance: importance(),
      score: float(),
      event_type: atom(),
      template_key: atom() | nil
    }

    @enforce_keys [:subject, :predicate, :importance]
    defstruct [
      :subject,
      :predicate,
      :importance,
      object: nil,
      participants: [],
      tense: nil,
      score: 0.0,
      event_type: :statement,
      template_key: nil
    ]

    def new(subject, predicate, importance, opts \\ []) do
      %__MODULE__{
        subject: subject,
        predicate: predicate,
        importance: importance,
        object: Keyword.get(opts, :object),
        participants: Keyword.get(opts, :participants, []),
        tense: Keyword.get(opts, :tense),
        score: Keyword.get(opts, :score, 0.0),
        event_type: Keyword.get(opts, :event_type, :statement),
        template_key: Keyword.get(opts, :template_key)
      }
    end
  end

  # ============================================================================
  # Summary - Final Output
  # ============================================================================

  defmodule Summary do
    @moduledoc """
    The final generated summary with metadata.
    """
    @type t :: %__MODULE__{
      text: String.t(),
      sentences: [String.t()],
      participants: [String.t()],
      fact_count: non_neg_integer(),
      facts_used: [Fact.t()]
    }

    @enforce_keys [:text]
    defstruct [
      :text,
      sentences: [],
      participants: [],
      fact_count: 0,
      facts_used: []
    ]

    def new(text, opts \\ []) do
      %__MODULE__{
        text: text,
        sentences: Keyword.get(opts, :sentences, []),
        participants: Keyword.get(opts, :participants, []),
        fact_count: Keyword.get(opts, :fact_count, 0),
        facts_used: Keyword.get(opts, :facts_used, [])
      }
    end
  end

  # ============================================================================
  # Speaker Context - Used by Speaker Tracker
  # ============================================================================

  defmodule SpeakerContext do
    @moduledoc """
    Tracks speaker information and pronoun mappings across dialogue.
    """
    @type t :: %__MODULE__{
      participants: [String.t()],
      current_speaker: String.t() | nil,
      addressee: String.t() | nil,
      pronoun_map: %{String.t() => String.t()},
      mentioned_entities: [%{text: String.t(), type: atom(), turn: non_neg_integer()}]
    }

    defstruct [
      participants: [],
      current_speaker: nil,
      addressee: nil,
      pronoun_map: %{},
      mentioned_entities: []
    ]

    def new(participants \\ []) do
      %__MODULE__{
        participants: participants,
        pronoun_map: %{}
      }
    end

    @doc """
    Update context for a new turn.
    """
    def update_for_turn(%__MODULE__{} = ctx, speaker, addressee \\ nil) do
      # When speaker says "I", it refers to themselves
      # When speaker says "you", it refers to addressee
      pronoun_map = %{
        "i" => speaker,
        "me" => speaker,
        "my" => speaker,
        "myself" => speaker,
        "we" => speaker  # Could be refined if multiple speakers
      }

      pronoun_map = if addressee do
        Map.merge(pronoun_map, %{
          "you" => addressee,
          "your" => addressee,
          "yours" => addressee
        })
      else
        pronoun_map
      end

      %{ctx |
        current_speaker: speaker,
        addressee: addressee,
        pronoun_map: pronoun_map
      }
    end

    @doc """
    Add a mentioned entity for later anaphora resolution.
    """
    def add_entity(%__MODULE__{} = ctx, text, type, turn_index) do
      entity = %{text: text, type: type, turn: turn_index}
      %{ctx | mentioned_entities: [entity | ctx.mentioned_entities]}
    end
  end
end
