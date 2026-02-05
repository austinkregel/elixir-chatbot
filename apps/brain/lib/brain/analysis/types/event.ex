defmodule Brain.Analysis.Types.Event do
  @moduledoc """
  Represents a structured event extracted from user input.

  Events capture the semantic structure of actions described in text:
  - Who did it (actor)
  - What they did (action/verb)
  - To what/whom (object)
  - How/when/where (modifiers)

  Events are extracted from POS tags and entities using tensor-based
  pattern matching - no string matching or regex.

  ## Examples

      # "I want coffee" -> Event with actor=I, action=want, object=coffee
      %Event{
        id: "evt_abc123",
        action: %{verb: "want", lemma: "want", tense: :present},
        actor: %{text: "I", type: "pronoun", token_index: 0},
        object: %{text: "coffee", type: "noun", token_index: 2},
        modifiers: [],
        confidence: 0.92,
        source_tokens: [0, 1, 2]
      }

      # "Play me some jazz music" -> Event with imperative action
      %Event{
        id: "evt_def456",
        action: %{verb: "Play", lemma: "play", tense: :imperative},
        actor: nil,  # Implied "you" (the bot)
        object: %{text: "jazz music", type: "noun_phrase", token_index: 3},
        modifiers: [%{type: :benefactive, text: "me", token_index: 1}],
        confidence: 0.88,
        source_tokens: [0, 1, 2, 3, 4]
      }
  """

  @type action :: %{
          verb: String.t(),
          lemma: String.t(),
          tense: :present | :past | :future | :imperative | :infinitive | :unknown
        }

  @type participant :: %{
          text: String.t(),
          type: String.t(),
          token_index: non_neg_integer(),
          entity_type: String.t() | nil
        }

  @type modifier :: %{
          type: :temporal | :spatial | :manner | :benefactive | :instrumental | :purpose | :other,
          text: String.t(),
          token_index: non_neg_integer()
        }

  @type t :: %__MODULE__{
          id: String.t(),
          action: action(),
          actor: participant() | nil,
          object: participant() | nil,
          modifiers: [modifier()],
          confidence: float(),
          source_tokens: [non_neg_integer()]
        }

  defstruct [
    :id,
    :action,
    :actor,
    :object,
    modifiers: [],
    confidence: 0.0,
    source_tokens: []
  ]

  @doc """
  Creates a new Event with a generated ID.
  """
  def new(action, opts \\ []) do
    %__MODULE__{
      id: generate_id(),
      action: action,
      actor: Keyword.get(opts, :actor),
      object: Keyword.get(opts, :object),
      modifiers: Keyword.get(opts, :modifiers, []),
      confidence: Keyword.get(opts, :confidence, 0.0),
      source_tokens: Keyword.get(opts, :source_tokens, [])
    }
  end

  @doc """
  Checks if the event has both an actor and object (complete structure).
  """
  def complete?(%__MODULE__{actor: actor, object: object}) do
    actor != nil and object != nil
  end

  @doc """
  Checks if the event is an imperative (command).
  """
  def imperative?(%__MODULE__{action: %{tense: :imperative}}), do: true
  def imperative?(_), do: false

  @doc """
  Returns the primary action lemma for this event.
  """
  def action_lemma(%__MODULE__{action: %{lemma: lemma}}), do: lemma
  def action_lemma(_), do: nil

  @doc """
  Converts the event to a human-readable description.
  """
  def to_description(%__MODULE__{} = event) do
    actor_text = if event.actor, do: event.actor.text, else: "(someone)"
    object_text = if event.object, do: event.object.text, else: "(something)"
    verb = event.action.lemma

    "#{actor_text} #{verb} #{object_text}"
  end

  # Generate a unique event ID
  defp generate_id do
    "evt_" <> Base.encode16(:crypto.strong_rand_bytes(8), case: :lower)
  end
end
