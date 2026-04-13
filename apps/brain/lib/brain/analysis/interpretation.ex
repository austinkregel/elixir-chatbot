defmodule Brain.Analysis.Interpretation do
  @moduledoc "Represents an interpretation of user input with activation levels.\n\nUnlike binary classification results, interpretations carry:\n- Activation levels (0.0-1.0) representing confidence\n- Source tracking (which analyzer produced this)\n- Calibrated scores based on historical accuracy\n- Alternative interpretations that lost the race but stay \"warm\"\n\nThis enables competitive activation where multiple interpretations\nrace and the first to threshold wins, while runners-up remain\navailable for backtracking.\n"

  alias Brain.Analysis.{AnalyzerResult, SlotResult}

  @type source ::
          :memory_match | :pattern_recognition | :structural | :keyword | :model | :heuristic
  @type scope :: :global | :cohort | :user

  @type alternative :: %{
          intent: String.t(),
          activation: float(),
          source: source(),
          raw_score: float()
        }

  @type t :: %__MODULE__{
          intent: String.t() | nil,
          text: String.t(),
          raw_activation: float(),
          activation: float(),
          calibrated_activation: float(),
          source: source(),
          triggering_heuristic_id: String.t() | nil,
          heuristic_scope: scope() | nil,
          analyzer_results: list(AnalyzerResult.t()),
          slots: SlotResult.t() | nil,
          entities: list(map()),
          alternatives: list(alternative()),
          created_at: integer(),
          backtrack_count: integer(),
          was_promoted: boolean(),
          metadata: map()
        }

  defstruct [
    :intent,
    :text,
    raw_activation: 0.0,
    activation: 0.0,
    calibrated_activation: 0.0,
    source: :structural,
    triggering_heuristic_id: nil,
    heuristic_scope: nil,
    analyzer_results: [],
    slots: nil,
    entities: [],
    alternatives: [],
    created_at: nil,
    backtrack_count: 0,
    was_promoted: false,
    metadata: %{}
  ]

  @doc "Creates a new interpretation with the given intent and activation.\n"
  def new(intent, text, activation, source) do
    %__MODULE__{
      intent: intent,
      text: text,
      raw_activation: activation,
      activation: activation,
      calibrated_activation: activation,
      source: source,
      created_at: System.monotonic_time(:millisecond)
    }
  end

  @doc "Creates an interpretation from analyzer results, picking the winner.\n"
  def from_analyzer_results(text, results) when is_list(results) do
    sorted = Enum.sort_by(results, & &1.calibrated_activation, :desc)

    case sorted do
      [] ->
        new("unknown", text, 0.0, :structural)

      [winner | runners_up] ->
        alternatives =
          runners_up
          |> Enum.take(5)
          |> Enum.map(fn r ->
            %{
              intent: r.intent,
              activation: r.calibrated_activation,
              source: r.analyzer,
              raw_score: r.raw_score
            }
          end)

        %__MODULE__{
          intent: winner.intent,
          text: text,
          raw_activation: winner.raw_score,
          activation: winner.calibrated_activation,
          calibrated_activation: winner.calibrated_activation,
          source: winner.analyzer,
          analyzer_results: sorted,
          alternatives: alternatives,
          created_at: System.monotonic_time(:millisecond)
        }
    end
  end

  @doc "Updates the interpretation with slot detection results.\n"
  def with_slots(%__MODULE__{} = interp, %SlotResult{} = slots) do
    %{interp | slots: slots}
  end

  @doc "Updates the interpretation with extracted entities.\n"
  def with_entities(%__MODULE__{} = interp, entities) when is_list(entities) do
    %{interp | entities: entities}
  end

  @doc "Sets the normalized activation after global inhibition.\n"
  def with_normalized_activation(%__MODULE__{} = interp, normalized) do
    %{interp | activation: normalized}
  end

  @doc "Marks this interpretation as having come from a heuristic fast-path.\n"
  def with_heuristic(%__MODULE__{} = interp, heuristic_id, scope) do
    %{interp | triggering_heuristic_id: heuristic_id, heuristic_scope: scope}
  end

  @doc "Promotes the highest-ranked alternative to primary, demoting current.\n\nReturns {:ok, new_interpretation} or {:error, :no_alternatives}\n"
  def promote_alternative(%__MODULE__{alternatives: []} = _interp) do
    {:error, :no_alternatives}
  end

  def promote_alternative(%__MODULE__{alternatives: [next | rest]} = interp) do
    demoted = %{
      intent: interp.intent,
      activation: interp.activation * 0.5,
      source: interp.source,
      raw_score: interp.raw_activation
    }

    promoted = %{
      interp
      | intent: next.intent,
        raw_activation: next.raw_score,
        activation: next.activation,
        calibrated_activation: next.activation,
        source: next.source,
        alternatives: rest ++ [demoted],
        backtrack_count: interp.backtrack_count + 1,
        was_promoted: true
    }

    {:ok, promoted}
  end

  @doc "Checks if this interpretation has required slots missing.\n"
  def has_missing_required?(%__MODULE__{slots: nil}) do
    false
  end

  def has_missing_required?(%__MODULE__{slots: slots}) do
    not slots.all_required_filled
  end

  @doc "Returns the list of missing required slots.\n"
  def missing_required(%__MODULE__{slots: nil}) do
    []
  end

  def missing_required(%__MODULE__{slots: slots}) do
    slots.missing_required
  end

  @doc "Checks if this interpretation was triggered by a heuristic.\n"
  def from_heuristic?(%__MODULE__{triggering_heuristic_id: nil}) do
    false
  end

  def from_heuristic?(%__MODULE__{}) do
    true
  end

  @doc "Returns the confidence level as a category.\n"
  def confidence_level(%__MODULE__{activation: a}) when a >= 0.85 do
    :high
  end

  def confidence_level(%__MODULE__{activation: a}) when a >= 0.6 do
    :medium
  end

  def confidence_level(%__MODULE__{activation: a}) when a >= 0.3 do
    :low
  end

  def confidence_level(%__MODULE__{}) do
    :very_low
  end

  @doc "Calculates how much \"headroom\" remains for activation boosts.\n"
  def activation_headroom(%__MODULE__{activation: a}) do
    max(0.0, 1.0 - a)
  end
end

defmodule Brain.Analysis.AnalyzerResult do
  @moduledoc "Result from a single analyzer in the racing system.\n\nEach analyzer outputs a rich result with both raw and calibrated scores,\nallowing the system to compare apples-to-apples across different analyzers.\n"

  @type analyzer_type ::
          :memory_similarity
          | :pattern_recognition
          | :structural
          | :keyword
          | :model
          | :early_confidence

  @type t :: %__MODULE__{
          analyzer: analyzer_type(),
          intent: String.t() | nil,
          raw_score: float(),
          confidence_estimate: float(),
          calibrated_activation: float(),
          historical_calibration_error: float(),
          indicators: list(String.t()),
          metadata: map()
        }

  defstruct [
    :analyzer,
    :intent,
    raw_score: 0.0,
    confidence_estimate: 0.0,
    calibrated_activation: 0.0,
    historical_calibration_error: 0.0,
    indicators: [],
    metadata: %{}
  ]

  @doc "Creates a new analyzer result.\n"
  def new(analyzer, intent, raw_score, opts \\ []) do
    %__MODULE__{
      analyzer: analyzer,
      intent: intent,
      raw_score: raw_score,
      confidence_estimate: Keyword.get(opts, :confidence_estimate, raw_score),
      calibrated_activation: Keyword.get(opts, :calibrated_activation, raw_score),
      historical_calibration_error: Keyword.get(opts, :calibration_error, 0.0),
      indicators: Keyword.get(opts, :indicators, []),
      metadata: Keyword.get(opts, :metadata, %{})
    }
  end

  @doc "Updates the result with calibrated activation.\n"
  def with_calibration(%__MODULE__{} = result, calibrated, error) do
    %{result | calibrated_activation: calibrated, historical_calibration_error: error}
  end
end