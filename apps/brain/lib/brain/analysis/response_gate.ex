defmodule Brain.Analysis.ResponseGate do
  @moduledoc """
  Evaluates whether a response is appropriate based on speech act sequences.

  This module reasons over *classified speech acts*, not raw text. All detection
  happens through the existing analysis passes (structural, pragmatic, keyword, etc.)
  in SpeechActClassifier. ResponseGate simply examines the resulting classifications
  and conversation history to determine if a response is needed.

  ## Response Optionality Patterns

  - **Gratitude loops**: User thanks → Bot acknowledges → User thanks again → defer
  - **Backchannels**: Classified as `:backchannel` by pragmatic pass → optional
  - **Compliments**: Classified as `:compliment` without question → optional
  - **Continuations**: Classified as `:continuation` by structural pass → defer
  - **Rapid-fire input**: Multiple messages within a tight window → defer (user still typing)
  - **Echo/repetition**: Same speech act category+sub_type as previous → optional
  - **Self-correction**: Correction/amendment sub_type or same-category rapid restatement → defer

  ## Usage

      case ResponseGate.evaluate(analysis_model, conversation_memory, opts) do
        {:respond, reason} -> generate_response(...)
        {:optional, confidence, reason} -> maybe_respond_based_on_confidence(...)
        {:defer, reason} -> return_no_response(...)
      end
  """

  alias Brain.Analysis.{InternalModel, SpeechActResult, LearningStore}
  require Logger

  @doc "Evaluates whether a response is appropriate given the current analysis and conversation history.\n\nReturns:\n- `{:respond, %{reason: String.t()}}` - Response is expected\n- `{:optional, float(), %{reason: String.t()}}` - Response is situational (confidence 0-1)\n- `{:defer, %{reason: String.t()}}` - Silence is appropriate\n"
  def evaluate(%InternalModel{} = analysis_model, conversation_memory, opts \\ []) do
    current_speech_act = get_primary_speech_act(analysis_model)
    history = get_speech_act_history(conversation_memory)
    learned_params = get_learned_params()

    evaluate_response_optionality(current_speech_act, history, learned_params, opts)
  end

  @doc "Simplified evaluation when you only have the speech act (for testing).\n"
  def evaluate_speech_act(speech_act, history \\ [], opts \\ []) do
    learned_params = get_learned_params()
    evaluate_response_optionality(speech_act, history, learned_params, opts)
  end

  defp evaluate_response_optionality(nil, _history, _params, _opts) do
    {:respond, %{reason: "no speech act classification available"}}
  end

  defp evaluate_response_optionality(current, history, params, opts) do
    category = get_category(current)
    sub_type = get_sub_type(current)
    is_question = get_is_question(current)
    memory = Keyword.get(opts, :conversation_memory, [])

    cond do
      category == :directive ->
        {:respond, %{reason: "directive speech act expects response"}}

      is_question ->
        {:respond, %{reason: "question structure detected"}}

      rapid_fire?(memory) ->
        {:defer, %{reason: "rapid-fire input detected - waiting for user to finish"}}

      self_correction?(current, history, memory) ->
        {:defer, %{reason: "self-correction detected - waiting for amended input"}}

      echo_repetition?(current, history) ->
        {:optional, 0.8, %{reason: "repeated speech act pattern - likely nothing new to add"}}

      gratitude_loop?(current, history) ->
        {:defer, %{reason: "gratitude loop detected: thanks→ack→thanks sequence"}}

      sub_type == :backchannel ->
        confidence = Map.get(params, :backchannel_defer_confidence, 0.8)
        {:optional, confidence, %{reason: "backchannel - minimal response expected"}}

      sub_type == :acknowledgment and recent_thanks?(history) ->
        {:optional, 0.7, %{reason: "acknowledgment following thanks - potential loop"}}

      sub_type == :compliment and not is_question ->
        confidence = Map.get(params, :compliment_defer_confidence, 0.6)
        {:optional, confidence, %{reason: "compliment without question"}}

      sub_type == :continuation ->
        {:defer, %{reason: "continuation detected - waiting for more input"}}

      category == :expressive and not expects_response?(current) ->
        {:optional, 0.5, %{reason: "expressive without clear response expectation"}}

      true ->
        {:respond, %{reason: "default - no deferral conditions met"}}
    end
  end

  @rapid_fire_window_ms 2_000
  @rapid_fire_count 3

  @doc """
  Detects rapid-fire input by checking timestamps of recent user messages.

  If the last N user messages all arrived within a short window, the user
  is likely still typing and we should wait rather than respond to each fragment.
  """
  def rapid_fire?(memory) when is_list(memory) do
    user_timestamps =
      memory
      |> Enum.filter(fn msg -> msg[:role] == "user" and is_integer(msg[:timestamp]) end)
      |> Enum.map(& &1[:timestamp])
      |> Enum.take(-@rapid_fire_count)

    if length(user_timestamps) >= @rapid_fire_count do
      first = List.first(user_timestamps)
      last = List.last(user_timestamps)
      (last - first) < @rapid_fire_window_ms
    else
      false
    end
  end

  def rapid_fire?(_), do: false

  @doc """
  Detects echo/repetition by comparing the current speech act with the
  immediately preceding user speech act. If both have the same category
  and sub_type, the user is repeating themselves and we likely have
  nothing new to add.
  """
  def echo_repetition?(current, history) do
    case Enum.take(history, -1) do
      [prev] ->
        same_category = get_category(current) == get_category(prev)
        same_sub_type = get_sub_type(current) == get_sub_type(prev)
        same_category and same_sub_type and get_category(current) != nil

      _ ->
        false
    end
  end

  @doc """
  Detects self-correction by looking for a pattern where the current message
  amends or restates the previous one. Identified by matching intents with
  a `:correction` or `:amendment` sub_type, or by detecting the same category
  appearing in rapid succession with a `:continuation` variant.
  """
  def self_correction?(current, history, memory) do
    sub = get_sub_type(current)

    if sub in [:correction, :amendment] do
      true
    else
      case Enum.take(history, -1) do
        [prev] ->
          same_category = get_category(current) == get_category(prev)
          recent_pair = recent_user_messages?(memory, 2, 3_000)
          same_category and recent_pair and get_category(current) != nil

        _ ->
          false
      end
    end
  end

  defp recent_user_messages?(memory, count, window_ms) when is_list(memory) do
    user_timestamps =
      memory
      |> Enum.filter(fn msg -> msg[:role] == "user" and is_integer(msg[:timestamp]) end)
      |> Enum.map(& &1[:timestamp])
      |> Enum.take(-count)

    if length(user_timestamps) >= count do
      first = List.first(user_timestamps)
      last = List.last(user_timestamps)
      (last - first) < window_ms
    else
      false
    end
  end

  defp recent_user_messages?(_, _, _), do: false

  @doc "Detects a gratitude loop by examining speech act sequence.\n\nA gratitude loop is: user thanks → bot acknowledges/welcomes → user thanks again\n\nThis is detected purely through speech act sub_types, not string matching.\n"
  def gratitude_loop?(current, history) do
    get_sub_type(current) == :thanks and
      has_recent_pattern?(history, [:thanks, :acknowledgment])
  end

  @doc "Checks if there was a recent thanks in the conversation.\n"
  def recent_thanks?(history) do
    history
    |> Enum.take(-3)
    |> Enum.any?(fn sa -> get_sub_type(sa) == :thanks end)
  end

  @doc "Checks if the recent history matches a pattern of speech act sub_types.\n"
  def has_recent_pattern?(history, pattern) when is_list(pattern) do
    pattern_length = length(pattern)

    recent_sub_types =
      history
      |> Enum.take(-pattern_length)
      |> Enum.map(&get_sub_type/1)

    recent_sub_types == pattern
  end

  defp get_primary_speech_act(%InternalModel{analyses: []}) do
    nil
  end

  defp get_primary_speech_act(%InternalModel{analyses: analyses}) do
    analyses
    |> Enum.max_by(& &1.confidence, fn -> nil end)
    |> case do
      nil -> nil
      analysis -> analysis.speech_act
    end
  end

  defp get_speech_act_history(memory) when is_list(memory) do
    memory
    |> Enum.filter(fn msg ->
      msg[:role] == "user" and is_map(msg[:context])
    end)
    |> Enum.map(fn msg ->
      msg[:context][:speech_act]
    end)
    |> Enum.filter(&(&1 != nil))
  end

  defp get_speech_act_history(_) do
    []
  end

  defp get_category(%SpeechActResult{category: cat}) do
    cat
  end

  defp get_category(%{category: cat}) do
    cat
  end

  defp get_category(_) do
    nil
  end

  defp get_sub_type(%SpeechActResult{sub_type: st}) do
    st
  end

  defp get_sub_type(%{sub_type: st}) do
    st
  end

  defp get_sub_type(_) do
    nil
  end

  defp get_is_question(%SpeechActResult{is_question: q}) do
    q
  end

  defp get_is_question(%{is_question: q}) do
    q
  end

  defp get_is_question(_) do
    false
  end

  defp expects_response?(%SpeechActResult{} = sa) do
    case SpeechActResult.expects_response?(sa) do
      true -> true
      false -> false
      :optional -> false
      :unknown -> true
    end
  end

  defp expects_response?(%{category: :directive}) do
    true
  end

  defp expects_response?(%{sub_type: :greeting}) do
    true
  end

  defp expects_response?(%{sub_type: :backchannel}) do
    false
  end

  defp expects_response?(%{sub_type: :continuation}) do
    false
  end

  defp expects_response?(_) do
    true
  end

  defp get_learned_params do
    if function_exported?(LearningStore, :get_params, 1) do
      case LearningStore.get_params(:response_optionality) do
        {:ok, params} when is_map(params) ->
          atomize_keys(params)

        _ ->
          default_params()
      end
    else
      default_params()
    end
  rescue
    _ -> default_params()
  end

  defp atomize_keys(map) when is_map(map) do
    Map.new(map, fn
      {k, v} when is_binary(k) -> {safe_atomize_key(k), v}
      {k, v} -> {k, v}
    end)
  end

  defp atomize_keys(other) do
    other
  end

  defp safe_atomize_key(k) when is_binary(k) do
    String.to_existing_atom(k)
  rescue
    ArgumentError -> k
  end

  defp default_params do
    %{
      gratitude_loop_threshold: 2,
      compliment_defer_confidence: 0.6,
      backchannel_defer_confidence: 0.8,
      acknowledgment_defer_confidence: 0.7
    }
  end
end
