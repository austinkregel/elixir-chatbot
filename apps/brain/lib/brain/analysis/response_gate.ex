defmodule Brain.Analysis.ResponseGate do
  @moduledoc "Evaluates whether a response is appropriate based on speech act sequences.\n\nThis module reasons over *classified speech acts*, not raw text. All detection\nhappens through the existing analysis passes (structural, pragmatic, keyword, etc.)\nin SpeechActClassifier. ResponseGate simply examines the resulting classifications\nand conversation history to determine if a response is needed.\n\n## Response Optionality Patterns\n\n- **Gratitude loops**: User thanks → Bot acknowledges → User thanks again → defer\n- **Backchannels**: Classified as `:backchannel` by pragmatic pass → optional\n- **Compliments**: Classified as `:compliment` without question → optional\n- **Continuations**: Classified as `:continuation` by structural pass → defer\n\n## Usage\n\n    case ResponseGate.evaluate(analysis_model, conversation_memory, opts) do\n      {:respond, reason} -> generate_response(...)\n      {:optional, confidence, reason} -> maybe_respond_based_on_confidence(...)\n      {:defer, reason} -> return_no_response(...)\n    end\n"

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

  defp evaluate_response_optionality(current, history, params, _opts) do
    category = get_category(current)
    sub_type = get_sub_type(current)
    is_question = get_is_question(current)

    cond do
      category == :directive ->
        {:respond, %{reason: "directive speech act expects response"}}

      is_question ->
        {:respond, %{reason: "question structure detected"}}

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
      {k, v} when is_binary(k) -> {String.to_atom(k), v}
      {k, v} -> {k, v}
    end)
  end

  defp atomize_keys(other) do
    other
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