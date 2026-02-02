defmodule Brain.Analysis.ParallelPipeline do
  @moduledoc """
  Orchestrates multiple analysis pipelines in parallel.

  Runs the standard NLP analysis pipeline alongside the summarization pipeline,
  collecting results from both without blocking on either.

  ## Use Cases

  - **Conversation Response**: Get immediate response while generating background summary
  - **Memory Creation**: Analyze input for response + create summary for episode memory
  - **Third-Person Analysis**: Generate both conversational and observer perspectives

  ## Example

      # Run both pipelines in parallel
      {:ok, results} = ParallelPipeline.process(input, conversation_history: history)

      # Access individual results
      results.analysis   # Standard NLP analysis model
      results.summary    # Dialogue summary (if history provided)
  """

  alias Brain.Analysis.Pipeline, as: AnalysisPipeline
  alias Brain.Summarization.Pipeline, as: SummarizationPipeline

  require Logger

  @type result :: %{
    analysis: map() | nil,
    summary: map() | nil,
    timings: %{analysis_ms: integer(), summary_ms: integer()}
  }

  @doc """
  Process input through both pipelines in parallel.

  ## Options

  - `:conversation_history` - List of previous messages for summarization context
  - `:skip_summary` - Skip summarization pipeline (default: false)
  - `:skip_analysis` - Skip analysis pipeline (default: false)
  - `:summary_opts` - Options passed to summarization pipeline
  - All other options passed to analysis pipeline
  """
  @spec process(String.t(), keyword()) :: {:ok, result()} | {:error, term()}
  def process(input, opts \\ []) do
    skip_summary = Keyword.get(opts, :skip_summary, false)
    skip_analysis = Keyword.get(opts, :skip_analysis, false)
    summary_opts = Keyword.get(opts, :summary_opts, [])
    conversation_history = Keyword.get(opts, :conversation_history, [])

    # Build dialogue from history for summarization
    dialogue = build_dialogue_from_history(conversation_history, input)

    # Start both pipelines as async tasks
    tasks = []

    tasks = if not skip_analysis do
      analysis_task = Task.async(fn ->
        start = System.monotonic_time(:millisecond)
        result = run_analysis(input, opts)
        elapsed = System.monotonic_time(:millisecond) - start
        {result, elapsed}
      end)
      [{:analysis, analysis_task} | tasks]
    else
      tasks
    end

    tasks = if not skip_summary and dialogue != "" do
      summary_task = Task.async(fn ->
        start = System.monotonic_time(:millisecond)
        result = run_summarization(dialogue, summary_opts)
        elapsed = System.monotonic_time(:millisecond) - start
        {result, elapsed}
      end)
      [{:summary, summary_task} | tasks]
    else
      tasks
    end

    # Await all tasks with timeout
    timeout = Keyword.get(opts, :timeout, 30_000)
    results = await_all(tasks, timeout)

    # Build result struct
    {:ok, %{
      analysis: get_in(results, [:analysis, :result]),
      summary: get_in(results, [:summary, :result]),
      timings: %{
        analysis_ms: get_in(results, [:analysis, :elapsed]) || 0,
        summary_ms: get_in(results, [:summary, :elapsed]) || 0
      }
    }}
  end

  @doc """
  Process input and return only when both pipelines complete.
  Returns a combined result with both analysis and summary.
  """
  @spec process_sync(String.t(), keyword()) :: {:ok, result()} | {:error, term()}
  def process_sync(input, opts \\ []) do
    process(input, opts)
  end

  @doc """
  Process input and fire-and-forget the summary.
  Returns analysis immediately, summary is processed in background.
  """
  @spec process_with_background_summary(String.t(), keyword()) :: 
    {:ok, %{analysis: map()}, Task.t()} | {:error, term()}
  def process_with_background_summary(input, opts \\ []) do
    conversation_history = Keyword.get(opts, :conversation_history, [])
    summary_opts = Keyword.get(opts, :summary_opts, [])
    dialogue = build_dialogue_from_history(conversation_history, input)

    # Start summary in background (fire and forget or caller can await)
    summary_task = if dialogue != "" do
      Task.async(fn ->
        run_summarization(dialogue, summary_opts)
      end)
    else
      nil
    end

    # Run analysis synchronously
    case run_analysis(input, opts) do
      {:ok, analysis} ->
        {:ok, %{analysis: analysis}, summary_task}
      {:error, _} = error ->
        # Cancel summary task if analysis failed
        if summary_task, do: Task.shutdown(summary_task, :brutal_kill)
        error
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp run_analysis(input, opts) do
    try do
      result = AnalysisPipeline.process(input, opts)
      {:ok, result}
    rescue
      e ->
        Logger.warning("Analysis pipeline failed: #{inspect(e)}")
        {:error, e}
    end
  end

  defp run_summarization(dialogue, opts) do
    try do
      case SummarizationPipeline.summarize(dialogue, opts) do
        {:ok, summary} -> {:ok, summary}
        {:error, _} = error -> error
      end
    rescue
      e ->
        Logger.warning("Summarization pipeline failed: #{inspect(e)}")
        {:error, e}
    end
  end

  defp build_dialogue_from_history(history, current_input) when is_list(history) do
    # Convert conversation history to dialogue format
    history_dialogue = 
      history
      |> Enum.map(fn
        %{role: role, content: content} ->
          speaker = if role == :user, do: "User", else: "Bot"
          "#{speaker}: #{content}"
        %{"role" => role, "content" => content} ->
          speaker = if role == "user", do: "User", else: "Bot"
          "#{speaker}: #{content}"
        {role, content} ->
          speaker = if role == :user, do: "User", else: "Bot"
          "#{speaker}: #{content}"
        _ -> nil
      end)
      |> Enum.reject(&is_nil/1)
      |> Enum.join(", ")

    # Add current input
    if history_dialogue != "" do
      history_dialogue <> ", User: #{current_input}"
    else
      "User: #{current_input}"
    end
  end

  defp build_dialogue_from_history(_, current_input), do: "User: #{current_input}"

  defp await_all(tasks, timeout) do
    tasks
    |> Enum.map(fn {key, task} ->
      result = try do
        case Task.await(task, timeout) do
          {{:ok, result}, elapsed} -> %{result: result, elapsed: elapsed}
          {{:error, _}, elapsed} -> %{result: nil, elapsed: elapsed}
          {result, elapsed} -> %{result: result, elapsed: elapsed}
        end
      catch
        :exit, {:timeout, _} ->
          Logger.warning("Pipeline #{key} timed out")
          Task.shutdown(task, :brutal_kill)
          %{result: nil, elapsed: timeout}
      end

      {key, result}
    end)
    |> Map.new()
  end
end
