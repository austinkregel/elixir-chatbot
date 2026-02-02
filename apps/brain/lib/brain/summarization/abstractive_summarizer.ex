defmodule Brain.Summarization.AbstractiveSummarizer do
  @moduledoc """
  Generates abstractive summaries using LSTM + Attention seq2seq models.
  
  Falls back to template-based SummaryBuilder if model is not ready.
  """
  
  require Logger
  
  alias Brain.ML.Seq2Seq
  alias Brain.Summarization.{SummaryBuilder, Types.Summary}
  
  @doc """
  Generate an abstractive summary from ranked facts.
  
  ## Options
  - `:max_length` - Maximum summary length (default: 50)
  - `:world_id` - World ID for world-scoped models (default: "default")
  """
  def summarize(facts, opts \\ []) do
    if Seq2Seq.ready?(world_id: Keyword.get(opts, :world_id, "default")) do
      generate_abstractive(facts, opts)
    else
      Logger.debug("Seq2Seq model not ready, falling back to template-based summarization")
      # Fallback to existing template-based
      SummaryBuilder.build(facts, opts)
    end
  end
  
  defp generate_abstractive(facts, opts) do
    max_length = Keyword.get(opts, :max_length, 50)
    world_id = Keyword.get(opts, :world_id, "default")
    
    # Linearize facts to input sequence
    input_sequence = linearize_facts(facts)
    
    # Run through seq2seq model
    case Seq2Seq.generate(input_sequence, world_id: world_id, max_length: max_length) do
      {:ok, generated_text} ->
        # Create summary with generated text
        Summary.new(
          generated_text,
          facts_used: facts,
          sentences: [generated_text],
          fact_count: length(facts)
        )
      
      {:error, reason} ->
        Logger.warning("Abstractive summarization failed: #{inspect(reason)}, falling back to template")
        SummaryBuilder.build(facts, opts)
    end
  end
  
  defp linearize_facts(facts) do
    # Convert facts to a single text sequence
    # Format: "subject predicate. subject predicate. ..."
    facts
    |> Enum.map(fn fact ->
      subject = fact.subject || "Someone"
      predicate = fact.predicate || ""
      
      if predicate != "" do
        "#{subject} #{predicate}"
      else
        subject
      end
    end)
    |> Enum.join(". ")
    |> String.trim()
  end
end
