defmodule ChatBot.Testing.TaskSchema do
  @moduledoc """
  Schema and utilities for domain-specific NLP benchmark tasks.

  ## Task Structure

  Each task file follows this JSON schema:

  ```json
  {
    "Contributors": ["Author names"],
    "Source": ["Dataset source"],
    "URL": ["Dataset URL"],
    "Categories": ["Task categories"],
    "Reasoning": ["Required reasoning types"],
    "Definition": ["Task description and instructions"],
    "Input_language": ["en"],
    "Output_language": ["en"],
    "Instruction_language": ["en"],
    "Domains": ["Wikipedia", "News", etc.],
    "Positive Examples": [
      {"input": "...", "output": "...", "explanation": "Why this is correct"}
    ],
    "Negative Examples": [
      {"input": "...", "output": "...", "explanation": "Why this is wrong"}
    ],
    "Instances": [
      {"id": "task001-abc", "input": "...", "output": ["..."]}
    ]
  }
  ```

  ## Key Fields

  | Field | Purpose |
  |-------|---------|
  | `Definition` | Explains what the task requires |
  | `Reasoning` | Types of reasoning needed (commonsense, temporal, etc.) |
  | `Positive Examples` | Correct input/output pairs with explanations |
  | `Negative Examples` | Incorrect outputs with explanations of mistakes |
  | `Instances` | Test cases to evaluate |
  | `Domains` | Source domains (Wikipedia, Science, News) |
  | `Categories` | Task categories for capability mapping |
  """

  @type example :: %{
    input: String.t(),
    output: String.t(),
    explanation: String.t()
  }

  @type instance :: %{
    id: String.t(),
    input: String.t(),
    output: [String.t()]
  }

  @type t :: %{
    contributors: [String.t()],
    source: [String.t()],
    url: [String.t()],
    categories: [String.t()],
    reasoning: [String.t()],
    definition: [String.t()],
    input_language: [String.t()],
    output_language: [String.t()],
    instruction_language: [String.t()],
    domains: [String.t()],
    positive_examples: [example()],
    negative_examples: [example()],
    instances: [instance()]
  }

  @doc """
  Parses a raw task map into a structured format.
  """
  @spec parse(map()) :: {:ok, t()} | {:error, term()}
  def parse(raw) when is_map(raw) do
    task = %{
      contributors: get_list(raw, "Contributors"),
      source: get_list(raw, "Source"),
      url: get_list(raw, "URL"),
      categories: get_list(raw, "Categories"),
      reasoning: get_list(raw, "Reasoning"),
      definition: get_list(raw, "Definition"),
      input_language: get_list(raw, "Input_language"),
      output_language: get_list(raw, "Output_language"),
      instruction_language: get_list(raw, "Instruction_language"),
      domains: get_list(raw, "Domains"),
      positive_examples: parse_examples(Map.get(raw, "Positive Examples", [])),
      negative_examples: parse_examples(Map.get(raw, "Negative Examples", [])),
      instances: parse_instances(Map.get(raw, "Instances", []))
    }

    {:ok, task}
  end

  def parse(_), do: {:error, :invalid_format}

  @doc """
  Gets the primary definition text.
  """
  @spec get_definition(t()) :: String.t()
  def get_definition(%{definition: [first | _]}), do: first
  def get_definition(_), do: ""

  @doc """
  Gets the reasoning types required for this task.
  """
  @spec get_reasoning_types(t()) :: [String.t()]
  def get_reasoning_types(%{reasoning: reasoning}), do: reasoning
  def get_reasoning_types(_), do: []

  @doc """
  Gets the primary category.
  """
  @spec get_primary_category(t()) :: String.t() | nil
  def get_primary_category(%{categories: [first | _]}), do: first
  def get_primary_category(_), do: nil

  @doc """
  Gets positive examples for learning expected patterns.
  """
  @spec get_positive_examples(t(), non_neg_integer()) :: [example()]
  def get_positive_examples(task, limit \\ 5)
  def get_positive_examples(%{positive_examples: examples}, limit) do
    Enum.take(examples, limit)
  end
  def get_positive_examples(_, _), do: []

  @doc """
  Gets negative examples for understanding mistakes.
  """
  @spec get_negative_examples(t(), non_neg_integer()) :: [example()]
  def get_negative_examples(task, limit \\ 5)
  def get_negative_examples(%{negative_examples: examples}, limit) do
    Enum.take(examples, limit)
  end
  def get_negative_examples(_, _), do: []

  @doc """
  Extracts patterns from positive examples that could inform evaluation.
  
  Returns a map with:
  - :input_patterns - Common structures in inputs
  - :output_patterns - Common structures in outputs
  - :explanations - Key insights from explanations
  """
  @spec extract_patterns(t()) :: map()
  def extract_patterns(%{positive_examples: examples, negative_examples: neg_examples}) do
    # Extract key terms from explanations
    positive_insights =
      examples
      |> Enum.map(& &1.explanation)
      |> Enum.reject(&is_nil/1)
      |> Enum.reject(&(&1 == ""))

    negative_insights =
      neg_examples
      |> Enum.map(& &1.explanation)
      |> Enum.reject(&is_nil/1)
      |> Enum.reject(&(&1 == ""))

    %{
      positive_count: length(examples),
      negative_count: length(neg_examples),
      positive_insights: positive_insights,
      negative_insights: negative_insights
    }
  end
  def extract_patterns(_), do: %{positive_count: 0, negative_count: 0, positive_insights: [], negative_insights: []}

  @doc """
  Determines if the task is English-only.
  """
  @spec english_only?(t()) :: boolean()
  def english_only?(%{input_language: input, output_language: output}) do
    input_en = Enum.all?(input, &(&1 in ["English", "en", ""]))
    output_en = Enum.all?(output, &(&1 in ["English", "en", ""]))
    input_en and output_en
  end
  def english_only?(_), do: true

  @doc """
  Gets test instances, optionally limited.
  """
  @spec get_instances(t(), non_neg_integer()) :: [instance()]
  def get_instances(task, limit \\ 100)
  def get_instances(%{instances: instances}, limit) do
    Enum.take(instances, limit)
  end
  def get_instances(_, _), do: []

  @doc """
  Summarizes a task for display.
  """
  @spec summarize(t()) :: map()
  def summarize(task) do
    %{
      definition: get_definition(task) |> String.slice(0, 200),
      categories: task.categories,
      reasoning: task.reasoning,
      domains: task.domains,
      instance_count: length(task.instances),
      positive_example_count: length(task.positive_examples),
      negative_example_count: length(task.negative_examples),
      english_only: english_only?(task)
    }
  end

  @doc """
  Maps reasoning types to capabilities.
  """
  @spec reasoning_to_capability(String.t()) :: atom() | nil
  def reasoning_to_capability(reasoning) do
    case String.downcase(reasoning) do
      r when r in ["commonsense reasoning", "commonsense"] -> :commonsense
      r when r in ["temporal reasoning", "temporal"] -> :temporal_reasoning
      r when r in ["numerical reasoning", "numerical", "arithmetic"] -> :numerical
      r when r in ["causal reasoning", "causal"] -> :causal
      r when r in ["analogical reasoning", "analogy"] -> :analogy
      r when r in ["spatial reasoning", "spatial"] -> :spatial
      r when r in ["deductive reasoning", "deductive"] -> :deductive
      r when r in ["abductive reasoning", "abductive"] -> :abductive
      _ -> nil
    end
  end

  @doc """
  Maps categories to capabilities.
  """
  @spec category_to_capability(String.t()) :: atom() | nil
  def category_to_capability(category) do
    case category do
      c when c in ["Question Answering", "Reading Comprehension"] -> :question_answering
      c when c in ["Named Entity Recognition", "Entity Detection"] -> :entity_recognition
      c when c in ["Sentiment Analysis", "Emotion Detection"] -> :sentiment
      c when c in ["Text Classification", "Classification"] -> :classification
      c when c in ["Coreference Resolution"] -> :coreference
      c when c in ["Temporal Reasoning"] -> :temporal_reasoning
      c when c in ["Commonsense Reasoning", "Reasoning"] -> :commonsense
      c when c in ["Textual Entailment", "NLI"] -> :entailment
      c when c in ["Summarization", "Text Summarization"] -> :summarization
      c when c in ["Paraphrasing", "Paraphrase Detection"] -> :paraphrase
      c when c in ["Cause Effect Classification"] -> :causal
      c when c in ["Intent Identification"] -> :intent
      c when c in ["Pos Tagging"] -> :pos_tagging
      _ -> nil
    end
  end

  # Private helpers

  defp get_list(map, key) do
    case Map.get(map, key) do
      list when is_list(list) -> list
      nil -> []
      value -> [value]
    end
  end

  defp parse_examples(examples) when is_list(examples) do
    examples
    |> Enum.map(fn ex ->
      %{
        input: Map.get(ex, "input", ""),
        output: normalize_output(Map.get(ex, "output", "")),
        explanation: Map.get(ex, "explanation", "")
      }
    end)
  end
  defp parse_examples(_), do: []

  defp parse_instances(instances) when is_list(instances) do
    instances
    |> Enum.map(fn inst ->
      %{
        id: Map.get(inst, "id", ""),
        input: Map.get(inst, "input", ""),
        output: normalize_output_list(Map.get(inst, "output", []))
      }
    end)
  end
  defp parse_instances(_), do: []

  defp normalize_output(output) when is_list(output), do: Enum.join(output, "; ")
  defp normalize_output(output) when is_binary(output), do: output
  defp normalize_output(_), do: ""

  defp normalize_output_list(output) when is_list(output), do: output
  defp normalize_output_list(output) when is_binary(output), do: [output]
  defp normalize_output_list(_), do: []
end
