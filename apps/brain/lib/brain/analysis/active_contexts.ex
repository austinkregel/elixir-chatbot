defmodule Brain.Analysis.ActiveContexts do
  @moduledoc """
  The conversational contexts a turn continues from.

  A follow-up ("to 5", "a bit more", "what about the kitchen") is only
  intelligible after the turn it follows. The intent registry records this the
  way the Dialogflow export did: an intent sets `output_contexts`, each with a
  lifespan in turns, and an intent that can continue from a context lists it in
  `input_contexts`.

  ## At runtime

  The active contexts of a turn are the `output_contexts` of the previous user
  turns' intents that are still within their lifespan: a context set `n` turns
  ago is active while `n <= lifespan`. An intent with no registry entry sets no
  context.

  ## In training and evaluation

  A gold row from a follow-up carries `followup_context`, the context the
  utterance continues from; `for_row/1` gives the active contexts the pipeline is
  run with for that row. A standalone row has none.

  ## The vocabulary

  Every context some intent continues from (the union of the registry's
  `input_contexts`) is one dimension of the feature vector's follow-up group, so
  the context reaches the classifier rather than ending in a lookup table. The
  vocabulary and the output contexts are read from the registry once at boot;
  a registry that changes needs a restart and retrained models.
  """

  alias Brain.Analysis.SlotDetector

  @enforce_keys [:vocabulary, :outputs]
  defstruct [:vocabulary, :outputs]

  @type t :: %__MODULE__{
          vocabulary: [String.t()],
          outputs: %{String.t() => [{String.t(), pos_integer()}]}
        }

  @doc "Builds the tables from an intent registry map."
  @spec from_registry(map()) :: t()
  def from_registry(registry) when is_map(registry) do
    vocabulary =
      registry
      |> Enum.flat_map(fn {_intent, schema} -> Enum.map(Map.get(schema, "input_contexts", []), &context_name!/1) end)
      |> Enum.uniq()
      |> Enum.sort()

    outputs =
      Map.new(registry, fn {intent, schema} ->
        {intent,
         Enum.map(Map.get(schema, "output_contexts", []), fn ctx ->
           {context_name!(ctx), lifespan!(ctx, intent)}
         end)}
      end)

    %__MODULE__{vocabulary: vocabulary, outputs: outputs}
  end

  @doc """
  The contexts active for the next turn, given the conversation's earlier user
  turns in chronological order (each a map with `:intent`). Sorted.
  """
  @spec from_history([map()], t()) :: [String.t()]
  def from_history(history, %__MODULE__{outputs: outputs}) when is_list(history) do
    history
    |> Enum.reverse()
    |> Enum.with_index(1)
    |> Enum.flat_map(fn {turn, turns_ago} ->
      outputs
      |> Map.get(to_string(Map.get(turn, :intent) || ""), [])
      |> Enum.filter(fn {_name, lifespan} -> turns_ago <= lifespan end)
      |> Enum.map(&elem(&1, 0))
    end)
    |> Enum.uniq()
    |> Enum.sort()
  end

  @doc "The active contexts a gold row is analyzed with: its `followup_context`, if any."
  @spec for_row(map()) :: [String.t()]
  def for_row(%{"followup_context" => context}) when is_binary(context), do: [context]
  def for_row(%{} = _row), do: []

  @doc "Returns `names` sorted when every one is in the vocabulary; raises otherwise."
  @spec validate!([String.t()], t()) :: [String.t()]
  def validate!(names, %__MODULE__{vocabulary: vocabulary}) when is_list(names) do
    case Enum.reject(names, &(&1 in vocabulary)) do
      [] -> names |> Enum.uniq() |> Enum.sort()
      unknown -> raise ArgumentError, "ActiveContexts: #{inspect(unknown)} are not contexts any registry intent continues from"
    end
  end

  @doc "The one-hot follow-up feature values for `active`, in vocabulary order."
  @spec features([String.t()], t()) :: [float()]
  def features(active, %__MODULE__{vocabulary: vocabulary}) when is_list(active) do
    Enum.map(vocabulary, fn name -> if name in active, do: 1.0, else: 0.0 end)
  end

  @doc "The feature names, one per context in the vocabulary."
  @spec names(t()) :: [atom()]
  def names(%__MODULE__{vocabulary: vocabulary}), do: Enum.map(vocabulary, &String.to_atom("ctx_" <> &1))

  @doc "The tables loaded at boot."
  @spec loaded!() :: t()
  def loaded! do
    case :persistent_term.get(__MODULE__, nil) do
      %__MODULE__{} = tables -> tables
      nil -> raise "ActiveContexts: not loaded; Brain.Analysis.ActiveContexts must start before analysis runs"
    end
  end

  def child_spec(_opts), do: %{id: __MODULE__, start: {__MODULE__, :start_link, []}, restart: :temporary}

  @doc false
  def start_link do
    :persistent_term.put(__MODULE__, from_registry(SlotDetector.registry()))
    :ignore
  end

  defp context_name!(%{"name" => name}) when is_binary(name) and name != "", do: name
  defp context_name!(other), do: raise("ActiveContexts: registry context without a name: #{inspect(other)}")

  defp lifespan!(%{"lifespan" => n}, _intent) when is_integer(n) and n > 0, do: n
  defp lifespan!(ctx, intent), do: raise("ActiveContexts: #{intent} output context #{inspect(ctx)} has no positive lifespan")
end
