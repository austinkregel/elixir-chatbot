defmodule Brain.ML.InformalExpansions do
  @moduledoc """
  Data-driven informal speech expansion.

  Loads expansion mappings from `informal_expansions.json` in the configured
  training data directory (`config :brain, :ml, :training_data_path`) and
  provides a simple lookup API.

  The dataset can be regenerated/extended using:
      python scripts/generate_informal_expansions.py

  A contraction ending in a clitic the treebank gives more than one meaning
  ("he's", "she'd"; `Brain.Lexicon.Clitics.ambiguous?/1`) must not be in
  the dataset: loading one raises. Those are split by the tokenizer and read
  by the POS tagger in context.

  The data is required. A missing, unreadable or malformed file fails the
  agent's start, and a lookup before the agent has started raises: running
  without the expansions would silently read "gonna" as an unknown word.

  ## Usage

      iex> InformalExpansions.expand("gonna")
      {:ok, "going to"}

      iex> InformalExpansions.expand("hello")
      :not_found
  """

  use Agent

  alias Brain.Lexicon.Clitics

  @data_file "informal_expansions.json"

  # ============================================================================
  # API
  # ============================================================================

  @doc """
  Starts the InformalExpansions agent, loading data from JSON. The start
  fails when the data cannot be loaded.
  """
  def start_link(_opts \\ []) do
    Agent.start_link(&load_expansions!/0, name: __MODULE__)
  end

  @doc """
  Look up an expansion for an informal token.

  Returns {:ok, expansion} if found, :not_found otherwise.
  Lookup is case-insensitive but preserves input case in output.
  Raises when the agent has not started.
  """
  def expand(token) when is_binary(token) do
    lower = String.downcase(token)

    case get_expansion(lower) do
      nil ->
        :not_found

      expansion ->
        # Preserve original case
        {:ok, preserve_case(token, expansion)}
    end
  end

  @doc """
  Check if a token has an expansion in our dataset. Raises when the agent
  has not started.
  """
  def has_expansion?(token) when is_binary(token) do
    lower = String.downcase(token)
    get_expansion(lower) != nil
  end

  @doc """
  Get all expansions (for debugging/testing). Raises when the agent has not
  started.
  """
  def all_expansions do
    ensure_started!()
    Agent.get(__MODULE__, & &1.expansions)
  end

  @doc """
  Get metadata about the loaded dataset. Raises when the agent has not
  started.
  """
  def metadata do
    ensure_started!()
    Agent.get(__MODULE__, &Map.delete(&1, :expansions))
  end

  @doc """
  Reload the dataset from disk. Raises when the agent has not started or the
  data cannot be loaded.
  """
  def reload do
    ensure_started!()
    Agent.update(__MODULE__, fn _ -> load_expansions!() end)
  end

  @doc """
  Check if the agent is running and ready.
  """
  def ready? do
    Process.whereis(__MODULE__) != nil
  end

  # ============================================================================
  # Private
  # ============================================================================

  defp get_expansion(lower_token) do
    ensure_started!()
    Agent.get(__MODULE__, &Map.get(&1.expansions, lower_token))
  end

  defp ensure_started! do
    unless ready?() do
      raise "InformalExpansions is not started; it loads #{data_path()} at application start. " <>
              "Start the :brain application (or the agent) before expanding text."
    end
  end

  defp data_path do
    :brain
    |> Application.fetch_env!(:ml)
    |> Keyword.fetch!(:training_data_path)
    |> Path.join(@data_file)
  end

  defp load_expansions! do
    path = data_path()

    data =
      case File.read(path) do
        {:ok, contents} ->
          case Jason.decode(contents) do
            {:ok, data} -> data
            {:error, error} -> raise "InformalExpansions: #{path} is not valid JSON: #{Exception.message(error)}"
          end

        {:error, reason} ->
          raise "InformalExpansions: cannot read #{path}: #{:file.format_error(reason)}"
      end

    expansions =
      case data do
        %{"expansions" => expansions} when is_map(expansions) and map_size(expansions) > 0 -> expansions
        _ -> raise "InformalExpansions: #{path} has no non-empty \"expansions\" object"
      end

    refuse_ambiguous_clitics!(expansions, path)

    %{
      loaded: true,
      path: path,
      version: Map.get(data, "version"),
      total_entries: map_size(expansions),
      expansions: expansions
    }
  end

  # A contraction ending in a clitic the treebank reads more than one way
  # ("he's": is, has or the possessive) cannot be expanded without guessing.
  # The tokenizer splits it ("he" + "'s") and the POS tagger reads it in
  # context instead.
  defp refuse_ambiguous_clitics!(expansions, path) do
    ambiguous =
      for {informal, _} <- expansions,
          {_host, clitic} <- [Clitics.split(informal)],
          Clitics.ambiguous?(clitic),
          do: informal

    if ambiguous != [] do
      raise "InformalExpansions: #{path} expands #{inspect(Enum.sort(ambiguous))}, but each ends in a clitic " <>
              "the treebank gives more than one meaning (Brain.Lexicon.Clitics.ambiguous?/1). Remove them: " <>
              "the tokenizer splits them and the POS tagger reads them in context."
    end
  end

  # Preserve the original case pattern when expanding
  defp preserve_case(original, expansion) do
    cond do
      # All uppercase
      original == String.upcase(original) and original != String.downcase(original) ->
        String.upcase(expansion)

      # First letter uppercase (title case)
      first_char_uppercase?(original) ->
        capitalize_first(expansion)

      # Lowercase or mixed - use expansion as-is (lowercase)
      true ->
        expansion
    end
  end

  defp first_char_uppercase?(text) do
    case String.first(text) do
      nil -> false
      char -> char == String.upcase(char) and char != String.downcase(char)
    end
  end

  defp capitalize_first(text) do
    case String.split_at(text, 1) do
      {"", rest} -> rest
      {first, rest} -> String.upcase(first) <> rest
    end
  end
end
