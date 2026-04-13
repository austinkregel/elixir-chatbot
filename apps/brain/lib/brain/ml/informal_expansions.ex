defmodule Brain.ML.InformalExpansions do
  @moduledoc """
  Data-driven informal speech expansion.

  Loads expansion mappings from data/informal_expansions.json and provides
  a simple lookup API. This replaces complex heuristic-based expansion with
  a trainable, data-driven approach.

  The dataset can be regenerated/extended using:
      python scripts/generate_informal_expansions.py

  ## Usage

      iex> InformalExpansions.expand("gonna")
      {:ok, "going to"}

      iex> InformalExpansions.expand("hello")
      :not_found
  """

  use Agent

  @data_file "data/informal_expansions.json"

  # ============================================================================
  # API
  # ============================================================================

  @doc """
  Starts the InformalExpansions agent, loading data from JSON.
  """
  def start_link(_opts \\ []) do
    Agent.start_link(&load_expansions/0, name: __MODULE__)
  end

  @doc """
  Look up an expansion for an informal token.

  Returns {:ok, expansion} if found, :not_found otherwise.
  Lookup is case-insensitive but preserves input case in output.
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
  Check if a token has an expansion in our dataset.
  """
  def has_expansion?(token) when is_binary(token) do
    lower = String.downcase(token)
    get_expansion(lower) != nil
  end

  @doc """
  Get all expansions (for debugging/testing).
  """
  def all_expansions do
    case Process.whereis(__MODULE__) do
      nil -> %{}
      _pid -> Agent.get(__MODULE__, & &1.expansions)
    end
  end

  @doc """
  Get metadata about the loaded dataset.
  """
  def metadata do
    case Process.whereis(__MODULE__) do
      nil -> %{loaded: false}
      _pid -> Agent.get(__MODULE__, &Map.delete(&1, :expansions))
    end
  end

  @doc """
  Reload the dataset from disk.
  """
  def reload do
    case Process.whereis(__MODULE__) do
      nil ->
        {:error, :not_started}

      _pid ->
        Agent.update(__MODULE__, fn _ -> load_expansions() end)
        :ok
    end
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
    case Process.whereis(__MODULE__) do
      nil ->
        # Agent not started - return nil immediately.
        # Fallback file reads per-token are too expensive during training
        # (5000+ samples x many tokens = thousands of disk reads).
        # The system works without expansions; they're a nice-to-have.
        nil

      _pid ->
        Agent.get(__MODULE__, fn state ->
          Map.get(state.expansions, lower_token)
        end)
    end
  end

  defp load_expansions do
    path = Path.join(Application.app_dir(:brain, "priv"), "../" <> @data_file)

    # Try multiple paths since we might be in different contexts
    # (umbrella root, app directory, or compiled app context)
    paths_to_try = [
      @data_file,
      Path.join(File.cwd!(), @data_file),
      # Umbrella app context: apps/brain -> ../../data/...
      Path.join([File.cwd!(), "..", "..", @data_file]),
      path
    ]

    result =
      Enum.find_value(paths_to_try, fn p ->
        if File.exists?(p) do
          case File.read(p) do
            {:ok, contents} ->
              case Jason.decode(contents) do
                {:ok, data} -> {:ok, p, data}
                {:error, _} -> nil
              end

            {:error, _} ->
              nil
          end
        end
      end)

    case result do
      {:ok, loaded_path, data} ->
        %{
          loaded: true,
          path: loaded_path,
          version: Map.get(data, "version", "unknown"),
          total_entries: Map.get(data, "total_entries", 0),
          expansions: Map.get(data, "expansions", %{})
        }

      nil ->
        # No data file found - start with empty expansions
        # The system will work, just without informal expansion
        %{
          loaded: false,
          path: nil,
          version: nil,
          total_entries: 0,
          expansions: %{}
        }
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
