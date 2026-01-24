defmodule ChatBot.Response.Composer do
  @moduledoc """
  Composes multi-part responses using speech act analysis
  and connector templates loaded from data files.

  This module handles:
  - Sorting response parts by speech act priority
  - Selecting appropriate connector phrases between parts
  - Applying confidence-based hedging to responses
  """

  require Logger

  @connectors_path "data/response_connectors.json"

  # Cache for loaded connectors
  @connectors_key :response_composer_connectors

  @doc """
  Weaves multiple response parts into a cohesive response.

  Takes a list of {response_text, analysis} tuples and combines them
  using appropriate connector phrases based on speech act transitions.
  """
  def weave_multi_chunk_response(response_parts, analyses) when is_list(response_parts) do
    connectors = load_connectors()

    # Sort by speech act priority
    sorted_pairs =
      Enum.zip(response_parts, analyses)
      |> Enum.sort_by(fn {_, analysis} -> speech_act_priority(analysis) end)

    # Combine using connector phrases
    combine_with_connectors(sorted_pairs, connectors)
  end

  @doc """
  Applies confidence-based hedging to a response.
  """
  def apply_hedging(response, confidence_level) when is_binary(response) do
    connectors = load_connectors()
    starters = Map.get(connectors, "response_starters", %{})

    hedging_key =
      cond do
        confidence_level >= 0.8 -> "high_confidence"
        confidence_level >= 0.5 -> "medium_confidence"
        true -> "low_confidence"
      end

    case Map.get(starters, hedging_key, [""]) do
      [] -> response
      prefixes -> "#{Enum.random(prefixes)}#{response}"
    end
  end

  @doc """
  Gets a clarification prefix for partial responses.
  """
  def get_clarification_prefix do
    connectors = load_connectors()

    case Map.get(connectors, "clarification_prefixes", []) do
      [] -> ""
      prefixes -> Enum.random(prefixes)
    end
  end

  @doc """
  Gets a connector for partial response with clarification.
  """
  def get_partial_connector do
    connectors = load_connectors()

    case Map.get(connectors, "partial_response_connectors", []) do
      [] -> " "
      phrases -> Enum.random(phrases)
    end
  end

  # Private functions

  defp speech_act_priority(analysis) do
    speech_act = Map.get(analysis, :speech_act, %{})
    category = Map.get(speech_act, :category)
    is_question = Map.get(speech_act, :is_question, false)
    sub_type = Map.get(speech_act, :sub_type)

    cond do
      category == :directive -> 0
      category == :assertive and is_question -> 1
      category == :assertive -> 2
      category == :expressive and sub_type == :greeting -> 3
      category == :expressive -> 4
      true -> 5
    end
  end

  defp combine_with_connectors(sorted_pairs, connectors) do
    connector_map = Map.get(connectors, "connectors", %{})

    sorted_pairs
    |> Enum.with_index()
    |> Enum.map(fn {{response, analysis}, idx} ->
      if idx == 0 do
        response
      else
        # Get previous analysis for transition type
        {_, prev_analysis} = Enum.at(sorted_pairs, idx - 1)

        prev_category = get_category(prev_analysis)
        curr_category = get_category(analysis)

        connector = select_connector(prev_category, curr_category, connector_map)
        "#{connector}#{downcase_if_needed(response, connector)}"
      end
    end)
    |> Enum.filter(&(&1 != nil and &1 != ""))
    |> Enum.join(" ")
  end

  defp get_category(analysis) do
    speech_act = Map.get(analysis, :speech_act, %{})
    category = Map.get(speech_act, :category, :unknown)
    is_question = Map.get(speech_act, :is_question, false)

    if is_question, do: :question, else: category
  end

  defp select_connector(prev_category, curr_category, connector_map) do
    # Build transition key
    key = "#{prev_category}_to_#{curr_category}"

    case Map.get(connector_map, key) do
      nil ->
        # Try default
        Map.get(connector_map, "default", [" "]) |> Enum.random()

      connectors when is_list(connectors) ->
        Enum.random(connectors)

      connector ->
        connector
    end
  end

  defp downcase_if_needed(response, connector) when is_binary(response) do
    # If connector is non-empty, downcase the first letter of the response
    if connector != "" and String.length(connector) > 0 do
      case String.graphemes(response) do
        [first | rest] -> String.downcase(first) <> Enum.join(rest)
        _ -> response
      end
    else
      response
    end
  end

  defp downcase_if_needed(response, _), do: response

  defp load_connectors do
    # Check cache first
    case Process.get(@connectors_key) do
      nil ->
        connectors = load_connectors_from_file()
        Process.put(@connectors_key, connectors)
        connectors

      cached ->
        cached
    end
  end

  defp load_connectors_from_file do
    paths_to_try = [
      @connectors_path,
      Path.join(File.cwd!(), @connectors_path)
    ]

    result =
      Enum.find_value(paths_to_try, fn path ->
        if File.exists?(path) do
          case File.read(path) do
            {:ok, contents} ->
              case Jason.decode(contents) do
                {:ok, data} -> {:ok, data}
                {:error, _} -> nil
              end

            {:error, _} ->
              nil
          end
        end
      end)

    case result do
      {:ok, data} ->
        data

      nil ->
        Logger.warning("Response connectors file not found, using defaults")
        default_connectors()
    end
  end

  defp default_connectors do
    %{
      "connectors" => %{
        "default" => [" ", "Also, ", ""]
      },
      "response_starters" => %{
        "high_confidence" => [""],
        "medium_confidence" => ["I think "],
        "low_confidence" => ["I'm not sure, but "]
      },
      "clarification_prefixes" => ["To help you better, "],
      "partial_response_connectors" => ["However, "]
    }
  end
end
