defmodule Brain.Test.HTTPSnapshot do
  @moduledoc """
  HTTP response snapshot system for testing external API integrations.

  This module provides a cassette-style approach to testing external HTTP APIs:
  1. Snapshots are recorded from real API responses
  2. Tests replay snapshots instead of hitting real APIs
  3. Snapshots can be updated when API behavior changes

  ## Usage

  In tests:

      test "fetches data from external API" do
        # Load snapshot for this specific test
        HTTPSnapshot.use_snapshot("semantic_scholar/search_transformer")

        {:ok, papers} = SemanticScholar.search("transformer", limit: 3)
        assert length(papers) == 3
      end

  ## Updating Snapshots

  To record new snapshots or update existing ones:

      MIX_ENV=test mix snapshot.record

  Or update a specific snapshot:

      MIX_ENV=test mix snapshot.record --only semantic_scholar/search_transformer

  ## Snapshot Files

  Snapshots are stored in `test/fixtures/http_snapshots/` as JSON files containing:
  - The request URL and parameters
  - The response status, headers, and body
  - Metadata about when it was recorded
  """

  use GenServer
  require Logger

  @snapshots_dir Path.join([__DIR__, "..", "fixtures", "http_snapshots"])
  @table_name :http_snapshots

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Starts the snapshot server.
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Loads a snapshot for use in the current test.
  Returns {:ok, snapshot} or {:error, :not_found}.
  """
  def use_snapshot(snapshot_name) do
    case load_snapshot(snapshot_name) do
      {:ok, snapshot} ->
        GenServer.call(__MODULE__, {:set_snapshot, snapshot_name, snapshot})
        {:ok, snapshot}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Clears any loaded snapshots.
  """
  def clear_snapshots do
    GenServer.call(__MODULE__, :clear_snapshots)
  end

  @doc """
  Gets the response for a given URL from loaded snapshots.
  Returns {:ok, response} or {:error, :no_snapshot}.
  """
  def get_response(url, params \\ %{}) do
    GenServer.call(__MODULE__, {:get_response, url, params})
  end

  @doc """
  Records a response to a snapshot file.
  """
  def record_snapshot(snapshot_name, url, params, response) do
    snapshot = %{
      "url" => url,
      "params" => params,
      "response" => %{
        "status" => response.status,
        "headers" => response.headers,
        "body" => response.body
      },
      "recorded_at" => DateTime.utc_now() |> DateTime.to_iso8601(),
      "note" => "Auto-recorded. Run `mix snapshot.record` to update."
    }

    path = snapshot_path(snapshot_name)
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Jason.encode!(snapshot, pretty: true))

    Logger.info("Recorded snapshot: #{snapshot_name}")
    :ok
  end

  @doc """
  Lists all available snapshots.
  """
  def list_snapshots do
    Path.wildcard(Path.join(@snapshots_dir, "**/*.json"))
    |> Enum.map(fn path ->
      path
      |> Path.relative_to(@snapshots_dir)
      |> String.replace_suffix(".json", "")
    end)
  end

  @doc """
  Checks if a snapshot exists.
  """
  def snapshot_exists?(snapshot_name) do
    File.exists?(snapshot_path(snapshot_name))
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    # Create ETS table for fast snapshot lookup
    :ets.new(@table_name, [:named_table, :set, :public, read_concurrency: true])
    {:ok, %{}}
  end

  @impl true
  def handle_call({:set_snapshot, name, snapshot}, _from, state) do
    :ets.insert(@table_name, {name, snapshot})
    {:reply, :ok, state}
  end

  @impl true
  def handle_call(:clear_snapshots, _from, state) do
    :ets.delete_all_objects(@table_name)
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:get_response, url, params}, _from, state) do
    # Look through all loaded snapshots for a matching URL
    result =
      :ets.tab2list(@table_name)
      |> Enum.find_value(fn {_name, snapshot} ->
        if matches_request?(snapshot, url, params) do
          {:ok, build_response(snapshot)}
        end
      end)

    reply = result || {:error, :no_snapshot}
    {:reply, reply, state}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp snapshot_path(name) do
    Path.join(@snapshots_dir, "#{name}.json")
  end

  defp load_snapshot(name) do
    path = snapshot_path(name)

    if File.exists?(path) do
      case File.read(path) do
        {:ok, content} ->
          {:ok, Jason.decode!(content)}

        {:error, reason} ->
          {:error, {:read_error, reason}}
      end
    else
      {:error, {:not_found, path}}
    end
  end

  defp matches_request?(snapshot, url, params) do
    # Match if URL contains the snapshot URL (allows for flexible matching)
    snapshot_url = snapshot["url"] || ""
    snapshot_params = snapshot["params"] || %{}

    # Parse URL and extract base URL and embedded params
    {base_url, url_params} = parse_url_with_params(url)

    # Combine URL params with explicit params
    all_params = Map.merge(url_params, stringify_keys(params))

    # URL matching - check if base URLs overlap (extract path from both for comparison)
    url_matches = urls_match?(base_url, snapshot_url) or urls_match?(url, snapshot_url)

    # For params, we check if the query params overlap
    params_match =
      if map_size(snapshot_params) == 0 do
        true
      else
        # Check if any key params from the snapshot match
        Enum.any?(snapshot_params, fn {k, v} ->
          all_params_value = Map.get(all_params, to_string(k))
          # Match if value contains the snapshot value or vice versa
          case {all_params_value, v} do
            {nil, _} -> false
            {val, expected} when is_binary(val) and is_binary(expected) ->
              String.contains?(val, expected) or String.contains?(expected, val)
            {val, expected} ->
              to_string(val) == to_string(expected)
          end
        end)
      end

    url_matches and params_match
  end

  defp urls_match?(url1, url2) do
    # Extract paths from URLs for comparison
    path1 = extract_url_path(url1)
    path2 = extract_url_path(url2)

    # Check for overlap
    String.contains?(path1, path2) or String.contains?(path2, path1) or
      String.contains?(url1, url2) or String.contains?(url2, url1)
  end

  defp extract_url_path(url) do
    case URI.parse(url) do
      %URI{path: nil} -> url
      %URI{path: path} -> path
    end
  end

  defp parse_url_with_params(url) do
    case String.split(url, "?", parts: 2) do
      [base_url, query_string] ->
        params =
          query_string
          |> String.split("&")
          |> Enum.map(fn pair ->
            case String.split(pair, "=", parts: 2) do
              [key, value] -> {key, URI.decode(value)}
              [key] -> {key, ""}
            end
          end)
          |> Map.new()

        {base_url, params}

      [base_url] ->
        {base_url, %{}}
    end
  end

  defp stringify_keys(map) when is_map(map) do
    Map.new(map, fn {k, v} -> {to_string(k), v} end)
  end

  defp build_response(snapshot) do
    resp = snapshot["response"]

    %{
      status: resp["status"],
      headers: resp["headers"] || [],
      body: resp["body"]
    }
  end
end
