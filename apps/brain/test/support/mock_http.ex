defmodule Brain.Test.MockHTTP do
  @moduledoc """
  Mock HTTP client that uses snapshots for testing.

  This module implements the same interface as `Req` but returns
  responses from recorded snapshots instead of making real HTTP requests.

  ## Configuration

  In `config/test.exs`:

      config :brain, :http_client, Brain.Test.MockHTTP

  ## Behavior

  1. When a request is made, it looks for a matching snapshot
  2. If found, returns the recorded response
  3. If not found, raises an error with instructions to record the snapshot

  ## Recording New Snapshots

  Run: `MIX_ENV=test mix snapshot.record --name <snapshot_name>`
  """

  require Logger

  alias Brain.Test.HTTPSnapshot

  @doc """
  Makes a GET request, returning snapshot data if available.

  Implements the Req.get!/2 interface.
  Returns a struct-like map with :status, :headers, and :body keys.
  """
  def get!(url, opts \\ []) do
    params = Keyword.get(opts, :params, %{})

    # Also check if URL contains query params (for weather service)
    url_for_matching =
      if String.contains?(url, "?") do
        # Parse query params from URL for matching
        url
      else
        url
      end

    case HTTPSnapshot.get_response(url_for_matching, params) do
      {:ok, response} ->
        %{
          status: response.status,
          headers: response.headers,
          body: response.body
        }

      {:error, :no_snapshot} ->
        raise """
        No HTTP snapshot found for request:
          URL: #{url}
          Params: #{inspect(params)}

        To record a snapshot, run:
          MIX_ENV=test mix snapshot.record

        Or create a fixture file manually at:
          test/fixtures/http_snapshots/<name>.json
        """
    end
  end

  # URLs that are used to test error handling - return appropriate errors silently
  @test_error_urls [
    "not-a-valid-url",
    "httpstat.us",
    "test-rate-limit.com",
    "example-timeout.test",
    "invalid-domain.test"
  ]

  @doc """
  Makes a GET request, returning {:ok, response} or {:error, reason}.

  Implements the Req.get/2 interface.
  This is the primary interface used by Brain services.
  """
  def get(url, opts \\ []) do
    # Check if this is a test URL for error handling (don't warn, just return error)
    if is_test_error_url?(url) do
      {:error, {:test_error_url, url}}
    else
      params = Keyword.get(opts, :params, %{})

      # Parse query params from URL if present (weather service embeds params in URL)
      {base_url, url_params} = parse_url_params(url)
      all_params = Map.merge(url_params, stringify_keys(params))

      case HTTPSnapshot.get_response(base_url, all_params) do
        {:ok, response} ->
          {:ok,
           %{
             status: response.status,
             headers: response.headers,
             body: response.body
           }}

        {:error, :no_snapshot} ->
          error_msg = """
          No HTTP snapshot found for request:
            URL: #{url}
            Base URL: #{base_url}
            Params: #{inspect(all_params)}

          To record a snapshot, run:
            MIX_ENV=test mix snapshot.record

          Or create a fixture file manually at:
            test/fixtures/http_snapshots/<name>.json
          """

          Logger.warning(error_msg)
          {:error, {:no_snapshot, url}}
      end
    end
  end

  defp is_test_error_url?(url) when is_binary(url) do
    Enum.any?(@test_error_urls, fn pattern -> String.contains?(url, pattern) end)
  end

  defp is_test_error_url?(_), do: false

  defp parse_url_params(url) do
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

  defp stringify_keys(_), do: %{}

  @doc """
  Makes a POST request, returning snapshot data if available.
  """
  def post!(url, opts \\ []) do
    body = Keyword.get(opts, :body, %{})
    json = Keyword.get(opts, :json, %{})
    params = Map.merge(body || %{}, json || %{})

    case HTTPSnapshot.get_response(url, params) do
      {:ok, response} ->
        %{
          status: response.status,
          headers: response.headers,
          body: response.body
        }

      {:error, :no_snapshot} ->
        raise """
        No HTTP snapshot found for POST request:
          URL: #{url}
          Body: #{inspect(params)}

        To record a snapshot, run:
          MIX_ENV=test mix snapshot.record
        """
    end
  end

  @doc """
  Makes a POST request, returning {:ok, response} or {:error, reason}.
  """
  def post(url, opts \\ []) do
    body = Keyword.get(opts, :body, %{})
    json = Keyword.get(opts, :json, %{})
    params = Map.merge(body || %{}, json || %{})

    case HTTPSnapshot.get_response(url, params) do
      {:ok, response} ->
        {:ok,
         %{
           status: response.status,
           headers: response.headers,
           body: response.body
         }}

      {:error, :no_snapshot} ->
        {:error, {:no_snapshot, url}}
    end
  end
end
