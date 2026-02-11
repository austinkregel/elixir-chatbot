defmodule Mix.Tasks.Snapshot.Record do
  @moduledoc """
  Records HTTP snapshots from real API responses.

  This task makes real HTTP requests to external APIs and saves the
  responses as snapshot files that can be replayed during testing.

  ## Usage

      # Record all defined snapshots
      MIX_ENV=test mix snapshot.record

      # Record a specific snapshot
      MIX_ENV=test mix snapshot.record --name semantic_scholar/search_transformer

      # List available snapshot definitions
      MIX_ENV=test mix snapshot.record --list

      # Force update existing snapshots
      MIX_ENV=test mix snapshot.record --force

  ## Snapshot Definitions

  Snapshots are defined in `test/fixtures/http_snapshots/_definitions.exs`.
  Each definition includes:
  - The snapshot name
  - The URL to request
  - Request parameters
  - Any required API keys

  ## Output

  Snapshots are saved to `test/fixtures/http_snapshots/<name>.json`
  """

  use Mix.Task
  require Logger

  @shortdoc "Record HTTP snapshots from real API responses"

  @snapshots_dir Path.join(["apps", "brain", "test", "fixtures", "http_snapshots"])

  # Snapshot definitions - which requests to record
  @snapshot_definitions [
    %{
      name: "semantic_scholar/search_transformer",
      url: "https://api.semanticscholar.org/graph/v1/paper/search",
      params: %{
        "query" => "transformer attention",
        "limit" => 3,
        "fields" =>
          "paperId,title,abstract,authors,venue,year,citationCount,url,externalIds,fieldsOfStudy"
      }
    },
    %{
      name: "open_alex/search_transformer",
      url: "https://api.openalex.org/works",
      params: %{
        "search" => "transformer attention",
        "per_page" => 3
      }
    },
    %{
      name: "open_alex/search_cs",
      url: "https://api.openalex.org/works",
      params: %{
        "search" => "machine learning",
        "filter" => "concepts.id:C41008148",
        "per_page" => 3
      }
    },
    %{
      name: "arxiv/search_transformer",
      url: "http://export.arxiv.org/api/query",
      params: %{
        "search_query" => "all:transformer attention",
        "max_results" => 3
      }
    },
    %{
      name: "weather/london",
      url: "https://api.openweathermap.org/data/2.5/weather",
      params: %{
        "q" => "London",
        "units" => "metric"
      },
      requires_api_key: {:env, "OPENWEATHERMAP_API_KEY", "appid"}
    }
  ]

  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          name: :string,
          list: :boolean,
          force: :boolean
        ]
      )

    # Start HTTP client
    Application.ensure_all_started(:req)

    if Keyword.get(opts, :list, false) do
      list_definitions()
    else
      record_snapshots(opts)
    end
  end

  defp list_definitions do
    Mix.shell().info("Available snapshot definitions:")
    Mix.shell().info("")

    Enum.each(@snapshot_definitions, fn def ->
      exists = File.exists?(snapshot_path(def.name))
      status = if exists, do: "[recorded]", else: "[not recorded]"
      Mix.shell().info("  #{status} #{def.name}")
    end)

    Mix.shell().info("")
    Mix.shell().info("Run `mix snapshot.record` to record all snapshots")
    Mix.shell().info("Run `mix snapshot.record --name <name>` to record a specific snapshot")
  end

  defp record_snapshots(opts) do
    force? = Keyword.get(opts, :force, false)
    target_name = Keyword.get(opts, :name)

    definitions =
      if target_name do
        Enum.filter(@snapshot_definitions, fn d -> d.name == target_name end)
      else
        @snapshot_definitions
      end

    if Enum.empty?(definitions) do
      Mix.shell().error("No matching snapshot definitions found")
      exit({:shutdown, 1})
    end

    Mix.shell().info("Recording #{length(definitions)} snapshot(s)...")
    Mix.shell().info("")

    results =
      Enum.map(definitions, fn definition ->
        record_one(definition, force?)
      end)

    successful = Enum.count(results, fn {status, _} -> status == :ok end)
    skipped = Enum.count(results, fn {status, _} -> status == :skipped end)
    failed = Enum.count(results, fn {status, _} -> status == :error end)

    Mix.shell().info("")
    Mix.shell().info("Done! #{successful} recorded, #{skipped} skipped, #{failed} failed")
  end

  defp record_one(definition, force?) do
    name = definition.name
    path = snapshot_path(name)

    if File.exists?(path) and not force? do
      Mix.shell().info("  [skip] #{name} (exists, use --force to overwrite)")
      {:skipped, name}
    else
      Mix.shell().info("  [recording] #{name}...")

      case make_request(definition) do
        {:ok, response} ->
          save_snapshot(name, definition, response)
          Mix.shell().info("  [ok] #{name}")
          {:ok, name}

        {:error, reason} ->
          Mix.shell().error("  [error] #{name}: #{inspect(reason)}")
          {:error, name}
      end
    end
  end

  defp make_request(definition) do
    url = definition.url
    params = definition.params

    # Add API key if required
    params =
      case Map.get(definition, :requires_api_key) do
        {:env, env_var, param_name} ->
          case System.get_env(env_var) do
            nil ->
              Mix.shell().info("    (skipping - #{env_var} not set)")
              throw({:missing_api_key, env_var})

            key ->
              Map.put(params, param_name, key)
          end

        _ ->
          params
      end

    try do
      # Add delay to respect rate limits
      Process.sleep(500)

      response = Req.get!(url, params: params)

      {:ok,
       %{
         status: response.status,
         headers: Map.new(response.headers),
         body: response.body
       }}
    rescue
      e -> {:error, Exception.message(e)}
    catch
      {:missing_api_key, _} = err -> {:error, err}
    end
  end

  defp save_snapshot(name, definition, response) do
    snapshot = %{
      "name" => name,
      "url" => definition.url,
      "params" => definition.params,
      "response" => %{
        "status" => response.status,
        "headers" => response.headers,
        "body" => response.body
      },
      "recorded_at" => DateTime.utc_now() |> DateTime.to_iso8601(),
      "update_command" => "MIX_ENV=test mix snapshot.record --name #{name} --force"
    }

    path = snapshot_path(name)
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Jason.encode!(snapshot, pretty: true))
  end

  defp snapshot_path(name) do
    Path.join(@snapshots_dir, "#{name}.json")
  end
end
