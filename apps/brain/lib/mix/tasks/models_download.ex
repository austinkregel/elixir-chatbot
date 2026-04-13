defmodule Mix.Tasks.Models.Download do
  @moduledoc """
  Download trained models from S3-compatible storage.

  By default fetches the `latest` version to `priv/ml_models/`.

  ## Usage

      mix models.download [options]

  ## Options

    --version PREFIX   Download a specific version instead of latest
    --list             List available versions and exit
    --bucket NAME      Override the configured bucket name
    --force            Download even when local models already exist

  ## Prerequisites

  Set `MODEL_STORE_ENABLED=true` in `.env` along with the S3 connection
  variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_HOST`, etc.).
  """

  use Mix.Task
  require Logger

  alias Brain.ML.ModelStore

  @shortdoc "Download trained models from S3-compatible storage"

  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          version: :string,
          list: :boolean,
          bucket: :string,
          force: :boolean
        ]
      )

    # Only load configs and start the S3/HTTP stack — we don't need the full
    # Brain/Atlas supervision trees (which would query DB tables that may not
    # exist yet if migrations haven't run).
    Mix.Task.run("app.config")
    Application.load(:brain)
    {:ok, _} = Application.ensure_all_started(:ex_aws)
    {:ok, _} = Application.ensure_all_started(:logger)

    unless System.get_env("MODEL_STORE_ENABLED") == "true" do
      Mix.shell().error("ModelStore is disabled. Set MODEL_STORE_ENABLED=true in .env")
      System.halt(1)
    end

    current = Application.get_env(:brain, ModelStore, [])
    Application.put_env(:brain, ModelStore, Keyword.put(current, :enabled, true))

    bucket_opts = if opts[:bucket], do: [bucket: opts[:bucket]], else: []

    cond do
      opts[:list] ->
        list_versions(bucket_opts)

      !opts[:force] && local_models_present?() ->
        Mix.shell().info("")
        Mix.shell().info("Models already present locally — skipping download.")
        Mix.shell().info("Use --force to re-download.")
        Mix.shell().info("")

      true ->
        download_models(opts, bucket_opts)
    end
  end

  defp list_versions(bucket_opts) do
    Mix.shell().info("")
    Mix.shell().info("Available model versions:")
    Mix.shell().info("")

    case ModelStore.list_versions(bucket_opts) do
      {:ok, []} ->
        Mix.shell().info("  (none)")

      {:ok, versions} ->
        Enum.each(versions, fn version ->
          Mix.shell().info("  #{version}")
        end)

        Mix.shell().info("")
        Mix.shell().info("#{length(versions)} version(s) found")

      {:error, reason} ->
        Mix.shell().error("Failed to list versions: #{inspect(reason)}")
        System.halt(1)
    end

    Mix.shell().info("")
  end

  defp download_models(opts, bucket_opts) do
    models_dir = Brain.priv_path("ml_models")

    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("Model Download from S3")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("  Destination: #{models_dir}")

    if opts[:version] do
      download_version(opts[:version], models_dir, bucket_opts)
    else
      download_latest(models_dir, bucket_opts)
    end
  end

  defp download_latest(models_dir, bucket_opts) do
    Mix.shell().info("  Version:     latest")
    Mix.shell().info("")

    case ModelStore.fetch_latest(models_dir, bucket_opts) do
      {:ok, count} ->
        Mix.shell().info("")
        Mix.shell().info("=" |> String.duplicate(60))
        Mix.shell().info("Download Complete!")
        Mix.shell().info("=" |> String.duplicate(60))
        Mix.shell().info("  Files downloaded: #{count}")
        Mix.shell().info("")

      {:error, reason} ->
        Mix.shell().error("Download failed: #{inspect(reason)}")
        System.halt(1)
    end
  end

  defp download_version(version_prefix, models_dir, bucket_opts) do
    version_prefix =
      if String.ends_with?(version_prefix, "/"),
        do: version_prefix,
        else: version_prefix <> "/"

    Mix.shell().info("  Version:     #{version_prefix}")
    Mix.shell().info("")

    bucket = bucket_opts[:bucket] || Application.get_env(:brain, ModelStore, [])[:bucket] || "chatbot-models"

    case ExAws.request(ExAws.S3.list_objects(bucket, prefix: version_prefix)) do
      {:ok, %{body: %{contents: objects}}} ->
        files =
          objects
          |> Enum.reject(&String.ends_with?(&1.key, "/"))

        if files == [] do
          Mix.shell().error("No files found under version #{version_prefix}")
          System.halt(1)
        end

        count =
          Enum.reduce(files, 0, fn obj, acc ->
            relative = String.replace_prefix(obj.key, version_prefix, "")
            local_path = Path.join(models_dir, relative)
            Mix.shell().info("  Downloading #{relative}...")

            case ModelStore.fetch(obj.key, local_path, bucket_opts) do
              :ok -> acc + 1
              {:error, reason} ->
                Mix.shell().error("    FAILED: #{inspect(reason)}")
                acc
            end
          end)

        Mix.shell().info("")
        Mix.shell().info("=" |> String.duplicate(60))
        Mix.shell().info("Download Complete!")
        Mix.shell().info("=" |> String.duplicate(60))
        Mix.shell().info("  Version:          #{version_prefix}")
        Mix.shell().info("  Files downloaded:  #{count}")
        Mix.shell().info("")

      {:error, reason} ->
        Mix.shell().error("Failed to list objects: #{inspect(reason)}")
        System.halt(1)
    end
  end

  @canary_files ~w(classifier.term entity_model.term pos_model.term)

  defp local_models_present? do
    models_dir = Brain.priv_path("ml_models")

    File.dir?(models_dir) &&
      Enum.all?(@canary_files, fn f -> File.exists?(Path.join(models_dir, f)) end)
  end
end
