defmodule Mix.Tasks.Models.Upload do
  @moduledoc """
  Upload all trained models to S3-compatible storage.

  Walks `priv/ml_models/` recursively, uploads every `.term` and `.json` file
  under a versioned prefix, and sets the `latest` pointer.

  ## Usage

      mix models.upload [options]

  ## Options

    --dry-run       List files that would be uploaded without uploading
    --bucket NAME   Override the configured bucket name

  ## Prerequisites

  Set `MODEL_STORE_ENABLED=true` in `.env` along with the S3 connection
  variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_HOST`, etc.).
  """

  use Mix.Task
  require Logger

  alias Brain.ML.ModelStore

  @shortdoc "Upload trained models to S3-compatible storage"

  @model_extensions ~w(.term .json .gz)

  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          dry_run: :boolean,
          bucket: :string
        ]
      )

    Mix.Task.run("app.start")

    unless ModelStore.enabled?() do
      Mix.shell().error("ModelStore is disabled. Set MODEL_STORE_ENABLED=true in .env")
      System.halt(1)
    end

    models_dir = Brain.priv_path("ml_models")

    unless File.dir?(models_dir) do
      Mix.shell().error("Models directory not found: #{models_dir}")
      System.halt(1)
    end

    files = collect_model_files(models_dir)

    if files == [] do
      Mix.shell().info("No model files found in #{models_dir}")
      System.halt(0)
    end

    version = ModelStore.version_prefix()

    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("Model Upload to S3")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("  Source:  #{models_dir}")
    Mix.shell().info("  Version: #{version}")
    Mix.shell().info("  Files:   #{length(files)}")
    Mix.shell().info("")

    if opts[:dry_run] do
      Mix.shell().info("DRY RUN — files that would be uploaded:")
      Mix.shell().info("")

      Enum.each(files, fn {local_path, relative} ->
        size = File.stat!(local_path).size
        Mix.shell().info("  #{relative} (#{format_bytes(size)})")
      end)

      total = files |> Enum.map(fn {p, _} -> File.stat!(p).size end) |> Enum.sum()
      Mix.shell().info("")
      Mix.shell().info("Total: #{length(files)} files, #{format_bytes(total)}")
    else
      bucket_opts = if opts[:bucket], do: [bucket: opts[:bucket]], else: []

      {uploaded, failed, total_bytes} =
        Enum.reduce(files, {0, 0, 0}, fn {local_path, relative}, {ok, err, bytes} ->
          remote_key = version <> relative
          size = File.stat!(local_path).size
          Mix.shell().info("  Uploading #{relative} (#{format_bytes(size)})...")

          case ModelStore.publish(local_path, remote_key, bucket_opts) do
            {:ok, _} -> {ok + 1, err, bytes + size}
            {:error, reason} ->
              Mix.shell().error("    FAILED: #{inspect(reason)}")
              {ok, err + 1, bytes}
          end
        end)

      case ModelStore.set_latest(version, bucket_opts) do
        :ok -> :ok
        {:error, reason} ->
          Mix.shell().error("  Failed to set latest pointer: #{inspect(reason)}")
      end

      Mix.shell().info("")
      Mix.shell().info("=" |> String.duplicate(60))
      Mix.shell().info("Upload Complete!")
      Mix.shell().info("=" |> String.duplicate(60))
      Mix.shell().info("  Version:  #{version}")
      Mix.shell().info("  Uploaded: #{uploaded} files (#{format_bytes(total_bytes)})")

      if failed > 0 do
        Mix.shell().error("  Failed:   #{failed} files")
      end

      Mix.shell().info("")
    end
  end

  defp collect_model_files(base_dir) do
    base_dir
    |> Path.join("**/*")
    |> Path.wildcard()
    |> Enum.filter(fn path ->
      File.regular?(path) and Path.extname(path) in @model_extensions
    end)
    |> Enum.map(fn path ->
      relative = Path.relative_to(path, base_dir)
      {path, relative}
    end)
    |> Enum.sort_by(fn {_, relative} -> relative end)
  end

  defp format_bytes(bytes) when bytes < 1024, do: "#{bytes} B"
  defp format_bytes(bytes) when bytes < 1024 * 1024, do: "#{Float.round(bytes / 1024, 1)} KB"
  defp format_bytes(bytes), do: "#{Float.round(bytes / (1024 * 1024), 1)} MB"
end
