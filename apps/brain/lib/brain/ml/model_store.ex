defmodule Brain.ML.ModelStore do
  @moduledoc """
  S3/MinIO-backed model storage for containerized training workflows.

  When enabled, training tasks publish `.term` model files to an S3-compatible
  object store (MinIO). Application GenServers download the latest models on
  startup before loading from disk.

  All operations are no-ops when `enabled: false` (the default), preserving
  the existing local-file-only behaviour.

  ## Configuration

      config :brain, Brain.ML.ModelStore,
        enabled: true,
        bucket: "chatbot-models"

      config :ex_aws,
        access_key_id: "minioadmin",
        secret_access_key: "minioadmin",
        s3: [scheme: "http://", host: "minio", port: 9000]
  """

  require Logger

  @default_bucket "chatbot-models"
  @multipart_threshold 5 * 1024 * 1024
  @upload_timeout 300_000
  @download_timeout 120_000

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------

  @doc """
  Uploads a local `.term` file to the model store under a versioned key.

  Returns `{:ok, remote_key}` on success, `{:error, reason}` on failure,
  or `:disabled` when the store is not enabled.

  The `remote_key` is built as `v<timestamp>/<relative_path>` so that every
  training run produces a unique version prefix.
  """
  @spec publish(String.t(), String.t(), keyword()) :: {:ok, String.t()} | {:error, term()} | :disabled
  def publish(local_path, remote_key, opts \\ []) do
    if enabled?() do
      bucket = opts[:bucket] || bucket()
      ensure_bucket(bucket)

      case File.stat(local_path) do
        {:ok, %{size: size}} when size > @multipart_threshold ->
          publish_multipart(local_path, bucket, remote_key, size)

        {:ok, %{size: size}} ->
          publish_single(local_path, bucket, remote_key, size)

        {:error, reason} ->
          Logger.warning("[ModelStore] Cannot stat #{local_path}: #{inspect(reason)}")
          {:error, reason}
      end
    else
      :disabled
    end
  end

  defp publish_multipart(local_path, bucket, remote_key, size) do
    Logger.info("[ModelStore] Multipart upload #{remote_key} (#{div(size, 1024)} KB)")

    result =
      local_path
      |> ExAws.S3.Upload.stream_file()
      |> ExAws.S3.upload(bucket, remote_key)
      |> ExAws.request(http_opts: [recv_timeout: @upload_timeout])

    case result do
      {:ok, _} ->
        Logger.info("[ModelStore] Published #{remote_key} (#{size} bytes, multipart)")
        {:ok, remote_key}

      {:error, reason} ->
        Logger.warning("[ModelStore] Failed multipart upload #{remote_key}: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp publish_single(local_path, bucket, remote_key, _size) do
    case File.read(local_path) do
      {:ok, body} ->
        request = ExAws.S3.put_object(bucket, remote_key, body)

        case ExAws.request(request, http_opts: [recv_timeout: @upload_timeout]) do
          {:ok, _} ->
            Logger.info("[ModelStore] Published #{remote_key} (#{byte_size(body)} bytes)")
            {:ok, remote_key}

          {:error, reason} ->
            Logger.warning("[ModelStore] Failed to publish #{remote_key}: #{inspect(reason)}")
            {:error, reason}
        end

      {:error, reason} ->
        Logger.warning("[ModelStore] Cannot read #{local_path}: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Downloads a model file from the store to a local path.

  Returns `:ok` on success, `{:error, reason}` on failure,
  or `:disabled` when the store is not enabled.
  """
  @spec fetch(String.t(), String.t(), keyword()) :: :ok | {:error, term()} | :disabled
  def fetch(remote_key, local_path, opts \\ []) do
    if enabled?() do
      bucket = opts[:bucket] || bucket()

      case ExAws.request(ExAws.S3.get_object(bucket, remote_key), http_opts: [recv_timeout: @download_timeout]) do
        {:ok, %{body: body}} ->
          File.mkdir_p!(Path.dirname(local_path))
          File.write!(local_path, body)
          Logger.info("[ModelStore] Fetched #{remote_key} -> #{local_path}")
          :ok

        {:error, reason} ->
          Logger.warning("[ModelStore] Failed to fetch #{remote_key}: #{inspect(reason)}")
          {:error, reason}
      end
    else
      :disabled
    end
  end

  @doc """
  Ensures a model file exists locally, downloading from S3 if missing.

  This is the primary integration point for GenServer init functions:
  call this before `File.read!` to transparently pull from MinIO when
  running in a container that starts without local model files.

  When the store is disabled or unreachable, this is a silent no-op so
  existing local-file loading continues to work.
  """
  @spec ensure_local(String.t(), String.t(), keyword()) :: :ok
  def ensure_local(remote_key, local_path, opts \\ []) do
    if enabled?() and not File.exists?(local_path) do
      Logger.info("[ModelStore] Local file missing, fetching #{remote_key}")

      case fetch_with_latest_prefix(remote_key, local_path, opts) do
        :ok ->
          :ok

        _ ->
          case fetch(remote_key, local_path, opts) do
            :ok -> :ok
            {:error, _} -> :ok
          end
      end
    else
      :ok
    end
  end

  @doc """
  Lists version prefixes in the bucket (directories matching `v<timestamp>/`).

  Returns `{:ok, [version_prefix]}` or `{:error, reason}`.
  """
  @spec list_versions(keyword()) :: {:ok, [String.t()]} | {:error, term()} | :disabled
  def list_versions(opts \\ []) do
    if enabled?() do
      bucket = opts[:bucket] || bucket()

      case ExAws.request(ExAws.S3.list_objects(bucket, prefix: "v", delimiter: "/")) do
        {:ok, %{body: %{common_prefixes: prefixes}}} ->
          versions =
            prefixes
            |> Enum.map(& &1.prefix)
            |> Enum.sort(:desc)

          {:ok, versions}

        {:ok, %{body: _}} ->
          {:ok, []}

        {:error, reason} ->
          {:error, reason}
      end
    else
      :disabled
    end
  end

  @doc """
  Returns the latest version prefix, or `nil` if none exist.
  """
  @spec latest_version(keyword()) :: {:ok, String.t() | nil} | {:error, term()} | :disabled
  def latest_version(opts \\ []) do
    case list_versions(opts) do
      {:ok, [latest | _]} -> {:ok, latest}
      {:ok, []} -> {:ok, nil}
      other -> other
    end
  end

  @doc """
  Writes a `latest` pointer file in the bucket containing the given version prefix.
  Consumers can read this to discover which version to pull.
  """
  @spec set_latest(String.t(), keyword()) :: :ok | {:error, term()} | :disabled
  def set_latest(version_prefix, opts \\ []) do
    if enabled?() do
      bucket = opts[:bucket] || bucket()

      case ExAws.request(ExAws.S3.put_object(bucket, "latest", version_prefix)) do
        {:ok, _} ->
          Logger.info("[ModelStore] Set latest -> #{version_prefix}")
          :ok

        {:error, reason} ->
          {:error, reason}
      end
    else
      :disabled
    end
  end

  @doc """
  Reads the `latest` pointer to discover the current version prefix,
  then downloads all objects under that prefix to the given local directory.
  """
  @spec fetch_latest(String.t(), keyword()) :: {:ok, non_neg_integer()} | {:error, term()} | :disabled
  def fetch_latest(local_dir, opts \\ []) do
    if enabled?() do
      bucket = opts[:bucket] || bucket()

      with {:ok, %{body: version_prefix}} <-
             ExAws.request(ExAws.S3.get_object(bucket, "latest")),
           version_prefix <- String.trim(version_prefix),
           {:ok, %{body: %{contents: objects}}} <-
             ExAws.request(ExAws.S3.list_objects(bucket, prefix: version_prefix)) do
        count =
          objects
          |> Enum.reject(&String.ends_with?(&1.key, "/"))
          |> Enum.map(fn obj ->
            relative = String.replace_prefix(obj.key, version_prefix, "")
            local_path = Path.join(local_dir, relative)
            fetch(obj.key, local_path, opts)
          end)
          |> Enum.count(&(&1 == :ok))

        Logger.info("[ModelStore] Fetched #{count} models from #{version_prefix}")
        {:ok, count}
      else
        {:error, reason} -> {:error, reason}
      end
    else
      :disabled
    end
  end

  @doc """
  Generates a version prefix based on the current UTC timestamp.
  """
  @spec version_prefix() :: String.t()
  def version_prefix do
    ts = Calendar.strftime(DateTime.utc_now(), "%Y%m%d%H%M%S")
    "v#{ts}/"
  end

  @doc """
  Returns whether the model store is enabled.
  """
  @spec enabled?() :: boolean()
  def enabled? do
    config()[:enabled] == true
  end

  @doc """
  Clears the cached `latest` version prefix so the next `ensure_local` call
  re-reads it from S3. Useful after a new upload or in tests.
  """
  @spec clear_latest_cache() :: :ok
  def clear_latest_cache do
    Application.delete_env(:brain, :model_store_latest_prefix)
    :ok
  end

  # ---------------------------------------------------------------------------
  # Private
  # ---------------------------------------------------------------------------

  defp fetch_with_latest_prefix(remote_key, local_path, opts) do
    case get_latest_prefix(opts) do
      {:ok, prefix} ->
        versioned_key = prefix <> remote_key
        Logger.info("[ModelStore] Resolving #{remote_key} -> #{versioned_key}")
        fetch(versioned_key, local_path, opts)

      {:error, _} ->
        {:error, :no_latest_prefix}
    end
  end

  defp get_latest_prefix(opts) do
    case Application.get_env(:brain, :model_store_latest_prefix) do
      nil ->
        bucket = opts[:bucket] || bucket()

        case ExAws.request(
               ExAws.S3.get_object(bucket, "latest"),
               http_opts: [recv_timeout: @download_timeout]
             ) do
          {:ok, %{body: prefix}} ->
            prefix = String.trim(prefix)
            prefix = if String.ends_with?(prefix, "/"), do: prefix, else: prefix <> "/"
            Application.put_env(:brain, :model_store_latest_prefix, prefix)
            {:ok, prefix}

          {:error, reason} ->
            Logger.warning("[ModelStore] Could not read latest pointer: #{inspect(reason)}")
            {:error, reason}
        end

      prefix ->
        {:ok, prefix}
    end
  end

  defp config do
    Application.get_env(:brain, __MODULE__, [])
  end

  defp bucket do
    config()[:bucket] || @default_bucket
  end

  defp ensure_bucket(bucket) do
    try do
      case ExAws.request(ExAws.S3.head_bucket(bucket)) do
        {:ok, _} ->
          :ok

        {:error, _} ->
          try do
            case ExAws.request(ExAws.S3.put_bucket(bucket, "us-east-1")) do
              {:ok, _} ->
                Logger.info("[ModelStore] Created bucket #{bucket}")
                :ok

              {:error, reason} ->
                Logger.warning("[ModelStore] Could not create bucket #{bucket}: #{inspect(reason)}")
                :ok
            end
          rescue
            e ->
              Logger.warning("[ModelStore] Could not create bucket #{bucket}: #{inspect(e)}")
              :ok
          end
      end
    rescue
      e ->
        Logger.warning("[ModelStore] Could not check bucket #{bucket}: #{inspect(e)}")
        :ok
    end
  end
end
