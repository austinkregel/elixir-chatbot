defmodule Brain.Services.CredentialVault do
  @moduledoc """
  Secure credential storage for external service API keys.

  Provides encrypted storage for sensitive credentials like API keys,
  with world-scoped isolation and disk persistence.

  ## Security Features

  - Credentials encrypted at rest using `Plug.Crypto.encrypt/4`
  - ETS for fast in-memory lookups
  - Disk persistence to a git-ignored location
  - Never exposes raw keys in logs or public APIs
  - World-scoped credentials for training environment isolation

  ## Usage

      # Store a credential
      CredentialVault.store(:weather, :openweathermap_api_key, "abc123")

      # Retrieve a credential
      {:ok, "abc123"} = CredentialVault.get(:weather, :openweathermap_api_key)

      # Check if configured
      CredentialVault.has_credential?(:weather, :openweathermap_api_key)

      # List all services with credentials
      CredentialVault.list_services()

      # Delete a credential
      CredentialVault.delete(:weather, :openweathermap_api_key)
  """

  use GenServer
  require Logger

  @table_name :credential_vault
  @persistence_file "credentials.enc"
  @default_world "default"

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Store a credential for a service.

  ## Parameters
    - service: Atom identifying the service (e.g., :weather, :news)
    - key: Atom identifying the credential (e.g., :api_key, :client_secret)
    - value: The credential value (string)
    - opts: Optional keyword list with :world (defaults to "default")

  ## Returns
    - :ok on success
    - {:error, reason} on failure
  """
  @spec store(atom(), atom(), String.t(), keyword()) :: :ok | {:error, term()}
  def store(service, key, value, opts \\ []) when is_atom(service) and is_atom(key) do
    GenServer.call(__MODULE__, {:store, service, key, value, opts})
  end

  @doc """
  Retrieve a credential for a service.

  ## Parameters
    - service: Atom identifying the service
    - key: Atom identifying the credential
    - opts: Optional keyword list with :world (defaults to "default")

  ## Returns
    - {:ok, value} on success
    - {:error, :not_found} if credential doesn't exist
  """
  @spec get(atom(), atom(), keyword()) :: {:ok, String.t()} | {:error, :not_found}
  def get(service, key, opts \\ []) when is_atom(service) and is_atom(key) do
    GenServer.call(__MODULE__, {:get, service, key, opts})
  end

  @doc """
  Delete a credential for a service.
  """
  @spec delete(atom(), atom(), keyword()) :: :ok
  def delete(service, key, opts \\ []) when is_atom(service) and is_atom(key) do
    GenServer.call(__MODULE__, {:delete, service, key, opts})
  end

  @doc """
  Check if a credential exists.
  """
  @spec has_credential?(atom(), atom(), keyword()) :: boolean()
  def has_credential?(service, key, opts \\ []) when is_atom(service) and is_atom(key) do
    GenServer.call(__MODULE__, {:has_credential?, service, key, opts})
  end

  @doc """
  List all services that have credentials configured.
  """
  @spec list_services(keyword()) :: [atom()]
  def list_services(opts \\ []) do
    GenServer.call(__MODULE__, {:list_services, opts})
  end

  @doc """
  Get all credential keys for a service (not the values).
  """
  @spec list_keys(atom(), keyword()) :: [atom()]
  def list_keys(service, opts \\ []) when is_atom(service) do
    GenServer.call(__MODULE__, {:list_keys, service, opts})
  end

  @doc """
  Check if the vault is ready.
  """
  @spec ready?() :: boolean()
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    table = :ets.new(@table_name, [:set, :private])

    state = %{
      table: table,
      encryption_key: get_encryption_key(),
      persistence_path: get_persistence_path()
    }

    # Load persisted credentials
    state = load_from_disk(state)

    Logger.info("CredentialVault initialized",
      persistence_path: state.persistence_path,
      services_loaded: length(list_services_internal(state))
    )

    {:ok, state}
  end

  @impl true
  def handle_call({:store, service, key, value, opts}, _from, state) do
    world = Keyword.get(opts, :world, @default_world)
    storage_key = {world, service, key}

    # Encrypt the value before storing
    encrypted = encrypt_value(value, state.encryption_key)
    :ets.insert(state.table, {storage_key, encrypted})

    # Persist to disk
    persist_to_disk(state)

    Logger.debug("Credential stored",
      service: service,
      key: key,
      world: world
    )

    # Emit telemetry (without exposing the credential value)
    Brain.Telemetry.emit_credential_operation(:store, service, world)

    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:get, service, key, opts}, _from, state) do
    world = Keyword.get(opts, :world, @default_world)
    storage_key = {world, service, key}

    result =
      case :ets.lookup(state.table, storage_key) do
        [{^storage_key, encrypted}] ->
          case decrypt_value(encrypted, state.encryption_key) do
            {:ok, value} -> {:ok, value}
            {:error, _} -> {:error, :decryption_failed}
          end

        [] ->
          {:error, :not_found}
      end

    {:reply, result, state}
  end

  @impl true
  def handle_call({:delete, service, key, opts}, _from, state) do
    world = Keyword.get(opts, :world, @default_world)
    storage_key = {world, service, key}

    :ets.delete(state.table, storage_key)
    persist_to_disk(state)

    Logger.debug("Credential deleted",
      service: service,
      key: key,
      world: world
    )

    # Emit telemetry
    Brain.Telemetry.emit_credential_operation(:delete, service, world)

    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:has_credential?, service, key, opts}, _from, state) do
    world = Keyword.get(opts, :world, @default_world)
    storage_key = {world, service, key}

    result = :ets.member(state.table, storage_key)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:list_services, opts}, _from, state) do
    world = Keyword.get(opts, :world, @default_world)

    services =
      :ets.tab2list(state.table)
      |> Enum.filter(fn {{w, _service, _key}, _} -> w == world end)
      |> Enum.map(fn {{_w, service, _key}, _} -> service end)
      |> Enum.uniq()

    {:reply, services, state}
  end

  @impl true
  def handle_call({:list_keys, service, opts}, _from, state) do
    world = Keyword.get(opts, :world, @default_world)

    keys =
      :ets.tab2list(state.table)
      |> Enum.filter(fn {{w, s, _key}, _} -> w == world and s == service end)
      |> Enum.map(fn {{_w, _s, key}, _} -> key end)

    {:reply, keys, state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp get_encryption_key do
    # Try to get from Brain config first, then fall back to ChatWeb endpoint
    case Application.get_env(:brain, :credential_encryption_key) do
      nil ->
        # Fall back to ChatWeb endpoint's secret_key_base
        case Application.get_env(:chat_web, ChatWeb.Endpoint) do
          nil ->
            # Generate a random key for testing/dev without config
            Logger.warning(
              "No encryption key configured, generating ephemeral key. " <>
                "Set :brain, :credential_encryption_key in config for persistence."
            )

            :crypto.strong_rand_bytes(32)

          config ->
            # Use secret_key_base from endpoint config
            secret = Keyword.get(config, :secret_key_base, "")

            if byte_size(secret) >= 32 do
              :crypto.hash(:sha256, secret)
            else
              Logger.warning("secret_key_base too short, generating ephemeral key")
              :crypto.strong_rand_bytes(32)
            end
        end

      key when is_binary(key) and byte_size(key) >= 32 ->
        :crypto.hash(:sha256, key)

      key when is_binary(key) ->
        Logger.warning("credential_encryption_key too short (need 32+ bytes)")
        :crypto.hash(:sha256, key <> String.duplicate("0", 32))
    end
  end

  defp get_persistence_path do
    # Store in priv/secrets/ which should be git-ignored
    base_path =
      Application.get_env(:brain, :secrets_path) ||
        Path.join(Brain.priv_path(""), "secrets")

    # Ensure directory exists
    File.mkdir_p!(base_path)

    Path.join(base_path, @persistence_file)
  end

  defp encrypt_value(value, key) when is_binary(value) do
    # Use authenticated encryption
    Plug.Crypto.encrypt(key, "credential_vault", value, max_age: :infinity)
  end

  defp decrypt_value(encrypted, key) when is_binary(encrypted) do
    case Plug.Crypto.decrypt(key, "credential_vault", encrypted, max_age: :infinity) do
      {:ok, value} -> {:ok, value}
      {:error, reason} -> {:error, reason}
    end
  end

  defp persist_to_disk(state) do
    # Get all entries and serialize
    entries = :ets.tab2list(state.table)

    # Encrypt the entire serialized data for additional security
    serialized = :erlang.term_to_binary(entries)
    encrypted = encrypt_value(serialized, state.encryption_key)

    case File.write(state.persistence_path, encrypted) do
      :ok ->
        :ok

      {:error, reason} ->
        Logger.error("Failed to persist credentials",
          path: state.persistence_path,
          reason: inspect(reason)
        )
    end
  end

  defp load_from_disk(state) do
    case File.read(state.persistence_path) do
      {:ok, encrypted} ->
        case decrypt_value(encrypted, state.encryption_key) do
          {:ok, serialized} ->
            entries = :erlang.binary_to_term(serialized)
            Enum.each(entries, fn entry -> :ets.insert(state.table, entry) end)

            Logger.debug("Loaded credentials from disk",
              entries_count: length(entries)
            )

            state

          {:error, reason} ->
            Logger.warning("Failed to decrypt persisted credentials",
              reason: inspect(reason)
            )

            state
        end

      {:error, :enoent} ->
        # File doesn't exist yet, that's fine
        state

      {:error, reason} ->
        Logger.warning("Failed to read persisted credentials",
          path: state.persistence_path,
          reason: inspect(reason)
        )

        state
    end
  end

  defp list_services_internal(state) do
    :ets.tab2list(state.table)
    |> Enum.map(fn {{_w, service, _key}, _} -> service end)
    |> Enum.uniq()
  end
end
