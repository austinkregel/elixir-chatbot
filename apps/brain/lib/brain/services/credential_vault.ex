defmodule Brain.Services.CredentialVault do
  @moduledoc """
  Secure credential storage for external service API keys.

  Provides encrypted storage for sensitive credentials like API keys,
  with world-scoped isolation.

  ## Security Features

  - Credentials encrypted at rest in Postgres via `Plug.Crypto.encrypt/4`
  - ETS for fast in-memory lookups (decrypted on load)
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
    GenServer.call(__MODULE__, {:store, service, key, value, opts}, 5_000)
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
    GenServer.call(__MODULE__, {:get, service, key, opts}, 5_000)
  end

  @doc """
  Delete a credential for a service.
  """
  @spec delete(atom(), atom(), keyword()) :: :ok
  def delete(service, key, opts \\ []) when is_atom(service) and is_atom(key) do
    GenServer.call(__MODULE__, {:delete, service, key, opts}, 5_000)
  end

  @doc """
  Check if a credential exists.
  """
  @spec has_credential?(atom(), atom(), keyword()) :: boolean()
  def has_credential?(service, key, opts \\ []) when is_atom(service) and is_atom(key) do
    GenServer.call(__MODULE__, {:has_credential?, service, key, opts}, 5_000)
  end

  @doc """
  List all services that have credentials configured.
  """
  @spec list_services(keyword()) :: [atom()]
  def list_services(opts \\ []) do
    GenServer.call(__MODULE__, {:list_services, opts}, 5_000)
  end

  @doc """
  Get all credential keys for a service (not the values).
  """
  @spec list_keys(atom(), keyword()) :: [atom()]
  def list_keys(service, opts \\ []) when is_atom(service) do
    GenServer.call(__MODULE__, {:list_keys, service, opts}, 5_000)
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
      encryption_key: get_encryption_key()
    }

    state = load_from_database(state)

    Logger.info("CredentialVault initialized",
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

    persist_to_database(state)

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
    persist_to_database(state)

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

  defp persist_to_database(state) do
    entries = :ets.tab2list(state.table)

    Brain.AtlasIntegration.async(fn ->
      Enum.each(entries, fn {{world, service, key}, encrypted_value} ->
        attrs = %{
          world: to_string(world),
          service: to_string(service),
          key: to_string(key),
          encrypted_value: encrypted_value
        }

        %Atlas.Schemas.Credential{}
        |> Atlas.Schemas.Credential.changeset(attrs)
        |> Atlas.Repo.insert(
          on_conflict: {:replace, [:encrypted_value, :updated_at]},
          conflict_target: [:world, :service, :key]
        )
      end)
    end)
  end

  defp load_from_database(state) do
    case Brain.AtlasIntegration.sync(fn ->
           credentials = Atlas.Repo.all(Atlas.Schemas.Credential)

           Enum.each(credentials, fn cred ->
             storage_key = {cred.world, String.to_atom(cred.service), String.to_atom(cred.key)}
             :ets.insert(state.table, {storage_key, cred.encrypted_value})
           end)

           Logger.debug("Loaded credentials from database",
             entries_count: length(credentials)
           )
         end) do
      {:ok, _} -> state
      {:error, reason} ->
        Logger.warning("Failed to load credentials from database: #{reason}")
        state
    end
  end

  defp list_services_internal(state) do
    :ets.tab2list(state.table)
    |> Enum.map(fn {{_w, service, _key}, _} -> service end)
    |> Enum.uniq()
  end
end
