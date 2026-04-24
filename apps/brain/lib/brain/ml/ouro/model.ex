defmodule Brain.ML.Ouro.Model do
  @moduledoc """
  GenServer for the Ouro LoopLM model.

  Manages model lifecycle and provides the `generate/2` function for
  autoregressive text generation.  Supports two backends:

    * `:sidecar` (default) -- delegates to `Brain.ML.Ouro.Client`,
      which calls the Ouro inference server via HTTP.
    * `:bumblebee` -- loads the model in-process with Bumblebee/EXLA
      (legacy; much slower on Apple Silicon).

  The backend is selected via `config :brain, :ml, ouro_backend:`.
  When the sidecar backend is active, `ready?/0` pings the server's
  health endpoint; no model weights are loaded into the BEAM.
  """

  use GenServer
  require Logger

  alias Brain.ML.Ouro.Client

  defstruct [
    :serving,
    :tokenizer,
    :backend,
    ready: false
  ]

  @default_max_new_tokens 1024
  @default_temperature 0.6
  @health_check_interval 10_000

  # --- Public API ---

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc "Returns whether the model is loaded and ready for inference."
  def ready?(name \\ __MODULE__) do
    try do
      GenServer.call(name, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc """
  Generates text from a prompt string or ChatML messages.

  Returns `{:ok, generated_text}`, `:fallback` if model is not ready,
  or `{:error, reason}` if inference fails.
  """
  def generate(messages_or_text, opts \\ [], name \\ __MODULE__) do
    timeout = generation_timeout()

    GenServer.call(name, {:generate, messages_or_text, opts}, timeout)
  catch
    :exit, {:timeout, info} ->
      Logger.error("Ouro generation timed out after #{generation_timeout()}ms: #{inspect(info)}")
      {:error, :timeout}

    :exit, reason ->
      Logger.error("Ouro generation crashed: #{inspect(reason)}")
      {:error, {:crashed, reason}}
  end

  @doc "Reloads model weights (bumblebee) or rechecks sidecar health."
  def reload(name \\ __MODULE__) do
    GenServer.cast(name, :reload)
  end

  @doc "Returns model info."
  def info(name \\ __MODULE__) do
    try do
      GenServer.call(name, :info, 1_000)
    catch
      :exit, _ -> %{ready: false, error: :not_running}
    end
  end

  # --- GenServer Callbacks ---

  @impl true
  def init(_opts) do
    skip_init = Application.get_env(:brain, :skip_ml_init, false)
    ouro_enabled = Application.get_env(:brain, :ouro_enabled, true)
    backend = ouro_backend()

    if not skip_init and ouro_enabled do
      send(self(), :init_backend)
    end

    {:ok, %__MODULE__{backend: backend}}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, state.ready, state}
  end

  def handle_call(:info, _from, state) do
    {:reply, %{ready: state.ready, backend: state.backend}, state}
  end

  def handle_call({:generate, messages_or_text, opts}, _from, state) do
    if state.ready do
      start_time = System.monotonic_time(:millisecond)
      result = do_generate(messages_or_text, opts, state)
      elapsed = System.monotonic_time(:millisecond) - start_time
      Logger.info("Ouro inference complete in #{elapsed}ms (backend=#{state.backend})")
      :erlang.garbage_collect()

      case result do
        {:error, {:http_error, _, %{"error" => %{"message" => msg}}}} ->
          if String.contains?(msg, "Memory pressure") do
            Logger.warning("Ouro sidecar under memory pressure, marking unavailable temporarily")
            Process.send_after(self(), :health_check, 5_000)
            {:reply, :fallback, %{state | ready: false}}
          else
            {:reply, result, state}
          end

        _ ->
          {:reply, result, state}
      end
    else
      {:reply, :fallback, state}
    end
  end

  @impl true
  def handle_cast(:reload, state) do
    send(self(), :init_backend)
    {:noreply, %{state | ready: false}}
  end

  @impl true
  def handle_info(:init_backend, %{backend: :sidecar} = state) do
    {:noreply, check_sidecar_health(state)}
  end

  def handle_info(:init_backend, state) do
    {:noreply, attempt_bumblebee_load(state)}
  end

  def handle_info(:health_check, %{backend: :sidecar} = state) do
    {:noreply, check_sidecar_health(state)}
  end

  def handle_info(:health_check, state) do
    {:noreply, state}
  end

  def handle_info(:load_model, state) do
    {:noreply, attempt_bumblebee_load(state)}
  end

  # --- Sidecar backend ---

  defp check_sidecar_health(state) do
    case Client.health_check() do
      :ok ->
        unless state.ready do
          Logger.info("Ouro: sidecar is healthy, backend=sidecar")
        end

        schedule_health_check()
        %{state | ready: true}

      {:error, reason} ->
        if state.ready do
          Logger.warning("Ouro: sidecar became unavailable: #{inspect(reason)}")
        else
          Logger.debug("Ouro: sidecar not yet available: #{inspect(reason)}")
        end

        schedule_health_check()
        %{state | ready: false}
    end
  end

  defp schedule_health_check do
    interval = health_check_interval()
    Process.send_after(self(), :health_check, interval)
  end

  defp health_check_interval do
    ml_config = Application.get_env(:brain, :ml, [])
    ml_config[:ouro_health_check_interval] || @health_check_interval
  end

  # --- Bumblebee backend (legacy) ---

  defp attempt_bumblebee_load(state) do
    model_dir = models_dir()

    unless File.dir?(model_dir) do
      Logger.warning(
        "Ouro: model directory not found at #{model_dir}. " <>
          "Run `mix ouro.download` to fetch the model."
      )

      state
    else
      do_bumblebee_load(model_dir, state)
    end
  end

  defp do_bumblebee_load(model_dir, state) do
    repo = {:local, model_dir}
    start = System.monotonic_time(:millisecond)

    with {:ok, model_info} <- load_bumblebee_model(repo),
         {:ok, tokenizer} <- Brain.ML.Ouro.Tokenizer.load(repo),
         generation_config <- load_generation_config(repo) do
      elapsed = System.monotonic_time(:millisecond) - start
      seq_len = ouro_sequence_length()

      serving =
        Bumblebee.Text.generation(model_info, tokenizer, generation_config,
          compile: [batch_size: 1, sequence_length: seq_len],
          defn_options: [compiler: EXLA]
        )

      Logger.info("Ouro: compiled with sequence_length=#{seq_len}")
      Logger.info("Ouro: model ready via Bumblebee (loaded in #{elapsed}ms)")

      %{state | serving: serving, tokenizer: tokenizer, ready: true}
    else
      {:error, reason} ->
        Logger.warning("Ouro: failed to load model: #{inspect(reason)}")
        state
    end
  rescue
    e ->
      Logger.warning("Ouro: load crashed: #{Exception.message(e)}")
      state
  end

  defp load_bumblebee_model(repo) do
    Bumblebee.load_model(repo,
      module: Brain.ML.Ouro.Spec,
      architecture: :for_causal_language_modeling,
      type: :bf16,
      backend: {EXLA.Backend, client: :host}
    )
  end

  defp load_generation_config(repo) do
    base =
      case Bumblebee.load_generation_config(repo, spec_module: Brain.ML.Ouro.Spec) do
        {:ok, config} -> config
        {:error, _} -> default_generation_config()
      end

    %{base | max_new_tokens: ouro_max_new_tokens()}
  rescue
    _ -> default_generation_config()
  end

  defp default_generation_config do
    %Bumblebee.Text.GenerationConfig{
      max_new_tokens: ouro_max_new_tokens(),
      strategy: %{type: :multinomial_sampling, top_p: 0.95},
      temperature: @default_temperature
    }
  end

  # --- Generation dispatch ---

  defp do_generate(messages_or_text, opts, %{backend: :sidecar}) do
    messages = normalize_messages(messages_or_text)

    Client.chat_completion(messages,
      max_tokens: Keyword.get(opts, :max_new_tokens, 256),
      temperature: Keyword.get(opts, :temperature, 0.6),
      repetition_penalty: Keyword.get(opts, :repetition_penalty, 1.3)
    )
  end

  defp do_generate(messages, _opts, state) when is_list(messages) do
    prompt = format_chatml(messages)
    do_bumblebee_generate(prompt, state)
  end

  defp do_generate(prompt, _opts, state) when is_binary(prompt) do
    do_bumblebee_generate(prompt, state)
  end

  defp do_bumblebee_generate(prompt, state) do
    case Nx.Serving.run(state.serving, prompt) do
      %{results: [%{text: text} | _]} ->
        {:ok, String.trim(text)}

      other ->
        {:error, "Unexpected serving result: #{inspect(other)}"}
    end
  end

  defp normalize_messages(text) when is_binary(text) do
    [%{role: "user", content: text}]
  end

  defp normalize_messages(messages) when is_list(messages) do
    Enum.map(messages, fn msg ->
      %{
        role: to_string(msg[:role] || msg["role"] || "user"),
        content: to_string(msg[:content] || msg["content"] || "")
      }
    end)
  end

  defp format_chatml(messages) do
    body =
      Enum.map_join(messages, fn %{role: role, content: content} ->
        "<|im_start|>#{role}\n#{content}<|im_end|>\n"
      end)

    body <> "<|im_start|>assistant\n"
  end

  # --- Config helpers ---

  defp ouro_backend do
    ml_config = Application.get_env(:brain, :ml, [])
    ml_config[:ouro_backend] || :sidecar
  end

  defp ouro_max_new_tokens do
    ml_config = Application.get_env(:brain, :ml, [])
    ml_config[:ouro_max_new_tokens] || @default_max_new_tokens
  end

  defp generation_timeout do
    ml_config = Application.get_env(:brain, :ml, [])
    ml_config[:ouro_generation_timeout] || 60_000
  end

  defp ouro_sequence_length do
    ml_config = Application.get_env(:brain, :ml, [])
    ml_config[:ouro_sequence_length] || 4096
  end

  defp models_dir do
    ml_config = Application.get_env(:brain, :ml, [])

    case ml_config[:ouro_models_path] do
      nil ->
        case :code.priv_dir(:brain) do
          {:error, _} -> "priv/ml_models/ouro"
          priv -> Path.join(to_string(priv), "ml_models/ouro")
        end

      path ->
        path
    end
  end
end
