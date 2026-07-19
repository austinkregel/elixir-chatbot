defmodule Brain.ML.Generation.OpenAICompatible do
  @moduledoc """
  Generation via any OpenAI-compatible `/v1/chat/completions` endpoint — Ollama,
  vLLM, LM Studio, a remote Ouro, OpenAI itself. Stateless HTTP: there is nothing
  to launch, so a whole Fleet shares one server rather than each agent spawning a
  multi-GB model process.

      config :brain, :generation,
        backend: :openai_compatible,
        openai_compatible: [
          base_url: "http://localhost:11434/v1",   # Ollama's OpenAI-compatible API
          model: "llama3.2",
          api_key: nil,                             # or {:system, "OPENAI_API_KEY"}
          timeout: 60_000,
          max_tokens: 4096                          # see note below on reasoning models
        ]

  This is the Fleet default: cheap, shared, and honest — if the endpoint is down,
  `generate/2` returns `{:error, reason}` and the caller surfaces it loudly.

  A note on `max_tokens` and reasoning/"thinking" models (Qwen3, DeepSeek-R1,
  etc.): those models spend tokens on a chain-of-thought preamble *before*
  `content`, all counted against the same `max_tokens` budget. A budget sized
  for a plain chat model's answer can be entirely consumed by the reasoning
  step, leaving `content` empty with `finish_reason: "length"` — a real
  failure (see `extract/1`), not a comprehension problem, and never silently
  papered over. If you switch to a reasoning model and see
  `:truncated_before_content` errors, raise `max_tokens` for it specifically.
  """
  @behaviour Brain.ML.Generation.Backend

  require Logger

  @default_base_url "http://localhost:11434/v1"
  @default_model "llama3.2"
  @default_timeout 60_000
  @default_max_tokens 4096

  @impl true
  def name, do: :openai_compatible

  @impl true
  def children, do: []

  @impl true
  def generate(messages, opts \\ []) when is_list(messages) do
    cfg = cfg()
    url = base_url(cfg) <> "/chat/completions"

    body = %{
      model: cfg[:model] || @default_model,
      messages: Enum.map(messages, &Map.take(&1, [:role, :content])),
      max_tokens: Keyword.get(opts, :max_tokens) || cfg[:max_tokens] || @default_max_tokens,
      temperature: Keyword.get(opts, :temperature, 1.0),
      top_p: Keyword.get(opts, :top_p, 0.7),
      stream: false
    }

    start = System.monotonic_time(:millisecond)

    req_opts =
      [json: body, receive_timeout: cfg[:timeout] || @default_timeout] ++
        auth(cfg) ++ Brain.HTTP.Retry.options("Generation (openai_compatible)", url)

    case Req.post(url, req_opts) do
      {:ok, %Req.Response{status: 200, body: resp}} ->
        elapsed = System.monotonic_time(:millisecond) - start
        Logger.info("Generation (openai_compatible): #{elapsed}ms via #{cfg[:model] || @default_model}")
        extract(resp)

      {:ok, %Req.Response{status: status, body: resp_body}} ->
        Logger.error("Generation (openai_compatible) HTTP #{status}: #{inspect(resp_body)}")
        {:error, {:http_error, status, resp_body}}

      {:error, %Req.TransportError{reason: :econnrefused}} ->
        Logger.warning("Generation (openai_compatible) endpoint not reachable at #{url}")
        {:error, :endpoint_unavailable}

      {:error, reason} ->
        Logger.error("Generation (openai_compatible) request failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @impl true
  def ready? do
    url = base_url(cfg()) <> "/models"
    req_opts = [receive_timeout: 3_000] ++ auth(cfg()) ++ Brain.HTTP.Retry.options("Generation (models)", url, max_retries: 0)

    case Req.get(url, req_opts) do
      {:ok, %Req.Response{status: status}} when status in 200..299 -> true
      _ -> false
    end
  rescue
    _ -> false
  catch
    _, _ -> false
  end

  # ── helpers ────────────────────────────────────────────────────────────────

  defp extract(%{"choices" => [%{"message" => %{"content" => c}} | _]}) when is_binary(c) and c != "",
    do: {:ok, c}

  defp extract(%{"choices" => [%{"text" => t} | _]}) when is_binary(t) and t != "", do: {:ok, t}

  # Reasoning models (Qwen3, DeepSeek-R1, ...) put chain-of-thought in a
  # separate `reasoning` field, charged against the same max_tokens budget as
  # `content` — hitting the limit mid-thought yields exactly this shape:
  # finish_reason "length", empty content, non-empty reasoning. A named error
  # here (instead of the generic :unexpected_response) makes the fix obvious
  # from the log line alone: raise max_tokens for this model.
  defp extract(%{"choices" => [%{"finish_reason" => "length", "message" => %{"content" => "", "reasoning" => r}} | _]})
       when is_binary(r) and r != "" do
    {:error, {:truncated_before_content, "response hit max_tokens while still reasoning — raise :max_tokens for this model"}}
  end

  defp extract(other), do: {:error, {:unexpected_response, other}}

  defp cfg, do: Keyword.get(Brain.ML.Generation.config(), :openai_compatible, [])

  defp base_url(cfg), do: String.trim_trailing(cfg[:base_url] || @default_base_url, "/")

  defp auth(cfg) do
    case resolve(cfg[:api_key]) do
      key when is_binary(key) and key != "" -> [auth: {:bearer, key}]
      _ -> []
    end
  end

  defp resolve({:system, var}), do: System.get_env(var)
  defp resolve(v), do: v
end
