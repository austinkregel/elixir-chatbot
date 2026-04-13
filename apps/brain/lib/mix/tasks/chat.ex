defmodule Mix.Tasks.Chat do
  @moduledoc """
  Simulates a chat message through the full Brain pipeline.

  Sends a message through the same path as the web UI and prints the response.
  Use `--debug` to dump the full analysis model as JSON before the response.

  ## Usage

      mix chat "Hello, I'm Austin"
      mix chat "What's the weather in NYC?" --debug
      mix chat "I helped program some of your systems" --debug --world default
  """

  use Mix.Task

  @shortdoc "Simulate a chat message through the Brain pipeline"

  @impl Mix.Task
  def run(args) do
    {opts, positional, _} =
      OptionParser.parse(args, strict: [debug: :boolean, world: :string])

    message =
      case positional do
        [msg | _] -> msg
        [] -> abort("Usage: mix chat \"your message\" [--debug] [--world WORLD_ID]")
      end

    debug? = Keyword.get(opts, :debug, false)
    world_id = Keyword.get(opts, :world, "default")

    info("Booting application...")
    {:ok, _} = Application.ensure_all_started(:brain)
    {:ok, _} = Application.ensure_all_started(:chat_web)

    wait_for_brain()

    info("Creating conversation (world: #{world_id})...")

    {:ok, conversation_id} = Brain.create_conversation(world_id: world_id)

    eval_opts = [include_analysis: debug?]
    info("Sending: #{inspect(message)}")
    separator()

    start = System.monotonic_time(:millisecond)

    case Brain.evaluate(conversation_id, message, eval_opts) do
      {:ok, result} ->
        elapsed = System.monotonic_time(:millisecond) - start

        if debug? do
          print_analysis(result[:analysis_model])
        end

        print_response(result.response, result.processing_method, elapsed)

      {:error, reason} ->
        error("Brain.evaluate failed: #{inspect(reason)}")
    end

    Brain.end_conversation(conversation_id)
  end

  defp wait_for_brain do
    Enum.reduce_while(1..30, nil, fn attempt, _ ->
      case GenServer.whereis(Brain) do
        pid when is_pid(pid) ->
          {:halt, :ok}

        nil ->
          if attempt == 1, do: info("Waiting for Brain GenServer...")
          Process.sleep(1_000)
          {:cont, nil}
      end
    end)
  end

  defp print_analysis(nil) do
    warn("No analysis model available")
    separator()
  end

  defp print_analysis(analysis_model) do
    header("Analysis")

    analysis_model
    |> sanitize_for_json()
    |> Jason.encode!(pretty: true)
    |> IO.puts()

    separator()
  end

  defp print_response(nil, method, elapsed) do
    header("Response")
    IO.puts("[silence — no response generated]")
    IO.puts("")
    detail("Method", inspect(method))
    detail("Elapsed", "#{elapsed}ms")
  end

  defp print_response(response, method, elapsed) do
    header("Response")
    IO.puts(response)
    IO.puts("")
    detail("Method", inspect(method))
    detail("Elapsed", "#{elapsed}ms")
  end

  defp sanitize_for_json(value) when is_struct(value) do
    value
    |> Map.from_struct()
    |> sanitize_for_json()
  end

  defp sanitize_for_json(value) when is_map(value) do
    Map.new(value, fn {k, v} -> {to_string(k), sanitize_for_json(v)} end)
  end

  defp sanitize_for_json(value) when is_list(value) do
    Enum.map(value, &sanitize_for_json/1)
  end

  defp sanitize_for_json(value) when is_tuple(value) do
    value |> Tuple.to_list() |> sanitize_for_json()
  end

  defp sanitize_for_json(value) when is_atom(value), do: Atom.to_string(value)
  defp sanitize_for_json(value) when is_pid(value), do: inspect(value)
  defp sanitize_for_json(value) when is_reference(value), do: inspect(value)
  defp sanitize_for_json(value) when is_function(value), do: inspect(value)
  defp sanitize_for_json(value), do: value

  defp header(title) do
    IO.puts("\n\e[1m═══ #{title} ═══\e[0m\n")
  end

  defp separator do
    IO.puts("\n" <> String.duplicate("─", 60) <> "\n")
  end

  defp detail(label, value) do
    IO.puts("  \e[36m#{String.pad_trailing(label <> ":", 10)}\e[0m #{value}")
  end

  defp info(msg), do: IO.puts("\e[33m#{msg}\e[0m")
  defp warn(msg), do: IO.puts("\e[33m⚠ #{msg}\e[0m")
  defp error(msg), do: IO.puts("\e[31m#{msg}\e[0m")

  defp abort(msg) do
    error(msg)
    System.halt(1)
  end
end
