defmodule Brain.Response.DecompressorCollector do
  @moduledoc """
  Collects training pairs from live response generation at two levels:

  1. **Primitive-level** -- `{primitive, rendered_text}` pairs from per-primitive
     template rendering. Useful for fine-tuning individual primitive realization.

  2. **Plan-level** -- `{plan + analysis + context + response}` examples capturing
     the full decompression from structured plan to natural language response.
     Useful for training plan-level realizers.

  Pairs are buffered in ETS and periodically flushed to disk in JSONL format
  (one JSON object per line, append-friendly).

  The collector only stores pairs where the rendered text is non-trivial
  (length > 3, not empty) to avoid noise in the training set.
  """

  use GenServer

  alias Brain.Response.RealizationPacket

  require Logger

  @primitive_table :decompressor_pairs
  @plan_table :decompressor_plans
  @flush_interval_ms 60_000
  @max_buffer_size 500
  @data_dir Path.join(["data", "decompressor"])

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  def ready?(name \\ __MODULE__) do
    try do
      GenServer.call(name, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc """
  Records a primitive->text training pair.

  Called by the SurfaceRealizer after rendering a primitive.
  Only collects pairs with meaningful rendered text.
  """
  def collect(primitive, rendered_text, name \\ __MODULE__)
  def collect(%{type: type, variant: variant, content: content}, rendered_text, name)
      when is_binary(rendered_text) do
    if String.length(rendered_text) > 3 do
      pair = %{
        primitive: %{
          type: to_string(type),
          variant: if(variant, do: to_string(variant)),
          content: serialize_content(content)
        },
        output: rendered_text,
        source: "runtime",
        collected_at: System.system_time(:millisecond)
      }

      try do
        GenServer.cast(name, {:collect_primitive, pair})
      catch
        :exit, _ -> :ok
      end
    end

    :ok
  end

  def collect(_, _, _), do: :ok

  @doc """
  Records a plan-level training example.

  Captures the full plan + serialized analysis + context + rendered response
  as a single training example for plan-level decompression learning.

  Called by SurfaceRealizer after full plan realization.
  """
  def collect_plan(primitives, response, opts, name \\ __MODULE__)
      when is_list(primitives) and is_binary(response) do
    if String.length(response) > 3 do
      analysis = Keyword.get(opts, :analysis)
      unified_context = Keyword.get(opts, :unified_context, %{})

      source =
        cond do
          Keyword.get(opts, :ouro_used, false) -> "ouro"
          true -> "template"
        end

      plan_example = %{
        plan: Enum.map(primitives, &RealizationPacket.serialize_primitive/1),
        analysis: serialize_analysis_summary(analysis),
        context: serialize_context_summary(unified_context),
        response: response,
        source: source,
        primitive_count: length(primitives),
        collected_at: System.system_time(:millisecond)
      }

      try do
        GenServer.cast(name, {:collect_plan, plan_example})
      catch
        :exit, _ -> :ok
      end
    end

    :ok
  rescue
    _ -> :ok
  end

  @doc "Returns the current buffer sizes."
  def stats(name \\ __MODULE__) do
    try do
      GenServer.call(name, :stats, 1_000)
    catch
      :exit, _ -> %{primitive_buffer: 0, plan_buffer: 0, total_collected: 0, status: :unavailable}
    end
  end

  @doc "Forces a flush of both buffers to disk."
  def flush(name \\ __MODULE__) do
    try do
      GenServer.call(name, :flush, 5_000)
    catch
      :exit, _ -> {:error, :unavailable}
    end
  end

  # --- Server callbacks ---

  @impl true
  def init(_opts) do
    :ets.new(@primitive_table, [:named_table, :bag, :public, read_concurrency: true])
    :ets.new(@plan_table, [:named_table, :bag, :public, read_concurrency: true])
    schedule_flush()
    {:ok, %{total_primitives: 0, total_plans: 0}}
  end

  @impl true
  def handle_call(:ready?, _from, state), do: {:reply, true, state}

  def handle_call(:stats, _from, state) do
    primitive_size = :ets.info(@primitive_table, :size)
    plan_size = :ets.info(@plan_table, :size)

    stats = %{
      primitive_buffer: primitive_size,
      plan_buffer: plan_size,
      total_collected: state.total_primitives + state.total_plans,
      total_primitives: state.total_primitives,
      total_plans: state.total_plans,
      status: :ok
    }

    {:reply, stats, state}
  end

  def handle_call(:flush, _from, state) do
    {flushed, new_state} = do_flush(state)
    {:reply, {:ok, flushed}, new_state}
  end

  @impl true
  def handle_cast({:collect_primitive, pair}, state) do
    :ets.insert(@primitive_table, {:pair, pair})
    new_state = %{state | total_primitives: state.total_primitives + 1}

    if :ets.info(@primitive_table, :size) >= @max_buffer_size do
      {_flushed, new_state} = do_flush(new_state)
      {:noreply, new_state}
    else
      {:noreply, new_state}
    end
  end

  def handle_cast({:collect_plan, plan_example}, state) do
    :ets.insert(@plan_table, {:plan, plan_example})
    new_state = %{state | total_plans: state.total_plans + 1}

    if :ets.info(@plan_table, :size) >= @max_buffer_size do
      {_flushed, new_state} = do_flush(new_state)
      {:noreply, new_state}
    else
      {:noreply, new_state}
    end
  end

  @impl true
  def handle_info(:flush, state) do
    {_flushed, new_state} = do_flush(state)
    schedule_flush()
    {:noreply, new_state}
  end

  def handle_info(_, state), do: {:noreply, state}

  # --- Internal ---

  defp do_flush(state) do
    primitives =
      :ets.tab2list(@primitive_table)
      |> Enum.map(fn {:pair, pair} -> pair end)

    plans =
      :ets.tab2list(@plan_table)
      |> Enum.map(fn {:plan, plan} -> plan end)

    :ets.delete_all_objects(@primitive_table)
    :ets.delete_all_objects(@plan_table)

    total = 0

    total =
      if primitives != [] do
        write_jsonl(primitives, "primitive_pairs.jsonl")
        total + length(primitives)
      else
        total
      end

    total =
      if plans != [] do
        write_jsonl(plans, "plan_examples.jsonl")
        total + length(plans)
      else
        total
      end

    {total, state}
  end

  defp write_jsonl(records, filename) do
    path = Path.join([@data_dir, filename])
    File.mkdir_p!(@data_dir)

    lines =
      Enum.map(records, fn record ->
        case Jason.encode(stringify_keys(record)) do
          {:ok, json} -> json
          {:error, _} -> nil
        end
      end)
      |> Enum.reject(&is_nil/1)

    if lines != [] do
      content = Enum.join(lines, "\n") <> "\n"
      File.write!(path, content, [:append])
      Logger.debug("DecompressorCollector: appended #{length(lines)} records to #{path}")
    end
  rescue
    e ->
      Logger.warning("DecompressorCollector: write failed: #{inspect(e)}")
  end

  defp serialize_content(content) when is_map(content) do
    Map.new(content, fn
      {k, v} when is_atom(k) -> {Atom.to_string(k), serialize_value(v)}
      {k, v} -> {k, serialize_value(v)}
    end)
  end

  defp serialize_content(_), do: %{}

  defp serialize_analysis_summary(nil), do: nil

  defp serialize_analysis_summary(analysis) when is_struct(analysis) do
    %{
      "intent" => Map.get(analysis, :intent),
      "confidence" => Map.get(analysis, :confidence),
      "response_strategy" => Map.get(analysis, :response_strategy) |> serialize_value(),
      "speech_act_category" =>
        case Map.get(analysis, :speech_act) do
          nil -> nil
          sa -> Map.get(sa, :category) |> serialize_value()
        end,
      "epistemic_status" => Map.get(analysis, :epistemic_status) |> serialize_value()
    }
  end

  defp serialize_analysis_summary(analysis) when is_map(analysis) do
    serialize_value(analysis)
  end

  defp serialize_analysis_summary(_), do: nil

  defp serialize_context_summary(ctx) when is_map(ctx) and ctx != %{} do
    acc = Map.get(ctx, :accumulator, %{})
    enrichment = Map.get(ctx, :enrichment, %{})

    summary = %{}

    summary =
      if is_map(acc) and acc != %{} do
        summary
        |> Map.put("should_hedge", Map.get(acc, :should_hedge, false))
        |> Map.put("entity_familiarity", Map.get(acc, :entity_familiarity))
      else
        summary
      end

    summary =
      if is_map(enrichment) do
        status = Map.get(enrichment, :enrichment_status)
        if status, do: Map.put(summary, "enrichment_status", serialize_value(status)), else: summary
      else
        summary
      end

    if summary == %{}, do: nil, else: summary
  end

  defp serialize_context_summary(_), do: nil

  defp serialize_value(v) when is_atom(v) and not is_nil(v) and not is_boolean(v),
    do: Atom.to_string(v)

  defp serialize_value(v) when is_list(v), do: Enum.map(v, &serialize_value/1)

  defp serialize_value(v) when is_struct(v) do
    v |> Map.from_struct() |> serialize_value()
  end

  defp serialize_value(v) when is_map(v), do: serialize_content(v)
  defp serialize_value(v), do: v

  defp stringify_keys(map) when is_map(map) do
    Map.new(map, fn
      {k, v} when is_atom(k) -> {Atom.to_string(k), stringify_keys(v)}
      {k, v} -> {k, stringify_keys(v)}
    end)
  end

  defp stringify_keys(list) when is_list(list), do: Enum.map(list, &stringify_keys/1)
  defp stringify_keys(v), do: v

  defp schedule_flush do
    Process.send_after(self(), :flush, @flush_interval_ms)
  end
end
