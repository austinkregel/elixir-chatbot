defmodule ChatWeb.CodeAnalysisLive do
  @moduledoc "LiveView for interacting with the code analysis system.\n\nProvides:\n- Code parsing and symbol extraction\n- Symbol browser and search\n- Relationship visualization\n- Codebase analysis tools\n- Performance metrics and statistics\n- Knowledge exploration\n"

  alias Brain.Metrics.Aggregator
  alias Brain.SystemStatus
  use ChatWeb, :live_view
  require Logger

  import ChatWeb.AppShell

  alias Brain.Code.{Pipeline, CodeGazetteer}
  @refresh_interval_ms 5000

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      :timer.send_interval(@refresh_interval_ms, self(), :refresh_status)
    end

    {:ok, socket}
  end

  @impl true
  def handle_params(_params, _uri, socket) do
    world_id = socket.assigns.current_world_id

    socket =
      socket
      |> assign(:world_id, world_id)
      |> assign(:code_status, load_code_status())
      |> assign(:symbols, [])
      |> assign(:search_query, "")
      |> assign(:selected_type, nil)
      |> assign(:selected_language, nil)
      |> assign(:code_input, "")
      |> assign(:selected_input_language, :elixir)
      |> assign(:analysis_result, nil)
      |> assign(:analyzing, false)
      |> assign(:active_tab, :browse)
      |> assign(:last_updated, DateTime.utc_now())
      |> assign(:selected_symbol, nil)
      |> assign(:symbol_relations, [])
      |> load_world_symbols()
      |> load_world_stats()
      |> load_metrics()

    {:noreply, socket}
  end

  @impl true
  def handle_info(:refresh_status, socket) do
    socket =
      socket
      |> assign(:code_status, load_code_status())
      |> load_world_stats()
      |> load_metrics()

    {:noreply, socket}
  end

  def handle_info({:world_context_changed, world_id}, socket) do
    socket =
      socket
      |> assign(:world_id, world_id)
      |> load_world_symbols()
      |> load_world_stats()

    {:noreply, socket}
  end

  @impl true
  def handle_event("switch_tab", %{"tab" => tab}, socket) do
    {:noreply, assign(socket, :active_tab, String.to_existing_atom(tab))}
  end

  def handle_event("switch_world", %{"world_id" => world_id}, socket) do
    socket =
      socket
      |> assign(:world_id, world_id)
      |> assign(:code_status, load_code_status())
      |> load_world_symbols()

    {:noreply, socket}
  end

  def handle_event("refresh_worlds", _params, socket) do
    {:noreply, socket}
  end

  def handle_event("search_symbols", %{"query" => query}, socket) do
    socket =
      socket
      |> assign(:search_query, query)
      |> load_world_symbols()

    {:noreply, socket}
  end

  def handle_event("filter_type", %{"type" => type}, socket) do
    type =
      if type == "" do
        nil
      else
        type
      end

    socket =
      socket
      |> assign(:selected_type, type)
      |> load_world_symbols()

    {:noreply, socket}
  end

  def handle_event("filter_language", %{"language" => language}, socket) do
    language =
      if language == "" do
        nil
      else
        String.to_existing_atom(language)
      end

    socket =
      socket
      |> assign(:selected_language, language)
      |> load_world_symbols()

    {:noreply, socket}
  end

  def handle_event("update_code_input", %{"code" => code}, socket) do
    {:noreply, assign(socket, :code_input, code)}
  end

  def handle_event("select_input_language", %{"language" => language}, socket) do
    {:noreply, assign(socket, :selected_input_language, String.to_existing_atom(language))}
  end

  def handle_event("analyze_code", _params, socket) do
    socket = assign(socket, :analyzing, true)

    code = socket.assigns.code_input
    language = socket.assigns.selected_input_language
    world_id = socket.assigns.world_id

    result =
      case Pipeline.process(code, language, world_id: world_id, store: true) do
        {:ok, analysis} ->
          %{
            success: true,
            symbols: analysis.symbols,
            relations: analysis.relations,
            summary: analysis.summary,
            stats: analysis.stats,
            errors: analysis.errors
          }

        {:error, reason} ->
          %{success: false, error: inspect(reason)}
      end

    socket =
      socket
      |> assign(:analyzing, false)
      |> assign(:analysis_result, result)
      |> load_world_symbols()

    {:noreply, socket}
  end

  def handle_event("clear_analysis", _params, socket) do
    socket =
      socket
      |> assign(:code_input, "")
      |> assign(:analysis_result, nil)

    {:noreply, socket}
  end

  def handle_event("clear_world_symbols", _params, socket) do
    world_id = socket.assigns.world_id

    case CodeGazetteer.clear_world(world_id) do
      :ok ->
        socket =
          socket
          |> assign(:symbols, [])
          |> assign(:selected_symbol, nil)
          |> load_world_stats()
          |> put_flash(:info, "Cleared all symbols for world #{world_id}")

        {:noreply, socket}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to clear: #{inspect(reason)}")}
    end
  end

  def handle_event("select_symbol", %{"name" => name}, socket) do
    world_id = socket.assigns.world_id

    case CodeGazetteer.lookup_qualified(world_id, name) do
      {:ok, symbol} ->
        relations = load_symbol_relations(world_id, symbol)

        socket =
          socket
          |> assign(:selected_symbol, symbol)
          |> assign(:symbol_relations, relations)

        {:noreply, socket}

      :not_found ->
        {:noreply, put_flash(socket, :error, "Symbol not found")}
    end
  end

  def handle_event("close_symbol_detail", _params, socket) do
    socket =
      socket
      |> assign(:selected_symbol, nil)
      |> assign(:symbol_relations, [])

    {:noreply, socket}
  end

  def handle_event("view_file_symbols", %{"path" => path}, socket) do
    world_id = socket.assigns.world_id
    symbols = CodeGazetteer.list_by_file(world_id, path)

    socket =
      socket
      |> assign(:symbols, symbols)
      |> assign(:search_query, "")
      |> assign(:selected_type, nil)
      |> assign(:active_tab, :browse)

    {:noreply, socket}
  end

  defp load_code_status do
    SystemStatus.get_code_analysis_status()
  end

  defp load_world_symbols(socket) do
    world_id = socket.assigns.world_id
    query = socket.assigns.search_query
    type = socket.assigns.selected_type
    language = socket.assigns.selected_language

    symbols =
      if String.length(query) > 0 do
        CodeGazetteer.search(world_id, query, entity_type: type, language: language, limit: 100)
      else
        if type do
          CodeGazetteer.list_by_type(world_id, type)
          |> Enum.take(100)
        else
          CodeGazetteer.entity_types()
          |> Enum.flat_map(fn t ->
            CodeGazetteer.list_by_type(world_id, t) |> Enum.take(10)
          end)
          |> Enum.take(100)
        end
      end

    assign(socket, :symbols, symbols)
  end

  defp load_world_stats(socket) do
    world_id = socket.assigns.world_id

    {type_breakdown, all_symbols} =
      CodeGazetteer.entity_types()
      |> Enum.reduce({[], []}, fn type, {types_acc, symbols_acc} ->
        symbols = CodeGazetteer.list_by_type(world_id, type)
        count = length(symbols)

        if count > 0 do
          {[{type, count} | types_acc], symbols ++ symbols_acc}
        else
          {types_acc, symbols_acc}
        end
      end)

    type_breakdown = Enum.sort_by(type_breakdown, fn {_, count} -> -count end)

    language_breakdown =
      all_symbols
      |> Enum.group_by(& &1.language)
      |> Enum.reject(fn {lang, _} -> is_nil(lang) end)
      |> Enum.map(fn {lang, symbols} -> {lang, length(symbols)} end)
      |> Enum.sort_by(fn {_, count} -> -count end)

    files =
      all_symbols
      |> Enum.map(& &1.file_path)
      |> Enum.reject(&is_nil/1)
      |> Enum.uniq()
      |> Enum.sort()

    total_symbols = length(all_symbols)
    total_relations = CodeGazetteer.stats(world_id) |> Map.get(:relations, 0)

    computed_stats = %{
      symbols: total_symbols,
      relations: total_relations,
      files: length(files),
      languages: length(language_breakdown)
    }

    socket
    |> assign(:world_stats, computed_stats)
    |> assign(:type_breakdown, type_breakdown)
    |> assign(:language_breakdown, language_breakdown)
    |> assign(:analyzed_files, files)
  end

  defp load_metrics(socket) do
    metrics = Aggregator.get_metrics()

    code_metrics = %{
      pipeline: Map.get(metrics, :code_pipeline, %{}),
      parse: Map.get(metrics, :code_parse, %{}),
      extract: Map.get(metrics, :code_extract, %{}),
      gazetteer_lookup: Map.get(metrics, :code_gazetteer_lookup, %{})
    }

    assign(socket, :code_metrics, code_metrics)
  end

  defp load_symbol_relations(world_id, symbol) do
    qualified_name = symbol.qualified_name
    relation_types = [:calls, :called_by, :extends, :implements, :imports, :uses]

    Enum.flat_map(relation_types, fn rel_type ->
      case CodeGazetteer.get_relations(world_id, qualified_name, rel_type) do
        targets when is_list(targets) ->
          Enum.map(targets, fn target -> %{type: rel_type, target: target} end)

        _ ->
          []
      end
    end)
  end

  def supported_languages do
    [:elixir, :python, :ruby, :go, :java, :c, :cpp, :csharp, :php]
  end

  def entity_types do
    CodeGazetteer.entity_types()
  end

  def language_name(:c) do
    "C"
  end

  def language_name(:cpp) do
    "C++"
  end

  def language_name(:csharp) do
    "C#"
  end

  def language_name(:php) do
    "PHP"
  end

  def language_name(lang) do
    lang |> to_string() |> String.capitalize()
  end

  def entity_type_label("code.function") do
    "Function"
  end

  def entity_type_label("code.class") do
    "Class"
  end

  def entity_type_label("code.method") do
    "Method"
  end

  def entity_type_label("code.variable") do
    "Variable"
  end

  def entity_type_label("code.constant") do
    "Constant"
  end

  def entity_type_label("code.type") do
    "Type"
  end

  def entity_type_label("code.interface") do
    "Interface"
  end

  def entity_type_label("code.enum") do
    "Enum"
  end

  def entity_type_label("code.namespace") do
    "Namespace"
  end

  def entity_type_label("code.import") do
    "Import"
  end

  def entity_type_label("code.keyword") do
    "Keyword"
  end

  def entity_type_label("code.parameter") do
    "Parameter"
  end

  def entity_type_label("code.field") do
    "Field"
  end

  def entity_type_label("code.macro") do
    "Macro"
  end

  def entity_type_label(type) do
    type |> String.replace("code.", "") |> String.capitalize()
  end

  def entity_type_icon("code.function") do
    "hero-code-bracket"
  end

  def entity_type_icon("code.class") do
    "hero-cube"
  end

  def entity_type_icon("code.method") do
    "hero-arrow-right-circle"
  end

  def entity_type_icon("code.variable") do
    "hero-variable"
  end

  def entity_type_icon("code.constant") do
    "hero-hashtag"
  end

  def entity_type_icon("code.type") do
    "hero-tag"
  end

  def entity_type_icon("code.interface") do
    "hero-puzzle-piece"
  end

  def entity_type_icon("code.enum") do
    "hero-list-bullet"
  end

  def entity_type_icon("code.namespace") do
    "hero-folder"
  end

  def entity_type_icon("code.import") do
    "hero-arrow-down-tray"
  end

  def entity_type_icon("code.keyword") do
    "hero-key"
  end

  def entity_type_icon("code.parameter") do
    "hero-arrow-right"
  end

  def entity_type_icon("code.field") do
    "hero-rectangle-stack"
  end

  def entity_type_icon("code.macro") do
    "hero-bolt"
  end

  def entity_type_icon(_) do
    "hero-code-bracket"
  end

  def entity_type_color("code.function") do
    "text-blue-500"
  end

  def entity_type_color("code.class") do
    "text-purple-500"
  end

  def entity_type_color("code.method") do
    "text-blue-400"
  end

  def entity_type_color("code.variable") do
    "text-green-500"
  end

  def entity_type_color("code.constant") do
    "text-amber-500"
  end

  def entity_type_color("code.type") do
    "text-cyan-500"
  end

  def entity_type_color("code.interface") do
    "text-violet-500"
  end

  def entity_type_color("code.enum") do
    "text-orange-500"
  end

  def entity_type_color("code.namespace") do
    "text-rose-500"
  end

  def entity_type_color("code.import") do
    "text-gray-500"
  end

  def entity_type_color("code.keyword") do
    "text-pink-500"
  end

  def entity_type_color(_) do
    "text-base-content"
  end

  def entity_type_bg("code.function") do
    "bg-blue-500/10"
  end

  def entity_type_bg("code.class") do
    "bg-purple-500/10"
  end

  def entity_type_bg("code.method") do
    "bg-blue-400/10"
  end

  def entity_type_bg("code.variable") do
    "bg-green-500/10"
  end

  def entity_type_bg("code.constant") do
    "bg-amber-500/10"
  end

  def entity_type_bg("code.type") do
    "bg-cyan-500/10"
  end

  def entity_type_bg("code.interface") do
    "bg-violet-500/10"
  end

  def entity_type_bg("code.enum") do
    "bg-orange-500/10"
  end

  def entity_type_bg("code.namespace") do
    "bg-rose-500/10"
  end

  def entity_type_bg("code.import") do
    "bg-gray-500/10"
  end

  def entity_type_bg(_) do
    "bg-base-200"
  end

  def relation_label(:calls) do
    "Calls"
  end

  def relation_label(:called_by) do
    "Called by"
  end

  def relation_label(:extends) do
    "Extends"
  end

  def relation_label(:implements) do
    "Implements"
  end

  def relation_label(:imports) do
    "Imports"
  end

  def relation_label(:uses) do
    "Uses"
  end

  def relation_label(rel) do
    rel |> to_string() |> String.replace("_", " ") |> String.capitalize()
  end

  def relation_icon(:calls) do
    "hero-arrow-right"
  end

  def relation_icon(:called_by) do
    "hero-arrow-left"
  end

  def relation_icon(:extends) do
    "hero-arrow-up"
  end

  def relation_icon(:implements) do
    "hero-puzzle-piece"
  end

  def relation_icon(:imports) do
    "hero-arrow-down-tray"
  end

  def relation_icon(:uses) do
    "hero-link"
  end

  def relation_icon(_) do
    "hero-arrow-right"
  end

  def format_metric_value(nil) do
    "-"
  end

  def format_metric_value(%{avg_ms: avg}) when is_number(avg) do
    "#{Float.round(avg * 1.0, 1)}ms"
  end

  def format_metric_value(_) do
    "-"
  end

  def format_metric_count(nil) do
    "0"
  end

  def format_metric_count(%{count: count}) do
    Integer.to_string(count)
  end

  def format_metric_count(_) do
    "0"
  end

  def format_metric_rate(nil) do
    "-"
  end

  def format_metric_rate(%{rate_per_minute: rate}) when is_number(rate) do
    "#{Float.round(rate * 1.0, 1)}/min"
  end

  def format_metric_rate(_) do
    "-"
  end

  def truncate_path(nil) do
    "-"
  end

  def truncate_path(path) when byte_size(path) > 50 do
    "..." <> String.slice(path, -47, 47)
  end

  def truncate_path(path) do
    path
  end
end