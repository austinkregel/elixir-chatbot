defmodule ChatWeb.TrainingStudioLive do
  @moduledoc """
  Training Data Studio — unified editor and diagnostic surface for all
  training-data source files.
  """

  use ChatWeb, :live_view
  require Logger

  import ChatWeb.AppShell

  alias Brain.ML.TrainingData.{Catalog, Diagnostics, RevisionLog, SourceDescriptors}
  alias Brain.ML.TrainingServer

  @page_size 50

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      Phoenix.PubSub.subscribe(Brain.PubSub, "training_data:updates")
      Phoenix.PubSub.subscribe(Brain.PubSub, "training:progress")
    end

    {:ok, socket}
  end

  @impl true
  def handle_params(params, _uri, socket) do
    active_tab = Map.get(params, "tab", "browse")
    source_id = parse_source_id(Map.get(params, "source"))
    page = parse_int(Map.get(params, "page"), 1)
    filter = Map.get(params, "filter", "")

    socket =
      socket
      |> assign(:active_tab, active_tab)
      |> assign(:selected_source, source_id)
      |> assign(:page, page)
      |> assign(:filter, filter)
      |> assign(:show_add_form, false)
      |> assign(:editing_index, nil)
      |> assign(:sources_by_category, Catalog.list_sources_by_category())
      |> assign(:summary_stats, Diagnostics.summary_stats())
      |> load_tab_data(active_tab, source_id, page, filter)

    {:noreply, socket}
  end

  @impl true
  def handle_event("select_source", %{"source" => source_id_str}, socket) do
    {:noreply,
     push_patch(socket,
       to: ~p"/training-studio?#{%{tab: socket.assigns.active_tab, source: source_id_str, page: 1, filter: ""}}"
     )}
  end

  def handle_event("switch_tab", %{"tab" => tab}, socket) do
    params = %{tab: tab}

    params =
      if socket.assigns.selected_source do
        Map.put(params, :source, socket.assigns.selected_source)
      else
        params
      end

    {:noreply, push_patch(socket, to: ~p"/training-studio?#{params}")}
  end

  def handle_event("filter", %{"filter" => filter}, socket) do
    params = %{
      tab: socket.assigns.active_tab,
      filter: filter,
      page: 1
    }

    params =
      if socket.assigns.selected_source do
        Map.put(params, :source, socket.assigns.selected_source)
      else
        params
      end

    {:noreply, push_patch(socket, to: ~p"/training-studio?#{params}")}
  end

  def handle_event("page", %{"page" => page_str}, socket) do
    page = parse_int(page_str, 1)

    params = %{
      tab: socket.assigns.active_tab,
      page: page,
      filter: socket.assigns.filter
    }

    params =
      if socket.assigns.selected_source do
        Map.put(params, :source, socket.assigns.selected_source)
      else
        params
      end

    {:noreply, push_patch(socket, to: ~p"/training-studio?#{params}")}
  end

  def handle_event("lookup", %{"query" => query}, socket) do
    results = Diagnostics.cross_source_lookup(query)
    {:noreply, assign(socket, :lookup_results, results) |> assign(:lookup_query, query)}
  end

  def handle_event("show_add_form", _params, socket) do
    {:noreply, assign(socket, :show_add_form, true)}
  end

  def handle_event("cancel_add", _params, socket) do
    {:noreply, assign(socket, :show_add_form, false)}
  end

  def handle_event("add_record", %{"record" => record_params}, socket) do
    source_id = socket.assigns.selected_source

    if source_id do
      record = build_record_from_params(record_params, socket.assigns.source_desc)

      case Catalog.add_record(source_id, record) do
        :ok ->
          {:noreply,
           socket
           |> assign(:show_add_form, false)
           |> put_flash(:info, "Record added")
           |> push_patch(to: current_path(socket))}

        {:error, reason} ->
          {:noreply, put_flash(socket, :error, "Failed to add: #{inspect(reason)}")}
      end
    else
      {:noreply, socket}
    end
  end

  def handle_event("delete_record", %{"index" => index_str}, socket) do
    source_id = socket.assigns.selected_source
    index = parse_int(index_str, -1)

    if source_id && index >= 0 do
      case Catalog.delete_record(source_id, index) do
        :ok ->
          {:noreply,
           socket
           |> put_flash(:info, "Record deleted")
           |> push_patch(to: current_path(socket))}

        {:error, reason} ->
          {:noreply, put_flash(socket, :error, "Failed to delete: #{inspect(reason)}")}
      end
    else
      {:noreply, socket}
    end
  end

  def handle_event("edit_record", %{"index" => index_str}, socket) do
    index = parse_int(index_str, -1)
    {:noreply, assign(socket, :editing_index, index)}
  end

  def handle_event("cancel_edit", _params, socket) do
    {:noreply, assign(socket, :editing_index, nil)}
  end

  def handle_event("save_edit", %{"index" => index_str, "record" => record_params}, socket) do
    source_id = socket.assigns.selected_source
    index = parse_int(index_str, -1)

    if source_id && index >= 0 do
      record = build_record_from_params(record_params, socket.assigns.source_desc)

      case Catalog.update_record(source_id, index, record) do
        :ok ->
          {:noreply,
           socket
           |> assign(:editing_index, nil)
           |> put_flash(:info, "Record updated")
           |> push_patch(to: current_path(socket))}

        {:error, reason} ->
          {:noreply, put_flash(socket, :error, "Failed to save: #{inspect(reason)}")}
      end
    else
      {:noreply, socket}
    end
  end

  def handle_event("trace", %{"text" => text}, socket) when is_binary(text) and text != "" do
    case Diagnostics.trace_prediction(text) do
      {:ok, result} ->
        {:noreply,
         socket
         |> assign(:trace_result, result)
         |> assign(:trace_text, text)}

      {:error, reason} ->
        {:noreply,
         socket
         |> assign(:trace_result, nil)
         |> assign(:trace_text, text)
         |> put_flash(:error, "Trace failed: #{inspect(reason)}")}
    end
  end

  def handle_event("trace", _params, socket), do: {:noreply, socket}

  def handle_event("run_pipeline", %{"action" => action} = params, socket) do
    config =
      case action do
        "evaluate" -> [task: Map.get(params, "task", "intent")]
        _ -> []
      end

    action_atom = String.to_existing_atom(action)

    case TrainingServer.start_training(action_atom, config) do
      {:ok, _} ->
        {:noreply,
         socket
         |> assign(:training_status, TrainingServer.get_status())
         |> put_flash(:info, "Started: #{action}")}

      {:error, {:already_training, current}} ->
        {:noreply, put_flash(socket, :error, "Already training: #{current}")}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed: #{inspect(reason)}")}
    end
  end

  @impl true
  def handle_info({:training_started, model_type, _started_at}, socket) do
    {:noreply,
     socket
     |> assign(:training_status, TrainingServer.get_status())
     |> put_flash(:info, "Training started: #{model_type}")}
  end

  def handle_info({:training_complete, model_type, result}, socket) do
    flash =
      case result do
        {:ok, _} -> {:info, "Training complete: #{model_type}"}
        {:error, reason} -> {:error, "Training failed: #{model_type} — #{inspect(reason)}"}
      end

    {:noreply,
     socket
     |> assign(:training_status, :idle)
     |> put_flash(elem(flash, 0), elem(flash, 1))
     |> assign(:sources_by_category, Catalog.list_sources_by_category())
     |> assign(:summary_stats, Diagnostics.summary_stats())}
  end

  def handle_info({:training_cancelled, model_type}, socket) do
    {:noreply,
     socket
     |> assign(:training_status, :idle)
     |> put_flash(:info, "Training cancelled: #{model_type}")}
  end

  def handle_info({:model_reloaded, _model_type, _result}, socket) do
    {:noreply, socket}
  end

  def handle_info({:training_data, :source_updated, _source_id}, socket) do
    socket =
      socket
      |> assign(:sources_by_category, Catalog.list_sources_by_category())
      |> assign(:summary_stats, Diagnostics.summary_stats())
      |> load_tab_data(
        socket.assigns.active_tab,
        socket.assigns.selected_source,
        socket.assigns.page,
        socket.assigns.filter
      )

    {:noreply, socket}
  end

  def handle_info(_, socket), do: {:noreply, socket}

  # ── Tab data loading ─────────────────────────────────────────────────

  defp load_tab_data(socket, "browse", source_id, page, filter) when not is_nil(source_id) do
    offset = (page - 1) * @page_size
    desc = SourceDescriptors.get(source_id)

    case Catalog.read_source_page(source_id, offset, @page_size, filter: filter) do
      {:ok, records, total} ->
        total_pages = max(1, ceil(total / @page_size))

        socket
        |> assign(:records, records)
        |> assign(:total_records, total)
        |> assign(:total_pages, total_pages)
        |> assign(:source_desc, desc)

      {:error, reason} ->
        socket
        |> assign(:records, [])
        |> assign(:total_records, 0)
        |> assign(:total_pages, 1)
        |> assign(:source_desc, desc)
        |> put_flash(:error, "Failed to load source: #{inspect(reason)}")
    end
  end

  defp load_tab_data(socket, "browse", _source_id, _page, _filter) do
    socket
    |> assign(:records, [])
    |> assign(:total_records, 0)
    |> assign(:total_pages, 1)
    |> assign(:source_desc, nil)
  end

  defp load_tab_data(socket, "drift", source_id, _page, _filter) when not is_nil(source_id) do
    desc = SourceDescriptors.get(source_id)

    case Diagnostics.skew_report(source_id) do
      {:ok, rows} ->
        socket
        |> assign(:skew_rows, rows)
        |> assign(:source_desc, desc)

      {:error, reason} ->
        socket
        |> assign(:skew_rows, [])
        |> assign(:source_desc, desc)
        |> put_flash(:info, "Skew report: #{inspect(reason)}")
    end
  end

  defp load_tab_data(socket, "drift", _source_id, _page, _filter) do
    all_skew =
      SourceDescriptors.all()
      |> Enum.filter(&(&1.tag in [:authoring, :build_artifact]))
      |> Enum.flat_map(fn desc ->
        case Diagnostics.skew_report(desc.id) do
          {:ok, rows} ->
            warnings = Enum.filter(rows, & &1.skew_warning)
            if warnings != [], do: [{desc, warnings}], else: []

          _ ->
            []
        end
      end)

    socket
    |> assign(:skew_rows, [])
    |> assign(:source_desc, nil)
    |> assign(:all_skew_warnings, all_skew)
  end

  defp load_tab_data(socket, "orphans", _source_id, _page, _filter) do
    case Diagnostics.orphan_report() do
      {:ok, report} ->
        assign(socket, :orphan_report, report)

      {:error, _} ->
        assign(socket, :orphan_report, %{
          gold_without_registry: [],
          registry_without_gold: [],
          gold_intent_count: 0,
          registry_intent_count: 0
        })
    end
  end

  defp load_tab_data(socket, "trace", _source_id, _page, _filter) do
    socket
    |> assign_new(:trace_result, fn -> nil end)
    |> assign_new(:trace_text, fn -> "" end)
  end

  defp load_tab_data(socket, "pipeline", _source_id, _page, _filter) do
    socket
    |> assign(:training_status, TrainingServer.get_status())
    |> assign(:recent_revisions, RevisionLog.recent(20))
  end

  defp load_tab_data(socket, _tab, _source_id, _page, _filter) do
    socket
    |> assign_new(:records, fn -> [] end)
    |> assign_new(:total_records, fn -> 0 end)
    |> assign_new(:total_pages, fn -> 1 end)
    |> assign_new(:source_desc, fn -> nil end)
    |> assign_new(:skew_rows, fn -> [] end)
    |> assign_new(:all_skew_warnings, fn -> [] end)
    |> assign_new(:orphan_report, fn -> %{} end)
    |> assign_new(:lookup_results, fn -> [] end)
    |> assign_new(:lookup_query, fn -> "" end)
  end

  # ── Helpers ──────────────────────────────────────────────────────────

  defp parse_source_id(nil), do: nil
  defp parse_source_id(""), do: nil

  defp parse_source_id(str) when is_binary(str) do
    try do
      String.to_existing_atom(str)
    rescue
      ArgumentError -> String.to_atom(str)
    end
  end

  defp parse_int(nil, default), do: default
  defp parse_int(str, default) when is_binary(str) do
    case Integer.parse(str) do
      {n, _} when n > 0 -> n
      _ -> default
    end
  end

  defp parse_int(n, _default) when is_integer(n) and n > 0, do: n
  defp parse_int(_, default), do: default

  defp visible_pages(current, total) do
    range_start = max(1, current - 2)
    range_end = min(total, current + 2)
    Enum.to_list(range_start..range_end)
  end

  defp current_path(socket) do
    params = %{
      tab: socket.assigns.active_tab,
      page: socket.assigns.page,
      filter: socket.assigns.filter
    }

    params =
      if socket.assigns.selected_source do
        Map.put(params, :source, socket.assigns.selected_source)
      else
        params
      end

    ~p"/training-studio?#{params}"
  end

  defp build_record_from_params(params, %{record_kind: :intent_example}) do
    %{
      "text" => Map.get(params, "text", ""),
      "intent" => Map.get(params, "intent", Map.get(params, "label", ""))
    }
  end

  defp build_record_from_params(params, %{record_kind: :text_classifier_row}) do
    %{
      "text" => Map.get(params, "text", ""),
      "label" => Map.get(params, "label", "")
    }
  end

  defp build_record_from_params(params, %{record_kind: :kg_negative}) do
    %{
      "head" => Map.get(params, "head", ""),
      "relation" => Map.get(params, "relation", ""),
      "tail" => Map.get(params, "tail", "")
    }
  end

  defp build_record_from_params(params, %{record_kind: :gazetteer_entry}) do
    synonyms =
      case Map.get(params, "synonyms", "") do
        "" -> []
        s -> String.split(s, ",") |> Enum.map(&String.trim/1) |> Enum.reject(&(&1 == ""))
      end

    %{
      "value" => Map.get(params, "value", ""),
      "synonyms" => synonyms
    }
  end

  defp build_record_from_params(params, _) do
    params
  end

  defp format_training_time(%DateTime{} = dt) do
    Calendar.strftime(dt, "%H:%M:%S")
  end

  defp format_training_time(_), do: "?"

  defp source_editable?(assigns) do
    assigns[:source_desc] != nil and
      Brain.ML.TrainingData.SourceDescriptors.editable?(assigns.source_desc)
  end
end
