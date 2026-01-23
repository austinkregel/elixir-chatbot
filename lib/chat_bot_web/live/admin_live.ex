defmodule ChatBotWeb.AdminLive do
  @moduledoc """
  Admin interface for managing gazetteer entries (locations, entities).
  """

  use ChatBotWeb, :live_view
  require Logger

  alias ChatBot.ML.Gazetteer

  @default_page_size 25
  @page_size_options [25, 50, 100]

  @impl true
  def mount(_params, _session, socket) do
    # Get entity types and initial list
    entity_types = get_entity_types()
    selected_type = List.first(entity_types) || "location"
    all_entries = get_all_entries_for_type(selected_type)
    stats = Gazetteer.stats()

    socket =
      socket
      |> assign(:entity_types, entity_types)
      |> assign(:selected_type, selected_type)
      |> assign(:all_entries, all_entries)
      |> assign(:search_query, "")
      |> assign(:search_results, [])
      |> assign(:stats, stats)
      |> assign(:show_add_modal, false)
      |> assign(:new_entry_name, "")
      |> assign(:new_entry_type, selected_type)
      |> assign(:new_entry_metadata, %{})
      # Pagination
      |> assign(:page, 1)
      |> assign(:page_size, @default_page_size)
      |> assign(:page_size_options, @page_size_options)
      # Sorting
      |> assign(:sort_by, :name)
      |> assign(:sort_dir, :asc)
      |> apply_sorting_and_pagination()

    {:ok, socket}
  end

  @impl true
  def handle_event("select_type", %{"type" => type}, socket) do
    all_entries = get_all_entries_for_type(type)

    {:noreply,
     socket
     |> assign(:selected_type, type)
     |> assign(:all_entries, all_entries)
     |> assign(:new_entry_type, type)
     |> assign(:page, 1)
     |> apply_sorting_and_pagination()}
  end

  def handle_event("change_page", %{"page" => page}, socket) do
    page = String.to_integer(page)

    {:noreply,
     socket
     |> assign(:page, page)
     |> apply_sorting_and_pagination()}
  end

  def handle_event("change_page_size", %{"size" => size}, socket) do
    size = String.to_integer(size)

    {:noreply,
     socket
     |> assign(:page_size, size)
     |> assign(:page, 1)
     |> apply_sorting_and_pagination()}
  end

  def handle_event("sort", %{"column" => column}, socket) do
    column = String.to_existing_atom(column)
    current_sort = socket.assigns.sort_by
    current_dir = socket.assigns.sort_dir

    new_dir =
      if column == current_sort do
        if current_dir == :asc, do: :desc, else: :asc
      else
        :asc
      end

    {:noreply,
     socket
     |> assign(:sort_by, column)
     |> assign(:sort_dir, new_dir)
     |> assign(:page, 1)
     |> apply_sorting_and_pagination()}
  end

  def handle_event("search", %{"query" => query}, socket) do
    results =
      if String.length(query) >= 2 do
        Gazetteer.search(query)
      else
        []
      end

    {:noreply,
     socket
     |> assign(:search_query, query)
     |> assign(:search_results, results)}
  end

  def handle_event("clear_search", _params, socket) do
    {:noreply,
     socket
     |> assign(:search_query, "")
     |> assign(:search_results, [])}
  end

  def handle_event("open_add_modal", _params, socket) do
    {:noreply, assign(socket, :show_add_modal, true)}
  end

  def handle_event("close_add_modal", _params, socket) do
    {:noreply,
     socket
     |> assign(:show_add_modal, false)
     |> assign(:new_entry_name, "")}
  end

  def handle_event("update_new_entry", %{"name" => name, "type" => type}, socket) do
    {:noreply,
     socket
     |> assign(:new_entry_name, name)
     |> assign(:new_entry_type, type)}
  end

  def handle_event("add_entry", %{"name" => name, "type" => type}, socket) do
    name = String.trim(name)
    type = String.trim(type)

    if name != "" and type != "" do
      case Gazetteer.add_entry(name, type) do
        {:ok, _key} ->
          # Refresh the list
          all_entries = get_all_entries_for_type(socket.assigns.selected_type)
          entity_types = get_entity_types()
          stats = Gazetteer.stats()

          {:noreply,
           socket
           |> assign(:all_entries, all_entries)
           |> assign(:entity_types, entity_types)
           |> assign(:stats, stats)
           |> assign(:new_entry_name, "")
           |> assign(:show_add_modal, false)
           |> apply_sorting_and_pagination()
           |> put_flash(:info, "Added \"#{name}\" as #{type}")}

        {:error, {:duplicate, existing_type}} ->
          {:noreply,
           put_flash(
             socket,
             :error,
             "\"#{name}\" already exists as a #{existing_type} entity"
           )}

        {:error, reason} ->
          {:noreply, put_flash(socket, :error, "Failed to add entry: #{inspect(reason)}")}
      end
    else
      {:noreply, put_flash(socket, :error, "Name and type are required")}
    end
  end

  def handle_event("remove_entry", %{"name" => name}, socket) do
    case Gazetteer.remove_entry(name) do
      :ok ->
        all_entries = get_all_entries_for_type(socket.assigns.selected_type)
        stats = Gazetteer.stats()

        {:noreply,
         socket
         |> assign(:all_entries, all_entries)
         |> assign(:stats, stats)
         |> apply_sorting_and_pagination()
         |> put_flash(:info, "Removed \"#{name}\"")}

      {:error, :not_found} ->
        {:noreply, put_flash(socket, :error, "Entry not found")}
    end
  end

  def handle_event("refresh", _params, socket) do
    all_entries = get_all_entries_for_type(socket.assigns.selected_type)
    entity_types = get_entity_types()
    stats = Gazetteer.stats()

    {:noreply,
     socket
     |> assign(:all_entries, all_entries)
     |> assign(:entity_types, entity_types)
     |> assign(:stats, stats)
     |> apply_sorting_and_pagination()}
  end

  def handle_event("clear_type", _params, socket) do
    type = socket.assigns.selected_type

    case Gazetteer.clear_by_type(type) do
      {:ok, count} ->
        all_entries = get_all_entries_for_type(type)
        entity_types = get_entity_types()
        stats = Gazetteer.stats()

        {:noreply,
         socket
         |> assign(:all_entries, all_entries)
         |> assign(:entity_types, entity_types)
         |> assign(:stats, stats)
         |> apply_sorting_and_pagination()
         |> put_flash(:info, "Cleared #{count} #{type} entries")}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to clear: #{inspect(reason)}")}
    end
  end

  def handle_event("clear_admin", _params, socket) do
    case Gazetteer.clear_admin_entries() do
      {:ok, count} ->
        all_entries = get_all_entries_for_type(socket.assigns.selected_type)
        entity_types = get_entity_types()
        stats = Gazetteer.stats()

        {:noreply,
         socket
         |> assign(:all_entries, all_entries)
         |> assign(:entity_types, entity_types)
         |> assign(:stats, stats)
         |> apply_sorting_and_pagination()
         |> put_flash(:info, "Cleared #{count} admin-added entries")}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to clear: #{inspect(reason)}")}
    end
  end

  def handle_event("reload_data", _params, socket) do
    # Clear all and reload from data files
    Gazetteer.clear_all()
    Gazetteer.load_all()

    all_entries = get_all_entries_for_type(socket.assigns.selected_type)
    entity_types = get_entity_types()
    stats = Gazetteer.stats()

    {:noreply,
     socket
     |> assign(:all_entries, all_entries)
     |> assign(:entity_types, entity_types)
     |> assign(:stats, stats)
     |> apply_sorting_and_pagination()
     |> put_flash(:info, "Reloaded all data from files")}
  end

  # Private helpers

  defp get_entity_types do
    types = Gazetteer.list_types()

    if types == [] do
      ["location", "city", "device", "room", "person"]
    else
      types
    end
  end

  defp get_all_entries_for_type(type) do
    Gazetteer.list_by_type(type)
  end

  defp apply_sorting_and_pagination(socket) do
    all_entries = socket.assigns.all_entries
    sort_by = socket.assigns.sort_by
    sort_dir = socket.assigns.sort_dir
    page = socket.assigns.page
    page_size = socket.assigns.page_size

    # Sort entries
    sorted =
      Enum.sort_by(all_entries, fn {key, info} ->
        case sort_by do
          :name -> String.downcase(info[:original_name] || info[:value] || key)
          :source -> to_string(info[:source] || "data")
          _ -> key
        end
      end)

    sorted = if sort_dir == :desc, do: Enum.reverse(sorted), else: sorted

    # Calculate pagination
    total_entries = length(sorted)
    total_pages = max(1, ceil(total_entries / page_size))
    page = min(page, total_pages)

    # Get page slice
    start_idx = (page - 1) * page_size
    entries = Enum.slice(sorted, start_idx, page_size)

    socket
    |> assign(:entries, entries)
    |> assign(:total_entries, total_entries)
    |> assign(:total_pages, total_pages)
    |> assign(:page, page)
  end

  # Template helpers

  def icon_for_type(type) do
    case type do
      "location" -> "hero-map-pin"
      "city" -> "hero-building-office-2"
      "country" -> "hero-globe-americas"
      "region" -> "hero-map"
      "device" -> "hero-device-phone-mobile"
      "room" -> "hero-home"
      "person" -> "hero-user"
      "artist" -> "hero-musical-note"
      "music-artist" -> "hero-musical-note"
      "emoji" -> "hero-face-smile"
      _ -> "hero-tag"
    end
  end

  def source_badge_class(source) do
    case source do
      :admin -> "badge-primary"
      :cities -> "badge-info"
      :entities -> "badge-secondary"
      :artists -> "badge-accent"
      :emojis -> "badge-warning"
      _ -> "badge-ghost"
    end
  end

  def format_number(nil), do: "-"

  def format_number(num) when is_integer(num) do
    num
    |> Integer.to_string()
    |> String.graphemes()
    |> Enum.reverse()
    |> Enum.chunk_every(3)
    |> Enum.join(",")
    |> String.reverse()
  end

  def format_number(num) when is_binary(num) do
    case Integer.parse(num) do
      {int, _} -> format_number(int)
      :error -> num
    end
  end

  def format_number(num), do: "#{num}"

  def sort_indicator(column, sort_by, sort_dir) do
    if column == sort_by do
      if sort_dir == :asc, do: "hero-chevron-up", else: "hero-chevron-down"
    else
      nil
    end
  end

  def page_range(current_page, total_pages) do
    # Show up to 5 page numbers centered around current page
    cond do
      total_pages <= 5 ->
        1..total_pages

      current_page <= 3 ->
        1..min(5, total_pages)

      current_page >= total_pages - 2 ->
        max(1, total_pages - 4)..total_pages

      true ->
        (current_page - 2)..(current_page + 2)
    end
    |> Enum.to_list()
  end
end
