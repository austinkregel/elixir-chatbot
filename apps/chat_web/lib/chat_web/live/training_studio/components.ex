defmodule ChatWeb.TrainingStudio.Components do
  @moduledoc """
  Shared components for the Training Data Studio.
  """

  use Phoenix.Component

  import ChatWeb.UI, only: [icon_btn: 1, execute_confirm: 1]

  alias Phoenix.LiveView.JS

  @doc """
  One record of a training-data source as a table row. When `editable`, the
  row ends in ghost Edit and Delete icon buttons. Delete opens an
  `execute_confirm/1` panel in a row of its own below this one, naming the
  source file and the record; the page holds which panel is open
  (`confirm_open`) and any failure (`confirm_error`).
  """
  attr :record, :map, required: true
  attr :kind, :atom, required: true
  attr :index, :integer,
    required: true,
    doc:
      "the record's position in the unfiltered source file: what Edit and Delete act on, " <>
        "and, plus one, the record number the row and its delete confirmation show"

  attr :editable, :boolean, default: false
  attr :source_path, :string, default: nil, doc: "the source file, as the delete confirmation names it; required when editable"
  attr :confirm_open, :boolean, default: false
  attr :confirm_error, :string, default: nil

  def record_row(assigns) do
    if assigns.editable and assigns.source_path in [nil, ""] do
      raise ArgumentError,
            "ChatWeb.TrainingStudio.Components.record_row/1: an editable row needs source_path, " <>
              "the file its delete confirmation names."
    end

    ~H"""
    <tr class="even:bg-surface-sunk border-b border-border text-body-dense text-ink">
      <%= case @kind do %>
        <% :intent_example -> %>
          <td class="h-row-compact px-space-sm text-ref max-w-[200px] truncate" title={get_in_rec(@record, "intent") || get_in_rec(@record, "speech_act") || get_in_rec(@record, "sentiment")}>
            {get_in_rec(@record, "intent") || get_in_rec(@record, "speech_act") || get_in_rec(@record, "sentiment") || "—"}
          </td>
          <td class="h-row-compact px-space-sm max-w-md truncate" title={get_in_rec(@record, "text")}>
            {get_in_rec(@record, "text") || "—"}
          </td>
          <td class="h-row-compact px-space-sm text-caption text-ink-muted">
            {extra_fields(@record, ~w(intent text speech_act sentiment))}
          </td>

        <% :text_classifier_row -> %>
          <td class="h-row-compact px-space-sm text-ref max-w-[200px] truncate" title={get_in_rec(@record, "label")}>
            {get_in_rec(@record, "label") || "—"}
          </td>
          <td class="h-row-compact px-space-sm max-w-md truncate" title={get_in_rec(@record, "text")}>
            {get_in_rec(@record, "text") || "—"}
          </td>
          <td class="h-row-compact px-space-sm text-value text-ink-muted">
            {@index + 1}
          </td>

        <% :fv_classifier_row -> %>
          <td class="h-row-compact px-space-sm text-ref max-w-[200px] truncate" title={get_in_rec(@record, "label")}>
            {get_in_rec(@record, "label") || "—"}
          </td>
          <td class="h-row-compact px-space-sm text-caption text-ink-muted">
            <% fv = get_in_rec(@record, "feature_vector") %>
            <%= if is_list(fv) do %>
              {length(fv)}-dim vector
            <% else %>
              —
            <% end %>
          </td>
          <td class="h-row-compact px-space-sm text-value text-ink-muted">
            {@index + 1}
          </td>

        <% :kg_negative -> %>
          <td class="h-row-compact px-space-sm text-ref">{get_in_rec(@record, "head") || "—"}</td>
          <td class="h-row-compact px-space-sm text-ref">{get_in_rec(@record, "relation") || "—"}</td>
          <td class="h-row-compact px-space-sm text-ref">{get_in_rec(@record, "tail") || "—"}</td>

        <% :registry_entry -> %>
          <% {key, val} = registry_kv(@record) %>
          <td class="h-row-compact px-space-sm text-ref max-w-[200px] truncate" title={key}>{key}</td>
          <td class="h-row-compact px-space-sm text-caption">{Map.get(val, "domain", "—")}</td>
          <td class="h-row-compact px-space-sm text-caption">{Map.get(val, "category", "—")}</td>
          <td class="h-row-compact px-space-sm text-caption">{Map.get(val, "speech_act", "—")}</td>
          <td class="h-row-compact px-space-sm text-term">{inspect(Map.get(val, "required", []))}</td>

        <% :speech_act_map_entry -> %>
          <% {key, val} = registry_kv(@record) %>
          <td class="h-row-compact px-space-sm text-ref">{key}</td>
          <td class="h-row-compact px-space-sm text-ref">{val}</td>

        <% :gazetteer_entry -> %>
          <td class="h-row-compact px-space-sm text-ref max-w-[200px] truncate" title={gazetteer_value(@record)}>
            {gazetteer_value(@record)}
          </td>
          <td class="h-row-compact px-space-sm text-caption max-w-md truncate" title={gazetteer_synonyms_text(@record)}>
            {gazetteer_synonyms_text(@record)}
          </td>
          <td class="h-row-compact px-space-sm text-value text-ink-muted">
            {gazetteer_synonym_count(@record)}
          </td>

        <% :csv_row -> %>
          <td class="h-row-compact px-space-sm text-term max-w-lg truncate">
            {get_in_rec(@record, "line") || inspect(@record)}
          </td>

        <% _ -> %>
          <td class="h-row-compact px-space-sm text-term max-w-lg truncate">
            {inspect(@record) |> String.slice(0, 200)}
          </td>
      <% end %>
      <%= if @editable do %>
        <td class="h-row-compact px-space-sm whitespace-nowrap">
          <div class="flex items-center justify-end gap-space-xs">
            <.icon_btn
              id={"record-#{@index}-edit"}
              type="button"
              size={:sm}
              title="Edit record"
              phx-click="edit_record"
              phx-value-index={@index}
            >
              <span class="hero-pencil-square-micro size-4" aria-hidden="true" />
            </.icon_btn>
            <.icon_btn
              id={"record-#{@index}-delete"}
              type="button"
              size={:sm}
              title="Delete record"
              phx-click="open_confirm"
              phx-value-id={"confirm-delete-record-#{@index}"}
            >
              <span class="hero-trash-micro size-4" aria-hidden="true" />
            </.icon_btn>
          </div>
        </td>
      <% end %>
    </tr>
    <tr :if={@editable and @confirm_open} class="border-b border-border">
      <td colspan="10" class="px-space-sm py-space-sm">
        <.execute_confirm
          id={"confirm-delete-record-#{@index}"}
          open={@confirm_open}
          reach={:local}
          removes
          verb="Delete record"
          target={"#{@source_path} · record #{@index + 1}"}
          consequence={"Rewrites #{@source_path} on this node without record #{@index + 1}. The revision log keeps only content hashes, so the record cannot be restored from here."}
          on_confirm={JS.push("delete_record", value: %{index: @index})}
          on_cancel="close_confirm"
          trigger_id={"record-#{@index}-delete"}
          error={@confirm_error}
          class="ml-auto"
        />
      </td>
    </tr>
    """
  end

  def table_headers(assigns) do
    ~H"""
    <tr class="text-label text-ink-muted">
      <%= case @kind do %>
        <% :intent_example -> %>
          <th class="h-row-compact px-space-sm text-left">Label</th>
          <th class="h-row-compact px-space-sm text-left">Text</th>
          <th class="h-row-compact px-space-sm text-left">Extra</th>

        <% :text_classifier_row -> %>
          <th class="h-row-compact px-space-sm text-left">Label</th>
          <th class="h-row-compact px-space-sm text-left">Text</th>
          <th class="h-row-compact px-space-sm text-left">#</th>

        <% :fv_classifier_row -> %>
          <th class="h-row-compact px-space-sm text-left">Label</th>
          <th class="h-row-compact px-space-sm text-left">Feature Vector</th>
          <th class="h-row-compact px-space-sm text-left">#</th>

        <% :kg_negative -> %>
          <th class="h-row-compact px-space-sm text-left">Head</th>
          <th class="h-row-compact px-space-sm text-left">Relation</th>
          <th class="h-row-compact px-space-sm text-left">Tail</th>

        <% :registry_entry -> %>
          <th class="h-row-compact px-space-sm text-left">Intent</th>
          <th class="h-row-compact px-space-sm text-left">Domain</th>
          <th class="h-row-compact px-space-sm text-left">Category</th>
          <th class="h-row-compact px-space-sm text-left">Speech Act</th>
          <th class="h-row-compact px-space-sm text-left">Required Slots</th>

        <% :speech_act_map_entry -> %>
          <th class="h-row-compact px-space-sm text-left">Speech Act</th>
          <th class="h-row-compact px-space-sm text-left">Canonical Intent</th>

        <% :gazetteer_entry -> %>
          <th class="h-row-compact px-space-sm text-left">Value</th>
          <th class="h-row-compact px-space-sm text-left">Synonyms</th>
          <th class="h-row-compact px-space-sm text-left">#</th>

        <% _ -> %>
          <th class="h-row-compact px-space-sm text-left">Data</th>
      <% end %>
    </tr>
    """
  end

  # A strategy is a choice the system made, not a fault, and none asks the
  # reader to act, so every strategy is ink and its name says which it is.
  @strategy_classes %{
    can_respond: "text-ink",
    needs_clarification: "text-ink",
    hedged_response: "text-ink",
    partial_response_with_clarification: "text-ink",
    cannot_respond: "text-ink",
    defer_to_user: "text-ink"
  }

  @doc """
  The text color for a traced response strategy: ink for every one. The
  strategy's name is always shown beside it. A strategy with no treatment
  raises.
  """
  def strategy_class(strategy) do
    case Map.fetch(@strategy_classes, strategy) do
      {:ok, class} ->
        class

      :error ->
        raise ArgumentError,
              "ChatWeb.TrainingStudio.Components.strategy_class/1: no treatment for " <>
                "strategy #{inspect(strategy)}. The strategies are " <>
                "#{inspect(Map.keys(@strategy_classes))}."
    end
  end

  defp get_in_rec(rec, key) when is_map(rec), do: Map.get(rec, key)
  defp get_in_rec(_, _), do: nil

  defp extra_fields(rec, exclude) when is_map(rec) do
    extras =
      rec
      |> Map.drop(exclude)
      |> Map.keys()
      |> Enum.reject(&(&1 == "feature_vector"))

    if extras == [], do: "", else: Enum.join(extras, ", ")
  end

  defp extra_fields(_, _), do: ""

  defp registry_kv({key, val}), do: {key, val}
  defp registry_kv(other), do: {"?", other}

  defp gazetteer_value(rec) when is_map(rec) do
    Map.get(rec, "value") || Map.get(rec, "name") || Map.get(rec, "entry") || "—"
  end

  defp gazetteer_value(_), do: "—"

  defp gazetteer_synonyms_text(rec) when is_map(rec) do
    case Map.get(rec, "synonyms", []) do
      syns when is_list(syns) and syns != [] -> Enum.join(syns, ", ")
      _ -> "—"
    end
  end

  defp gazetteer_synonyms_text(_), do: "—"

  defp gazetteer_synonym_count(rec) when is_map(rec) do
    case Map.get(rec, "synonyms", []) do
      syns when is_list(syns) -> length(syns)
      _ -> 0
    end
  end

  defp gazetteer_synonym_count(_), do: 0
end
