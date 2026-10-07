defmodule ChatWeb.TrainingStudio.Components do
  @moduledoc """
  Shared components for the Training Data Studio.
  """

  use Phoenix.Component

  import ChatWeb.UI, only: [btn: 1]

  attr :record, :map, required: true
  attr :kind, :atom, required: true
  attr :index, :integer, required: true
  attr :editable, :boolean, default: false

  def record_row(assigns) do
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
        <td class="h-row-compact px-space-sm text-right whitespace-nowrap">
          <.btn
            phx-click="edit_record"
            phx-value-index={@index}
            size={:xs}
            title="Edit"
          >
            Edit
          </.btn>
          <.btn
            phx-click="delete_record"
            phx-value-index={@index}
            size={:xs}
            title="Delete"
            data-confirm="Delete this record?"
          >
            Del
          </.btn>
        </td>
      <% end %>
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

  @strategy_classes %{
    can_respond: "text-ink",
    needs_clarification: "text-ochre",
    hedged_response: "text-red",
    partial_response_with_clarification: "text-ink",
    cannot_respond: "text-ink",
    defer_to_user: "text-ink"
  }

  @doc """
  The text color for a traced response strategy. The strategy's name is always
  shown beside it, so the color is never the only carrier of which one it is.
  A strategy with no treatment raises.
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
