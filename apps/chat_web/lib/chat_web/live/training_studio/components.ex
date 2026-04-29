defmodule ChatWeb.TrainingStudio.Components do
  @moduledoc """
  Shared components for the Training Data Studio.
  """

  use Phoenix.Component

  attr :record, :map, required: true
  attr :kind, :atom, required: true
  attr :index, :integer, required: true
  attr :editable, :boolean, default: false

  def record_row(assigns) do
    ~H"""
    <tr class="hover:bg-base-200/50 border-b border-base-300/30 text-sm">
      <%= case @kind do %>
        <% :intent_example -> %>
          <td class="px-3 py-2 font-mono text-xs max-w-[200px] truncate" title={get_in_rec(@record, "intent") || get_in_rec(@record, "speech_act") || get_in_rec(@record, "sentiment")}>
            {get_in_rec(@record, "intent") || get_in_rec(@record, "speech_act") || get_in_rec(@record, "sentiment") || "—"}
          </td>
          <td class="px-3 py-2 max-w-md truncate" title={get_in_rec(@record, "text")}>
            {get_in_rec(@record, "text") || "—"}
          </td>
          <td class="px-3 py-2 text-base-content/50 text-xs">
            {extra_fields(@record, ~w(intent text speech_act sentiment))}
          </td>

        <% :text_classifier_row -> %>
          <td class="px-3 py-2 font-mono text-xs max-w-[200px] truncate" title={get_in_rec(@record, "label")}>
            {get_in_rec(@record, "label") || "—"}
          </td>
          <td class="px-3 py-2 max-w-md truncate" title={get_in_rec(@record, "text")}>
            {get_in_rec(@record, "text") || "—"}
          </td>
          <td class="px-3 py-2 text-base-content/50 text-xs">
            {@index + 1}
          </td>

        <% :fv_classifier_row -> %>
          <td class="px-3 py-2 font-mono text-xs max-w-[200px] truncate" title={get_in_rec(@record, "label")}>
            {get_in_rec(@record, "label") || "—"}
          </td>
          <td class="px-3 py-2 text-xs text-base-content/50">
            <% fv = get_in_rec(@record, "feature_vector") %>
            <%= if is_list(fv) do %>
              {length(fv)}-dim vector
            <% else %>
              —
            <% end %>
          </td>
          <td class="px-3 py-2 text-base-content/50 text-xs">
            {@index + 1}
          </td>

        <% :kg_negative -> %>
          <td class="px-3 py-2 font-mono text-xs">{get_in_rec(@record, "head") || "—"}</td>
          <td class="px-3 py-2 font-mono text-xs">{get_in_rec(@record, "relation") || "—"}</td>
          <td class="px-3 py-2 font-mono text-xs">{get_in_rec(@record, "tail") || "—"}</td>

        <% :registry_entry -> %>
          <% {key, val} = registry_kv(@record) %>
          <td class="px-3 py-2 font-mono text-xs max-w-[200px] truncate" title={key}>{key}</td>
          <td class="px-3 py-2 text-xs">{Map.get(val, "domain", "—")}</td>
          <td class="px-3 py-2 text-xs">{Map.get(val, "category", "—")}</td>
          <td class="px-3 py-2 text-xs">{Map.get(val, "speech_act", "—")}</td>
          <td class="px-3 py-2 text-xs font-mono">{inspect(Map.get(val, "required", []))}</td>

        <% :speech_act_map_entry -> %>
          <% {key, val} = registry_kv(@record) %>
          <td class="px-3 py-2 font-mono text-xs">{key}</td>
          <td class="px-3 py-2 font-mono text-xs">{val}</td>

        <% :gazetteer_entry -> %>
          <td class="px-3 py-2 font-mono text-xs max-w-[200px] truncate" title={gazetteer_value(@record)}>
            {gazetteer_value(@record)}
          </td>
          <td class="px-3 py-2 text-xs max-w-md truncate" title={gazetteer_synonyms_text(@record)}>
            {gazetteer_synonyms_text(@record)}
          </td>
          <td class="px-3 py-2 text-xs text-base-content/50 tabular-nums">
            {gazetteer_synonym_count(@record)}
          </td>

        <% :csv_row -> %>
          <td class="px-3 py-2 text-xs font-mono max-w-lg truncate">
            {get_in_rec(@record, "line") || inspect(@record)}
          </td>

        <% _ -> %>
          <td class="px-3 py-2 text-xs font-mono max-w-lg truncate">
            {inspect(@record) |> String.slice(0, 200)}
          </td>
      <% end %>
      <%= if @editable do %>
        <td class="px-3 py-1.5 text-right whitespace-nowrap">
          <button
            phx-click="edit_record"
            phx-value-index={@index}
            class="btn btn-xs btn-ghost text-primary"
            title="Edit"
          >
            Edit
          </button>
          <button
            phx-click="delete_record"
            phx-value-index={@index}
            class="btn btn-xs btn-ghost text-error"
            title="Delete"
            data-confirm="Delete this record?"
          >
            Del
          </button>
        </td>
      <% end %>
    </tr>
    """
  end

  def table_headers(assigns) do
    ~H"""
    <tr class="text-xs font-semibold text-base-content/60 uppercase tracking-wider">
      <%= case @kind do %>
        <% :intent_example -> %>
          <th class="px-3 py-2 text-left">Label</th>
          <th class="px-3 py-2 text-left">Text</th>
          <th class="px-3 py-2 text-left">Extra</th>

        <% :text_classifier_row -> %>
          <th class="px-3 py-2 text-left">Label</th>
          <th class="px-3 py-2 text-left">Text</th>
          <th class="px-3 py-2 text-left">#</th>

        <% :fv_classifier_row -> %>
          <th class="px-3 py-2 text-left">Label</th>
          <th class="px-3 py-2 text-left">Feature Vector</th>
          <th class="px-3 py-2 text-left">#</th>

        <% :kg_negative -> %>
          <th class="px-3 py-2 text-left">Head</th>
          <th class="px-3 py-2 text-left">Relation</th>
          <th class="px-3 py-2 text-left">Tail</th>

        <% :registry_entry -> %>
          <th class="px-3 py-2 text-left">Intent</th>
          <th class="px-3 py-2 text-left">Domain</th>
          <th class="px-3 py-2 text-left">Category</th>
          <th class="px-3 py-2 text-left">Speech Act</th>
          <th class="px-3 py-2 text-left">Required Slots</th>

        <% :speech_act_map_entry -> %>
          <th class="px-3 py-2 text-left">Speech Act</th>
          <th class="px-3 py-2 text-left">Canonical Intent</th>

        <% :gazetteer_entry -> %>
          <th class="px-3 py-2 text-left">Value</th>
          <th class="px-3 py-2 text-left">Synonyms</th>
          <th class="px-3 py-2 text-left">#</th>

        <% _ -> %>
          <th class="px-3 py-2 text-left">Data</th>
      <% end %>
    </tr>
    """
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
