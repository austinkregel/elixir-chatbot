defmodule ChatWeb.LexiconLive do
  @moduledoc """
  Isolation page for the brain's lexicon.

  Look up a word and see exactly what the brain knows about it, where each
  answer came from, and which facts it owns as opposed to inheriting from
  WordNet.

  Every lookup on this page goes through `Brain.Lexicon`, the same path the
  rest of the brain uses, so what is shown here is what the pipeline sees.
  """

  use ChatWeb, :live_view

  import ChatWeb.AppShell

  alias Brain.Lexicon
  alias Brain.LinguisticData

  @examples ~w(unable hopeless dog bank geese smarthome)

  @impl true
  def mount(_params, _session, socket) do
    {:ok,
     socket
     |> assign(:word, "")
     |> assign(:result, nil)
     |> assign(:examples, @examples)}
  end

  @impl true
  def handle_event("lookup", %{"word" => word}, socket) do
    trimmed = String.trim(word)

    result = if trimmed == "", do: nil, else: look_up(trimmed)

    {:noreply, socket |> assign(:word, trimmed) |> assign(:result, result)}
  end

  def handle_event("example", %{"word" => word}, socket) do
    {:noreply, socket |> assign(:word, word) |> assign(:result, look_up(word))}
  end

  # Everything here goes through the facade. The page never calls WordNet
  # directly, so it cannot show an answer the brain would not get.
  defp look_up(word) do
    normalized = String.downcase(word)
    owned = Lexicon.owned_facts(normalized, include_archived: true)

    %{
      word: normalized,
      known: Lexicon.known?(normalized),
      lemma: Lexicon.lemma(normalized),
      pos: Lexicon.pos(normalized),
      primary_domain: Lexicon.primary_domain(normalized),
      polysemy: Lexicon.polysemy_count(normalized),
      definition: definition(normalized),
      senses: Lexicon.senses(normalized),
      owned: owned,
      negation: negation(normalized, owned),
      relations: [
        {"Synonyms", Lexicon.synonyms(normalized), owned_targets(owned, "synonym")},
        {"Hypernyms", Lexicon.hypernyms(normalized), owned_targets(owned, "hypernym")},
        {"Antonyms", Lexicon.antonyms(normalized), owned_targets(owned, "antonym")}
      ]
    }
  end

  defp definition(word) do
    case Lexicon.definition(word) do
      {:ok, gloss} -> gloss
      :not_found -> nil
    end
  end

  defp owned_targets(owned, key) do
    owned
    |> Enum.filter(&(&1.kind == "relation" and &1.key == key and not &1.archived))
    |> Enum.map(& &1.ref)
    |> MapSet.new()
  end

  defp negation(word, owned) do
    fact =
      Enum.find(owned, &(&1.kind == "property" and &1.key == "negation" and not &1.archived))

    %{
      negates: LinguisticData.negator?(word),
      closed_class: LinguisticData.negation?(word),
      morphological: LinguisticData.morphological_negator?(word),
      fact: fact
    }
  end

  defp seeded?(fact), do: String.starts_with?(fact.source, "seed:")

  defp origin_label(fact), do: if(seeded?(fact), do: "seeded", else: "learned")

  defp origin_variant(fact), do: if(seeded?(fact), do: :default, else: :primary)

  defp truncate(nil, _len), do: ""

  defp truncate(text, len) when is_binary(text) do
    if String.length(text) > len, do: String.slice(text, 0, len) <> "…", else: text
  end

  @impl true
  def render(assigns) do
    ~H"""
    <.app_shell
      current_world_id={@current_world_id}
      available_worlds={@available_worlds}
      current_path={@current_path}
      system_ready={@system_ready}
      flash={@flash}
    >
      <:page_header>
        <div>
          <h1 class="text-title text-ink">Lexicon</h1>
          <p class="text-body text-ink-muted">
            What the brain knows about a word, and where each answer came from
          </p>
        </div>
      </:page_header>

      <div class="p-space-lg space-y-space-lg">
        <form id="lexicon-lookup" phx-submit="lookup" class="flex flex-col sm:flex-row gap-space-sm">
          <input
            type="text"
            name="word"
            value={@word}
            placeholder="Look up a word"
            autocomplete="off"
            class="flex-1 h-control-md px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body text-ink placeholder:text-ink-muted"
          />
          <.btn type="submit" variant={:primary}>Look up</.btn>
        </form>

        <div class="flex flex-wrap items-center gap-space-sm text-body">
          <span class="text-ink-muted">Try:</span>
          <%= for example <- @examples do %>
            <.btn
              phx-click="example"
              phx-value-word={example}
              variant={:outline}
              size={:xs}
            >
              {example}
            </.btn>
          <% end %>
        </div>

        <%= if @result do %>
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-space-lg">
            <!-- What the brain knows -->
            <.card>
              <.card_body>
                <h2 class="text-heading text-ink">Summary</h2>

                <dl class="mt-space-sm grid grid-cols-3 gap-y-space-xs text-body text-ink">
                  <dt class="text-ink-muted">Known</dt>
                  <dd class="col-span-2">
                    <%= if @result.known do %>
                      <.badge variant={:success}>known</.badge>
                    <% else %>
                      <.badge variant={:warning}>out of vocabulary</.badge>
                    <% end %>
                  </dd>

                  <dt class="text-ink-muted">Lemma</dt>
                  <dd class="col-span-2 text-value">{@result.lemma}</dd>

                  <dt class="text-ink-muted">Parts of speech</dt>
                  <dd class="col-span-2 text-value">
                    {if @result.pos == [], do: "—", else: Enum.join(@result.pos, ", ")}
                  </dd>

                  <dt class="text-ink-muted">Domain</dt>
                  <dd class="col-span-2 text-value">{@result.primary_domain || "—"}</dd>

                  <dt class="text-ink-muted">Senses</dt>
                  <dd class="col-span-2">{@result.polysemy}</dd>

                  <dt class="text-ink-muted">Definition</dt>
                  <dd class="col-span-2">{@result.definition || "—"}</dd>
                </dl>
              </.card_body>
            </.card>

    <!-- Negation -->
            <.card>
              <.card_body>
                <h2 class="text-heading text-ink">Negation</h2>

                <dl class="mt-space-sm grid grid-cols-3 gap-y-space-xs text-body text-ink">
                  <dt class="text-ink-muted">Negates</dt>
                  <dd class="col-span-2">
                    <.badge>{if @result.negation.negates, do: "yes", else: "no"}</.badge>
                  </dd>

                  <dt class="text-ink-muted">Closed class</dt>
                  <dd class="col-span-2">{@result.negation.closed_class}</dd>

                  <dt class="text-ink-muted">Morphological</dt>
                  <dd class="col-span-2">{@result.negation.morphological}</dd>
                </dl>

                <%= if @result.negation.fact do %>
                  <div class="mt-space-sm text-value text-ink bg-surface-sunk rounded-sm p-space-sm">
                    {@result.word} = {@result.negation.fact.value["affix"]} + {@result.negation.fact.value[
                      "root"
                    ]}
                    <.badge variant={origin_variant(@result.negation.fact)} size={:xs} class="ml-space-sm">
                      {origin_label(@result.negation.fact)}
                    </.badge>
                  </div>
                <% else %>
                  <p class="text-body text-ink-muted mt-space-sm">
                    No negation fact. Closed-class negators come from
                    priv/knowledge/linguistic.json; morphological ones are seeded
                    from WordNet by mix atlas.seed.
                  </p>
                <% end %>
              </.card_body>
            </.card>
          </div>

    <!-- Relations -->
          <.card>
            <.card_body>
              <h2 class="text-heading text-ink">Relations</h2>
              <p class="text-body text-ink-muted">
                Answers from the facade. Highlighted entries are facts the brain
                owns; the rest come from WordNet.
              </p>

              <%= for {label, values, owned_set} <- @result.relations do %>
                <div class="mt-space-sm">
                  <div class="text-subheading text-ink">{label} ({length(values)})</div>
                  <%= if values == [] do %>
                    <div class="text-body text-ink-muted">—</div>
                  <% else %>
                    <div class="flex flex-wrap gap-space-xs mt-space-xs">
                      <%= for value <- Enum.take(values, 40) do %>
                        <.badge variant={if MapSet.member?(owned_set, value), do: :primary, else: :default}>
                          {value}
                        </.badge>
                      <% end %>
                      <%= if length(values) > 40 do %>
                        <span class="text-caption text-ink-muted self-center">
                          +{length(values) - 40} more
                        </span>
                      <% end %>
                    </div>
                  <% end %>
                </div>
              <% end %>
            </.card_body>
          </.card>

    <!-- Facts the brain owns -->
          <.card>
            <.card_body>
              <h2 class="text-heading text-ink">
                Owned facts ({length(@result.owned)})
              </h2>
              <p class="text-body text-ink-muted">
                Everything the brain stores about this word, with its source. A
                learned fact and the seeded fact it contradicts both appear here.
              </p>

              <%= if @result.owned == [] do %>
                <p class="text-body text-ink-muted">
                  Nothing stored. The brain only inherits WordNet for this word.
                </p>
              <% else %>
                <div class="overflow-x-auto mt-space-sm">
                  <table class="w-full text-left text-body-dense text-ink">
                    <thead class="bg-surface-sunk">
                      <tr>
                        <th class="h-row-compact px-space-sm text-label text-ink-muted">Kind</th>
                        <th class="h-row-compact px-space-sm text-label text-ink-muted">Key</th>
                        <th class="h-row-compact px-space-sm text-label text-ink-muted">Ref</th>
                        <th class="h-row-compact px-space-sm text-label text-ink-muted">Value</th>
                        <th class="h-row-compact px-space-sm text-label text-ink-muted">Source</th>
                        <th class="h-row-compact px-space-sm text-label text-ink-muted">Conf.</th>
                        <th class="h-row-compact px-space-sm text-label text-ink-muted">Seen</th>
                        <th class="h-row-compact px-space-sm text-label text-ink-muted">State</th>
                      </tr>
                    </thead>
                    <tbody class="divide-y divide-border">
                      <%= for fact <- @result.owned do %>
                        <tr>
                          <td class="h-row-compact px-space-sm text-value">{fact.kind}</td>
                          <td class="h-row-compact px-space-sm text-value">{fact.key}</td>
                          <td class="h-row-compact px-space-sm text-value">{fact.ref}</td>
                          <td class="h-row-compact px-space-sm text-term">{truncate(inspect(fact.value), 60)}</td>
                          <td class="h-row-compact px-space-sm">
                            <.badge variant={origin_variant(fact)} size={:xs}>
                              {fact.source}
                            </.badge>
                          </td>
                          <td class="h-row-compact px-space-sm text-value">{fact.confidence}</td>
                          <td class="h-row-compact px-space-sm text-value">{fact.frequency}</td>
                          <td class="h-row-compact px-space-sm">
                            <%= if fact.archived do %>
                              <.badge variant={:warning} size={:xs}>archived</.badge>
                            <% else %>
                              <.badge variant={:success} size={:xs}>active</.badge>
                            <% end %>
                          </td>
                        </tr>
                      <% end %>
                    </tbody>
                  </table>
                </div>
              <% end %>
            </.card_body>
          </.card>

    <!-- Senses -->
          <.card>
            <.card_body>
              <h2 class="text-heading text-ink">Senses ({length(@result.senses)})</h2>

              <%= if @result.senses == [] do %>
                <p class="text-body text-ink-muted">
                  No senses. Note that inflected forms are looked up as written, so
                  a plural can read as unknown even when its lemma is known.
                </p>
              <% else %>
                <div class="overflow-x-auto">
                  <table class="w-full text-left text-body-dense text-ink">
                    <thead class="bg-surface-sunk">
                      <tr>
                        <th class="h-row-compact px-space-sm text-label text-ink-muted">Synset</th>
                        <th class="h-row-compact px-space-sm text-label text-ink-muted">POS</th>
                        <th class="h-row-compact px-space-sm text-label text-ink-muted">Definition</th>
                      </tr>
                    </thead>
                    <tbody class="divide-y divide-border">
                      <%= for sense <- @result.senses do %>
                        <tr>
                          <td class="h-row-compact px-space-sm text-ref">{sense.synset_id}</td>
                          <td class="h-row-compact px-space-sm text-ref">{sense.pos}</td>
                          <td class="h-row-compact px-space-sm text-body">{truncate(sense.definition, 120)}</td>
                        </tr>
                      <% end %>
                    </tbody>
                  </table>
                </div>
              <% end %>
            </.card_body>
          </.card>
        <% end %>
      </div>
    </.app_shell>
    """
  end
end
