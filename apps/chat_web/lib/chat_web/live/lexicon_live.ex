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

  defp origin_class(fact), do: if(seeded?(fact), do: "badge-ghost", else: "badge-primary")

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
          <h1 class="text-xl font-bold">Lexicon</h1>
          <p class="text-sm text-base-content/60">
            What the brain knows about a word, and where each answer came from
          </p>
        </div>
      </:page_header>

      <div class="p-4 space-y-4">
        <form id="lexicon-lookup" phx-submit="lookup" class="flex flex-col sm:flex-row gap-2">
          <input
            type="text"
            name="word"
            value={@word}
            placeholder="Look up a word"
            autocomplete="off"
            class="input input-bordered flex-1"
          />
          <button type="submit" class="btn btn-primary">Look up</button>
        </form>

        <div class="flex flex-wrap items-center gap-2 text-sm">
          <span class="text-base-content/50">Try:</span>
          <%= for example <- @examples do %>
            <button
              phx-click="example"
              phx-value-word={example}
              class="btn btn-xs btn-outline"
            >
              {example}
            </button>
          <% end %>
        </div>

        <%= if @result do %>
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <!-- What the brain knows -->
            <div class="card bg-base-100 border border-base-300">
              <div class="card-body p-4">
                <h2 class="card-title text-base">Summary</h2>

                <dl class="grid grid-cols-3 gap-y-1 text-sm">
                  <dt class="text-base-content/60">Known</dt>
                  <dd class="col-span-2">
                    <%= if @result.known do %>
                      <span class="badge badge-success badge-sm">known</span>
                    <% else %>
                      <span class="badge badge-warning badge-sm">out of vocabulary</span>
                    <% end %>
                  </dd>

                  <dt class="text-base-content/60">Lemma</dt>
                  <dd class="col-span-2 font-mono">{@result.lemma}</dd>

                  <dt class="text-base-content/60">Parts of speech</dt>
                  <dd class="col-span-2 font-mono">
                    {if @result.pos == [], do: "—", else: Enum.join(@result.pos, ", ")}
                  </dd>

                  <dt class="text-base-content/60">Domain</dt>
                  <dd class="col-span-2 font-mono">{@result.primary_domain || "—"}</dd>

                  <dt class="text-base-content/60">Senses</dt>
                  <dd class="col-span-2">{@result.polysemy}</dd>

                  <dt class="text-base-content/60">Definition</dt>
                  <dd class="col-span-2">{@result.definition || "—"}</dd>
                </dl>
              </div>
            </div>
            
    <!-- Negation -->
            <div class="card bg-base-100 border border-base-300">
              <div class="card-body p-4">
                <h2 class="card-title text-base">Negation</h2>

                <dl class="grid grid-cols-3 gap-y-1 text-sm">
                  <dt class="text-base-content/60">Negates</dt>
                  <dd class="col-span-2">
                    <%= if @result.negation.negates do %>
                      <span class="badge badge-error badge-sm">yes</span>
                    <% else %>
                      <span class="badge badge-ghost badge-sm">no</span>
                    <% end %>
                  </dd>

                  <dt class="text-base-content/60">Closed class</dt>
                  <dd class="col-span-2">{@result.negation.closed_class}</dd>

                  <dt class="text-base-content/60">Morphological</dt>
                  <dd class="col-span-2">{@result.negation.morphological}</dd>
                </dl>

                <%= if @result.negation.fact do %>
                  <div class="mt-2 text-sm bg-base-200 rounded p-2 font-mono">
                    {@result.word} = {@result.negation.fact.value["affix"]} + {@result.negation.fact.value[
                      "root"
                    ]}
                    <span class={"badge badge-xs ml-2 #{origin_class(@result.negation.fact)}"}>
                      {origin_label(@result.negation.fact)}
                    </span>
                  </div>
                <% else %>
                  <p class="text-sm text-base-content/50 mt-2">
                    No negation fact. Closed-class negators come from
                    priv/knowledge/linguistic.json; morphological ones are seeded
                    from WordNet by mix atlas.seed.
                  </p>
                <% end %>
              </div>
            </div>
          </div>
          
    <!-- Relations -->
          <div class="card bg-base-100 border border-base-300">
            <div class="card-body p-4">
              <h2 class="card-title text-base">Relations</h2>
              <p class="text-sm text-base-content/60">
                Answers from the facade. Highlighted entries are facts the brain
                owns; the rest come from WordNet.
              </p>

              <%= for {label, values, owned_set} <- @result.relations do %>
                <div class="mt-2">
                  <div class="text-sm font-semibold">{label} ({length(values)})</div>
                  <%= if values == [] do %>
                    <div class="text-sm text-base-content/50">—</div>
                  <% else %>
                    <div class="flex flex-wrap gap-1 mt-1">
                      <%= for value <- Enum.take(values, 40) do %>
                        <span class={
                          if MapSet.member?(owned_set, value),
                            do: "badge badge-primary badge-sm",
                            else: "badge badge-ghost badge-sm"
                        }>
                          {value}
                        </span>
                      <% end %>
                      <%= if length(values) > 40 do %>
                        <span class="text-xs text-base-content/50 self-center">
                          +{length(values) - 40} more
                        </span>
                      <% end %>
                    </div>
                  <% end %>
                </div>
              <% end %>
            </div>
          </div>
          
    <!-- Facts the brain owns -->
          <div class="card bg-base-100 border border-base-300">
            <div class="card-body p-4">
              <h2 class="card-title text-base">
                Owned facts ({length(@result.owned)})
              </h2>
              <p class="text-sm text-base-content/60">
                Everything the brain stores about this word, with its source. A
                learned fact and the seeded fact it contradicts both appear here.
              </p>

              <%= if @result.owned == [] do %>
                <p class="text-sm text-base-content/50">
                  Nothing stored. The brain only inherits WordNet for this word.
                </p>
              <% else %>
                <div class="overflow-x-auto mt-2">
                  <table class="table table-sm">
                    <thead>
                      <tr>
                        <th>Kind</th>
                        <th>Key</th>
                        <th>Ref</th>
                        <th>Value</th>
                        <th>Source</th>
                        <th>Conf.</th>
                        <th>Seen</th>
                        <th>State</th>
                      </tr>
                    </thead>
                    <tbody>
                      <%= for fact <- @result.owned do %>
                        <tr>
                          <td class="font-mono">{fact.kind}</td>
                          <td class="font-mono">{fact.key}</td>
                          <td class="font-mono">{fact.ref}</td>
                          <td class="font-mono text-xs">{truncate(inspect(fact.value), 60)}</td>
                          <td>
                            <span class={"badge badge-xs #{origin_class(fact)}"}>
                              {fact.source}
                            </span>
                          </td>
                          <td>{fact.confidence}</td>
                          <td>{fact.frequency}</td>
                          <td>
                            <%= if fact.archived do %>
                              <span class="badge badge-warning badge-xs">archived</span>
                            <% else %>
                              <span class="badge badge-success badge-xs">active</span>
                            <% end %>
                          </td>
                        </tr>
                      <% end %>
                    </tbody>
                  </table>
                </div>
              <% end %>
            </div>
          </div>
          
    <!-- Senses -->
          <div class="card bg-base-100 border border-base-300">
            <div class="card-body p-4">
              <h2 class="card-title text-base">Senses ({length(@result.senses)})</h2>

              <%= if @result.senses == [] do %>
                <p class="text-sm text-base-content/50">
                  No senses. Note that inflected forms are looked up as written, so
                  a plural can read as unknown even when its lemma is known.
                </p>
              <% else %>
                <div class="overflow-x-auto">
                  <table class="table table-sm">
                    <thead>
                      <tr>
                        <th>Synset</th>
                        <th>POS</th>
                        <th>Definition</th>
                      </tr>
                    </thead>
                    <tbody>
                      <%= for sense <- @result.senses do %>
                        <tr>
                          <td class="font-mono text-xs">{sense.synset_id}</td>
                          <td class="font-mono text-xs">{sense.pos}</td>
                          <td class="text-sm">{truncate(sense.definition, 120)}</td>
                        </tr>
                      <% end %>
                    </tbody>
                  </table>
                </div>
              <% end %>
            </div>
          </div>
        <% end %>
      </div>
    </.app_shell>
    """
  end
end
