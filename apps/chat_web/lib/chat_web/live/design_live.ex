defmodule ChatWeb.DesignLive do
  @moduledoc """
  The `/design` gallery: every Retroduct design token and every shared
  component state, rendered in light and dark side by side.

  Each section renders its content twice, once inside a `data-theme="light"`
  panel and once inside a `data-theme="dark"` panel. The tokens in
  `assets/css/app.css` are themed per subtree, so both themes are visible at
  once regardless of the theme chosen in the shell.

  The harness components are rendered from real calls: `Runner.run/2` on small
  sample functions that record provenance in each of the five origins, a real
  `Atlas.Verification.Comparison.compare/3` result, and real
  `Atlas.Schemas.VerificationCase` structs. The regression gate's states are
  real `Brain.Evaluation.Gate.verdict/3` judgments of sample baselines and
  results. The samples are labeled as samples; nothing on this page is a
  verification result or an evaluation.

  The pagination samples are live: each bar keeps its own current page, so
  one page, a few pages and many pages with gaps can each be stepped through.

  The execute confirmations are live: each trigger opens its panel in place,
  Cancel and Escape close it, and confirming writes nothing, because this page
  has nothing to write. The device sample answers its confirm with the failure
  shown in the panel, which is how a failed confirmation looks. Visiting
  `/design?confirm=open` opens every panel at once.
  """

  use ChatWeb, :live_view

  import ChatWeb.AppShell

  alias Atlas.Schemas.VerificationCase
  alias Atlas.Verification.Comparison
  alias Brain.Evaluation.Gate
  alias Brain.Provenance
  alias ChatWeb.Harness.Diff
  alias ChatWeb.Harness.Runner

  # Every class string is written out in full so Tailwind's source scan sees it.
  @color_groups [
    {"Surfaces",
     [
       {"ground", "bg-ground"},
       {"surface", "bg-surface"},
       {"surface-sunk", "bg-surface-sunk"},
       {"surface-raised", "bg-surface-raised"}
     ]},
    {"Text", [{"ink", "bg-ink"}, {"ink-muted", "bg-ink-muted"}]},
    {"Borders", [{"border", "bg-border"}, {"border-strong", "bg-border-strong"}]},
    {"Action, selection and focus",
     [
       {"primary", "bg-primary"},
       {"primary-hover", "bg-primary-hover"},
       {"primary-wash", "bg-primary-wash"},
       {"on-primary", "bg-on-primary"},
       {"accent", "bg-accent"},
       {"accent-wash", "bg-accent-wash"},
       {"on-accent", "bg-on-accent"},
       {"focus", "bg-focus"}
     ]},
    {"Base hues",
     [
       {"slate", "bg-slate"},
       {"slate-wash", "bg-slate-wash"},
       {"ochre", "bg-ochre"},
       {"ochre-wash", "bg-ochre-wash"},
       {"sienna", "bg-sienna"},
       {"sienna-wash", "bg-sienna-wash"},
       {"red", "bg-red"},
       {"red-wash", "bg-red-wash"},
       {"plum", "bg-plum"},
       {"plum-wash", "bg-plum-wash"},
       {"green", "bg-green"},
       {"green-wash", "bg-green-wash"},
       {"blue", "bg-blue"},
       {"blue-light", "bg-blue-light"},
       {"blue-track", "bg-blue-track"}
     ]},
    {"Value origin",
     [
       {"origin-computed", "bg-origin-computed"},
       {"origin-declared", "bg-origin-declared"},
       {"origin-declared-wash", "bg-origin-declared-wash"},
       {"origin-default", "bg-origin-default"},
       {"origin-standin-wash", "bg-origin-standin-wash"},
       {"origin-absent", "bg-origin-absent"},
       {"origin-absent-wash", "bg-origin-absent-wash"},
       {"origin-unavailable", "bg-origin-unavailable"},
       {"origin-unavailable-wash", "bg-origin-unavailable-wash"},
       {"origin-unobserved", "bg-origin-unobserved"}
     ]},
    {"What a score means",
     [
       {"score-calibrated", "bg-score-calibrated"},
       {"score-relative", "bg-score-relative"},
       {"score-heuristic", "bg-score-heuristic"},
       {"score-unbounded", "bg-score-unbounded"},
       {"score-track", "bg-score-track"},
       {"score-axis", "bg-score-axis"},
       {"score-count", "bg-score-count"}
     ]},
    {"Candidate versus resolved",
     [
       {"candidate-outline", "bg-candidate-outline"},
       {"candidate-wash", "bg-candidate-wash"},
       {"resolved-mark", "bg-resolved-mark"},
       {"resolved-wash", "bg-resolved-wash"}
     ]},
    {"Reach of an action",
     [
       {"reach-read", "bg-reach-read"},
       {"reach-local", "bg-reach-local"},
       {"reach-shared", "bg-reach-shared"},
       {"reach-shared-confirm", "bg-reach-shared-confirm"},
       {"on-reach-shared", "bg-on-reach-shared"},
       {"reach-device", "bg-reach-device"},
       {"reach-device-confirm", "bg-reach-device-confirm"},
       {"on-reach-device", "bg-on-reach-device"}
     ]},
    {"Verdict",
     [
       {"verdict-pass", "bg-verdict-pass"},
       {"verdict-pass-wash", "bg-verdict-pass-wash"},
       {"verdict-fail", "bg-verdict-fail"},
       {"verdict-fail-wash", "bg-verdict-fail-wash"},
       {"verdict-error", "bg-verdict-error"},
       {"verdict-error-wash", "bg-verdict-error-wash"},
       {"verdict-pending", "bg-verdict-pending"}
     ]},
    {"Availability",
     [
       {"avail-ready", "bg-avail-ready"},
       {"avail-stale", "bg-avail-stale"},
       {"avail-stale-wash", "bg-avail-stale-wash"},
       {"avail-untrained", "bg-avail-untrained"},
       {"avail-missing", "bg-avail-missing"},
       {"avail-corrupt", "bg-avail-corrupt"},
       {"avail-corrupt-wash", "bg-avail-corrupt-wash"},
       {"avail-kind", "bg-avail-kind"}
     ]},
    {"Job progress", [{"progress-fill", "bg-progress-fill"}, {"progress-track", "bg-progress-track"}]}
  ]

  @type_styles [
    {"title", "text-title", "Speech act classifier"},
    {"heading", "text-heading", "Where each value came from"},
    {"subheading", "text-subheading", "Against the saved expectation"},
    {"body", "text-body", "Nothing on this path is instrumented yet."},
    {"body-dense", "text-body-dense", "fallback default"},
    {"caption", "text-caption", "7 of 143 values asserted"},
    {"label", "text-label", "Came from"},
    {"value", "text-value", "0.8314 · softmax share"},
    {"value-strong", "text-value-strong", "assertive"},
    {"term", "text-term", "%{\"__struct__\" => \"Brain.Lattice\"}"},
    {"ref", "text-ref", "runner.ex:222-281"},
    {"offset", "text-offset", "4..9 graphemes, end inclusive"}
  ]

  @spacing [
    {"space-2xs", "w-space-2xs", "2px"},
    {"space-xs", "w-space-xs", "4px"},
    {"space-sm", "w-space-sm", "8px"},
    {"space-md", "w-space-md", "12px"},
    {"space-lg", "w-space-lg", "16px"},
    {"space-xl", "w-space-xl", "24px"},
    {"space-2xl", "w-space-2xl", "32px"},
    {"space-3xl", "w-space-3xl", "48px"}
  ]

  @density [
    {"row-compact", "h-row-compact", "24px"},
    {"row-regular", "h-row-regular", "32px"},
    {"row-relaxed", "h-row-relaxed", "40px"},
    {"control-sm", "h-control-sm", "24px"},
    {"control-md", "h-control-md", "32px"}
  ]

  @radius [
    {"radius-none", "rounded-none", "0px"},
    {"radius-sm", "rounded-sm", "2px"},
    {"radius-md", "rounded-md", "4px"},
    {"radius-pip", "rounded-pip", "9999px"}
  ]

  @statuses [
    :ready,
    :running,
    :healthy,
    :initializing,
    :building_vocabulary,
    :loading,
    :idle,
    :not_started,
    :degraded,
    :warning,
    :error,
    :critical
  ]

  # Each score kind in its text form, and in its bar form where the kind has
  # one. The values are samples.
  @scores [
    %{kind: :model_confidence, value: 0.72},
    %{kind: :softmax_share, value: 0.12},
    %{kind: :margin, value: 0.31},
    %{kind: :entropy, value: 0.44},
    %{kind: :weighted_vote, value: 0.58},
    %{kind: :mapped_confidence, value: 0.78},
    %{kind: :reranked_confidence, value: 0.66},
    %{kind: :match_confidence, value: 0.81},
    %{kind: :completeness, value: 0.9, parts: ["actor", "object", "verb"]},
    %{kind: :belief_confidence, value: 0.7},
    %{kind: :analyzer_activation, value: 0.42},
    %{kind: :activation, value: 0.35},
    %{kind: :accumulated_confidence, value: 0.63},
    %{kind: :cosine_similarity, value: 0.61},
    %{kind: :cosine_similarity, value: -0.24},
    %{kind: :distance, value: 2.41, axis_max: 7}
  ]

  @text_only_scores [
    %{kind: :activation_sum, value: 1.37},
    %{kind: :raw_score, value: -3.2, source: "log-probability"},
    %{kind: :count, value: 17, noun: "times candidate"},
    %{kind: :count, value: 3, of: 12, noun: "voters"},
    %{kind: :unestablished, value: 0.64},
    %{kind: :unestablished, value: 0.64, candidates: "model confidence or weighted vote"}
  ]

  @confirm_kinds [:shared, :device, :local, :form]

  # Each pagination sample: its rows, its page size and the page it opens on.
  # One page; a few pages, every number shown; many pages, windowed with gaps.
  @page_samples %{
    "one" => %{title: "One page", total: 12, page_size: 50, page: 1},
    "few" => %{title: "A few pages", total: 180, page_size: 50, page: 2},
    "many" => %{title: "Many pages, with gaps", total: 1204, page_size: 20, page: 30}
  }

  @page_sample_order ["one", "few", "many"]

  @gate_sample_order [
    {:not_set, "gate not set"},
    {:pass, "pass"},
    {:fail, "fail"},
    {:canary_not_measured, "canary not measured"}
  ]

  @impl true
  def mount(_params, _session, socket) do
    returned = Runner.run(&sample_with_every_origin/1, "turn on the kitchen light")
    raised = Runner.run(&sample_that_raises/1, "turn on the kitchen light")
    unrecorded = Runner.run(&sample_without_provenance/1, "turn on the kitchen light")

    {:ok,
     socket
     |> assign(:page_title, "Design Language")
     |> assign(:returned, returned)
     |> assign(:raised, raised)
     |> assign(:unrecorded, unrecorded)
     |> assign(:failing, Comparison.compare(%{intent: "greeting", score: 0.5}, returned.value))
     |> assign(:passing, Comparison.compare(%{intent: "directive"}, returned.value))
     |> assign(:cases, sample_cases(returned.value))
     |> assign(:open_confirms, MapSet.new())
     |> assign(:confirm_errors, %{})
     |> assign(:confirmed, nil)
     |> assign(:sample_page, 2)
     |> assign(:page_samples, @page_samples)
     |> assign(:gate_verdicts, gate_samples())}
  end

  @impl true
  def handle_params(%{"confirm" => "open"}, _uri, socket) do
    every = for theme <- ["light", "dark"], kind <- @confirm_kinds, into: MapSet.new(), do: confirm_id(kind, theme)
    {:noreply, assign(socket, :open_confirms, every)}
  end

  def handle_params(_params, _uri, socket), do: {:noreply, socket}

  @impl true
  def handle_event("open_confirm", %{"id" => id}, socket) do
    {:noreply,
     socket
     |> update(:open_confirms, &MapSet.put(&1, id))
     |> update(:confirm_errors, &Map.delete(&1, id))}
  end

  def handle_event("close_confirm", %{"id" => id}, socket) do
    {:noreply,
     socket
     |> update(:open_confirms, &MapSet.delete(&1, id))
     |> update(:confirm_errors, &Map.delete(&1, id))}
  end

  def handle_event("confirm", %{"id" => id} = params, socket) do
    if String.starts_with?(id, "design-confirm-device-") do
      {:noreply,
       update(
         socket,
         :confirm_errors,
         &Map.put(
           &1,
           id,
           "Nothing was sent to light.kitchen: this gallery has no Home Assistant connection. " <>
             "A failed confirmation stays open and says what failed, as this one does."
         )
       )}
    else
      topic = params["topic"]

      {:noreply,
       socket
       |> update(:open_confirms, &MapSet.delete(&1, id))
       |> assign(
         :confirmed,
         "Confirmed #{id}#{if topic, do: " with topic \"#{topic}\"", else: ""}. The sample wrote nothing."
       )}
    end
  end

  def handle_event("design_page", %{"page" => page}, socket) do
    {:noreply, assign(socket, :sample_page, String.to_integer(page))}
  end

  def handle_event("design_page_" <> sample, %{"page" => page}, socket)
      when is_map_key(@page_samples, sample) do
    {:noreply, update(socket, :page_samples, &put_in(&1, [sample, :page], String.to_integer(page)))}
  end

  @impl true
  def handle_info({:world_context_changed, _world_id}, socket) do
    {:noreply, socket}
  end

  @doc false
  def sample_with_every_origin(input) do
    Provenance.record(["entity", "familiarity"], 0.76, :computed,
      source: "ChatWeb.DesignLive.sample_with_every_origin/1"
    )

    Provenance.record(["config", "threshold"], 0.6, :declared,
      source: "ChatWeb.DesignLive.sample_with_every_origin/1",
      meta: %{"file" => "config/config.exs"}
    )

    Provenance.record(["config", "domain_lemmas"], %{}, :default,
      source: "ChatWeb.DesignLive.sample_with_every_origin/1",
      meta: %{"reason" => "the declaration holds no domain_lemmas"}
    )

    Provenance.record(["memory", "context"], [], :absent,
      source: "ChatWeb.DesignLive.sample_with_every_origin/1",
      meta: %{"reason" => "no prior turns to compute from"}
    )

    Provenance.record(["config", "gazetteer"], nil, :unavailable,
      source: "ChatWeb.DesignLive.sample_with_every_origin/1",
      meta: %{"reason" => "the table could not be read"}
    )

    %{intent: "directive", score: 0.83, text: input, entities: [%{type: :device, label: "light"}]}
  end

  @doc false
  def sample_that_raises(_input) do
    raise ArgumentError, "a sample raise, rendered as the result it would be"
  end

  @doc false
  def sample_without_provenance(input) do
    %{intent: "directive", text: input}
  end

  defp sample_cases(returned) do
    run_at = ~U[2026-10-06 12:00:00.000000Z]
    actual = Comparison.normalize(returned)

    error = %{
      "__error__" => true,
      "kind" => "ArgumentError",
      "message" => "a sample raise",
      "stacktrace" => []
    }

    for {status, name, expected, last_actual, last_run_at} <- [
          {"pass", "a plain directive", %{"intent" => "directive"}, actual, run_at},
          {"fail", "a greeting read as a directive", %{"intent" => "greeting", "score" => 0.5}, actual,
           run_at},
          {"error", "an input that raises", %{"intent" => "directive"}, error, run_at},
          {"pending", "a case never run", %{"intent" => "directive", "score" => 0.83}, nil, nil}
        ] do
      %VerificationCase{
        subsystem: "speech_act",
        name: name,
        world_id: "default",
        status: status,
        expected: expected,
        last_actual: last_actual,
        last_run_at: last_run_at
      }
    end
  end

  # The gate's four states, each judged by `Gate.verdict/3` from a sample
  # baseline and result: no baseline; a fall within the allowance; a fall
  # beyond it with new failed predictions; and a result saved without
  # diagnostics, so the error canary cannot be measured and the gate fails
  # although macro-F1 held.
  defp gate_samples do
    baseline =
      {:ok,
       %{
         "intent" => %{
           "macro_f1" => 0.500,
           "accuracy" => 0.287,
           "diagnostics" => %{"ok" => 4860, "unknown" => 8, "errored" => 2}
         }
       }}

    %{
      not_set: Gate.verdict("intent", :not_set, nil),
      pass:
        Gate.verdict("intent", baseline, %{
          "macro_f1" => 0.496,
          "diagnostics" => %{"ok" => 4861, "unknown" => 7, "errored" => 2}
        }),
      fail:
        Gate.verdict("intent", baseline, %{
          "macro_f1" => 0.452,
          "diagnostics" => %{"ok" => 4855, "unknown" => 11, "errored" => 4}
        }),
      canary_not_measured: Gate.verdict("intent", baseline, %{"macro_f1" => 0.496})
    }
  end

  defp confirm_id(kind, theme), do: "design-confirm-#{kind}-#{theme}"
  defp trigger_id(kind, theme), do: "design-trigger-#{kind}-#{theme}"

  @impl true
  def render(assigns) do
    assigns =
      assigns
      |> assign(:color_groups, @color_groups)
      |> assign(:type_styles, @type_styles)
      |> assign(:spacing, @spacing)
      |> assign(:density, @density)
      |> assign(:radius, @radius)
      |> assign(:statuses, @statuses)
      |> assign(:scores, @scores)
      |> assign(:text_only_scores, @text_only_scores)
      |> assign(:page_sample_order, @page_sample_order)
      |> assign(:gate_sample_order, @gate_sample_order)

    ~H"""
    <.app_shell
      current_world_id={@current_world_id}
      available_worlds={@available_worlds}
      current_path={@current_path}
      system_ready={@system_ready}
      flash={@flash}
    >
      <:page_header>
        <h1 class="text-title text-ink">Design Language</h1>
        <p class="text-body text-ink-muted">
          Every token and every shared component state, light on the left and dark on the right.
          Utilities read from the token name: <span class="text-ref">bg-surface</span>, <span class="text-ref">text-ink-muted</span>,
          <span class="text-ref">border-border-strong</span>, <span class="text-ref">p-space-md</span>,
          <span class="text-ref">h-control-md</span>, <span class="text-ref">text-label</span>.
        </p>
      </:page_header>

      <div class="space-y-space-2xl p-space-lg">
        <.section title="Color tokens">
          <.both :let={_theme}>
            <div class="space-y-space-lg">
              <div :for={{group, tokens} <- @color_groups}>
                <h4 class="mb-space-sm text-label text-ink-muted">{group}</h4>
                <div class="grid grid-cols-2 gap-space-sm sm:grid-cols-3">
                  <div :for={{name, class} <- tokens} class="flex items-center gap-space-sm min-w-0">
                    <span class={["size-control-md shrink-0 rounded-sm border border-border-strong", class]} />
                    <div class="min-w-0">
                      <div class="text-ref text-ink truncate">{name}</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </.both>
        </.section>

        <.section title="Type">
          <.both :let={_theme}>
            <div class="space-y-space-sm">
              <div :for={{name, class, sample} <- @type_styles} class="flex items-baseline gap-space-md">
                <span class="w-28 shrink-0 text-ref text-ink-muted">{class}</span>
                <span class={[class, "text-ink min-w-0 truncate"]}>{sample}</span>
                <span class="sr-only">{name}</span>
              </div>
            </div>
          </.both>
        </.section>

        <.section title="Spacing, density, radius, stroke and elevation">
          <.both :let={_theme}>
            <div class="space-y-space-xl">
              <div>
                <h4 class="mb-space-sm text-label text-ink-muted">Spacing</h4>
                <div :for={{name, class, value} <- @spacing} class="flex items-center gap-space-md">
                  <span class="w-28 text-ref text-ink-muted">p-{name}</span>
                  <span class={["h-space-sm bg-blue", class]} />
                  <span class="text-caption text-ink-muted">{value}</span>
                </div>
              </div>
              <div>
                <h4 class="mb-space-sm text-label text-ink-muted">Density</h4>
                <div class="flex flex-wrap items-end gap-space-md">
                  <div :for={{name, class, value} <- @density} class="text-center">
                    <div class={["w-20 rounded-sm border border-border-strong bg-surface-sunk", class]} />
                    <div class="mt-space-xs text-ref text-ink">h-{name}</div>
                    <div class="text-caption text-ink-muted">{value}</div>
                  </div>
                </div>
              </div>
              <div>
                <h4 class="mb-space-sm text-label text-ink-muted">Radius</h4>
                <div class="flex flex-wrap gap-space-md">
                  <div :for={{name, class, value} <- @radius} class="text-center">
                    <div class={["size-12 border border-border-strong bg-surface-sunk", class]} />
                    <div class="mt-space-xs text-ref text-ink">{name}</div>
                    <div class="text-caption text-ink-muted">{value}</div>
                  </div>
                </div>
              </div>
              <div>
                <h4 class="mb-space-sm text-label text-ink-muted">Stroke</h4>
                <div class="flex flex-wrap items-center gap-space-xl">
                  <div class="text-center">
                    <div class="size-12 border-hairline border-border-strong" />
                    <div class="mt-space-xs text-ref text-ink">stroke-hairline</div>
                  </div>
                  <div class="text-center">
                    <div class="size-12 border-mark border-border-strong" />
                    <div class="mt-space-xs text-ref text-ink">stroke-mark</div>
                  </div>
                  <div class="text-center">
                    <div class="size-12 rounded-md bg-surface-sunk outline-mark outline-focus" />
                    <div class="mt-space-xs text-ref text-ink">stroke-focus at focus-offset</div>
                  </div>
                </div>
              </div>
              <div>
                <h4 class="mb-space-sm text-label text-ink-muted">Elevation</h4>
                <div class="w-56 rounded-md border border-border-strong bg-surface-raised p-space-md shadow-overlay">
                  <div class="text-subheading text-ink">shadow-overlay</div>
                  <div class="text-caption text-ink-muted">Menus, popovers, the execute confirmation.</div>
                </div>
              </div>
            </div>
          </.both>
        </.section>

        <.section title="Marks">
          <.both :let={_theme}>
            <div class="grid grid-cols-2 gap-space-sm sm:grid-cols-3">
              <div
                :for={
                  shape <- [
                    :filled_circle,
                    :hollow_circle,
                    :dashed_circle,
                    :dotted_circle,
                    :half_circle,
                    :struck_circle,
                    :filled_square,
                    :dotted_square,
                    :dashed_square,
                    :check_square,
                    :cross_square,
                    :alert_triangle,
                    :filled_diamond
                  ]
                }
                class="flex items-center gap-space-sm"
              >
                <.mark shape={shape} class="size-4 text-ink" />
                <span class="text-ref text-ink-muted">{shape}</span>
              </div>
            </div>
          </.both>
        </.section>

        <.section title="Value origin">
          <.both :let={_theme}>
            <div class="space-y-space-sm">
              <div
                :for={origin <- [:computed, :declared, :default, :absent, :unavailable, :unobserved]}
                class="flex items-center gap-space-md"
              >
                <span class="w-28 text-ref text-ink-muted">{origin}</span>
                <Runner.origin origin={origin} />
              </div>
              <p class="text-caption text-ink-muted">
                Underlines in a provenance row: computed and declared solid, default dashed,
                absent dotted, unavailable wavy. Stand-in rows take their origin's wash.
              </p>
            </div>
          </.both>
        </.section>

        <.section title="What a score means">
          <.both :let={_theme}>
            <div class="space-y-space-lg">
              <div>
                <h4 class="mb-space-sm text-label text-ink-muted">Bar form, beside the text form</h4>
                <div class="space-y-space-sm">
                  <div :for={score <- @scores}>
                    <.score_display {score} form={:bar} />
                  </div>
                </div>
              </div>
              <div>
                <h4 class="mb-space-sm text-label text-ink-muted">Text form, every kind</h4>
                <div class="flex flex-col gap-space-xs">
                  <.score_display :for={score <- @scores} {score} />
                  <.score_display :for={score <- @text_only_scores} {score} />
                  <.no_confidence method="intent inferred from speech act" />
                  <span class="inline-flex items-center gap-space-xs">
                    <span class="text-value text-ink underline underline-offset-2 decoration-dashed decoration-origin-default">
                      0.40
                    </span>
                    <Runner.origin origin={:default} />
                    <span class="text-caption text-ink-muted">a constant stand-in takes its origin, not a bar</span>
                  </span>
                </div>
              </div>
            </div>
          </.both>
        </.section>

        <.section title="Candidate versus resolved">
          <.both :let={_theme}>
            <div class="space-y-space-md">
              <div class="flex flex-wrap gap-space-sm bg-candidate-wash p-space-sm">
                <span
                  :for={{label, count} <- [{"polar_interrogative", 1}, {"assertion", 2}]}
                  class="inline-flex items-center gap-space-xs rounded-sm border border-dashed border-candidate-outline px-space-xs text-value text-ink"
                >
                  {label} <span class="text-score-count">{count}</span>
                </span>
              </div>
              <div class="flex items-center justify-between rounded-sm border border-resolved-mark bg-resolved-wash px-space-sm h-row-regular">
                <span class="inline-flex items-center gap-space-xs text-value-strong text-ink">
                  <.mark shape={:filled_diamond} class="size-2.5 text-resolved-mark" /> assertive
                </span>
                <span class="text-value text-ink-muted">margin 0.31</span>
              </div>
            </div>
          </.both>
        </.section>

        <.section title="Reach of an action">
          <.both :let={theme}>
            <div class="space-y-space-md">
              <div class="flex flex-wrap items-center gap-space-sm">
                <.btn>Run</.btn>
                <span class="text-caption text-ink-muted">read-only: no badge</span>
              </div>
              <div class="flex flex-wrap items-center gap-space-sm">
                <.btn id={"design-reach-local-#{theme}"} reach={:local} target="micro/intent.term">
                  Rebuild
                </.btn>
              </div>
              <div class="flex flex-wrap items-center gap-space-sm">
                <.btn id={"design-reach-shared-#{theme}"} reach={:shared} target="candidate 41">
                  Approve
                </.btn>
              </div>
              <div class="flex flex-wrap items-center gap-space-sm">
                <.btn id={"design-reach-device-#{theme}"} reach={:device} target="light.kitchen">
                  Turn on
                </.btn>
              </div>
              <div>
                <h4 class="mb-space-sm text-label text-ink-muted">Reach badges</h4>
                <div class="flex flex-wrap items-center gap-space-sm">
                  <.reach_badge reach={:local} />
                  <.reach_badge reach={:local} target="data/intents.json" />
                  <.reach_badge reach={:shared} target="learning session" />
                  <.reach_badge reach={:device} target="light.kitchen" />
                </div>
              </div>
            </div>
          </.both>
        </.section>

        <.section title="Execute confirmation">
          <.both :let={theme}>
            <div class="space-y-space-lg">
              <p :if={@confirmed} class="text-caption text-ink-muted">{@confirmed}</p>
              <div class="space-y-space-sm">
                <.btn
                  id={trigger_id(:shared, theme)}
                  reach={:shared}
                  target="candidate 41"
                  phx-click="open_confirm"
                  phx-value-id={confirm_id(:shared, theme)}
                >
                  Reject
                </.btn>
                <.execute_confirm
                  id={confirm_id(:shared, theme)}
                  open={MapSet.member?(@open_confirms, confirm_id(:shared, theme))}
                  reach={:shared}
                  verb="Reject"
                  target="candidate 41"
                  consequence="Marks this candidate rejected in the review store and records the rejection against its source's domain."
                  on_confirm={JS.push("confirm", value: %{id: confirm_id(:shared, theme)})}
                  on_cancel={JS.push("close_confirm", value: %{id: confirm_id(:shared, theme)})}
                  trigger_id={trigger_id(:shared, theme)}
                  error={@confirm_errors[confirm_id(:shared, theme)]}
                />
              </div>
              <div class="space-y-space-sm">
                <.btn
                  id={trigger_id(:device, theme)}
                  reach={:device}
                  target="light.kitchen"
                  phx-click="open_confirm"
                  phx-value-id={confirm_id(:device, theme)}
                >
                  Turn on
                </.btn>
                <.execute_confirm
                  id={confirm_id(:device, theme)}
                  open={MapSet.member?(@open_confirms, confirm_id(:device, theme))}
                  reach={:device}
                  verb="Turn on"
                  target="light.kitchen"
                  consequence="Calls the Home Assistant service light.turn_on."
                  current_state="off, as a sample"
                  on_confirm={JS.push("confirm", value: %{id: confirm_id(:device, theme)})}
                  on_cancel={JS.push("close_confirm", value: %{id: confirm_id(:device, theme)})}
                  trigger_id={trigger_id(:device, theme)}
                  error={@confirm_errors[confirm_id(:device, theme)]}
                />
              </div>
              <div class="space-y-space-sm">
                <.btn
                  id={trigger_id(:local, theme)}
                  variant={:outline}
                  reach={:local}
                  target="world star-trek"
                  phx-click="open_confirm"
                  phx-value-id={confirm_id(:local, theme)}
                >
                  Unload world
                </.btn>
                <.execute_confirm
                  id={confirm_id(:local, theme)}
                  open={MapSet.member?(@open_confirms, confirm_id(:local, theme))}
                  reach={:local}
                  removes
                  verb="Unload world"
                  target="world star-trek"
                  consequence="Unloads this world and its gazetteer overlay from this node. Its folder under priv/training_worlds stays, and the world loads again at the next start."
                  on_confirm={JS.push("confirm", value: %{id: confirm_id(:local, theme)})}
                  on_cancel={JS.push("close_confirm", value: %{id: confirm_id(:local, theme)})}
                  trigger_id={trigger_id(:local, theme)}
                />
              </div>
              <div class="space-y-space-sm">
                <.btn
                  id={trigger_id(:form, theme)}
                  reach={:shared}
                  target="learning session"
                  phx-click="open_confirm"
                  phx-value-id={confirm_id(:form, theme)}
                >
                  Start learning session
                </.btn>
                <.execute_confirm
                  id={confirm_id(:form, theme)}
                  open={MapSet.member?(@open_confirms, confirm_id(:form, theme))}
                  reach={:shared}
                  verb="Start session"
                  target="learning session"
                  consequence="Saves a session and its goals to Atlas; the agents it dispatches add candidates to the review queue."
                  on_confirm={JS.push("confirm", value: %{id: confirm_id(:form, theme)})}
                  on_cancel={JS.push("close_confirm", value: %{id: confirm_id(:form, theme)})}
                  trigger_id={trigger_id(:form, theme)}
                >
                  <:fields>
                    <.input name="topic" label="Topic" value="" />
                  </:fields>
                </.execute_confirm>
              </div>
            </div>
          </.both>
        </.section>

        <.section title="Verdict">
          <.both :let={_theme}>
            <div class="space-y-space-sm">
              <div class="flex flex-wrap items-center gap-space-sm">
                <Diff.verdict :for={status <- ["pass", "fail", "error", "pending"]} status={status} />
              </div>
              <div class="flex flex-wrap items-center gap-space-sm">
                <Diff.verdict status="pending" label="gate not set" />
                <span class="text-caption text-ink-muted">the not-run verdict, worded for what has not run</span>
              </div>
              <Diff.coverage checked={7} total={143} />
              <div><Diff.coverage checked={3} total={3} /></div>
              <div><Diff.coverage checked={2} /></div>
            </div>
          </.both>
        </.section>

        <.section title="Empty states">
          <.both :let={_theme}>
            <div class="space-y-space-sm">
              <.empty_panel kind={:observed_clean}>
                A sample: 11 values recorded from your input; none came from a fallback, a
                stand-in or an unreadable source.
              </.empty_panel>
              <.empty_panel kind={:not_observable}>
                A sample: this call did its work inside <span class="text-ref">Brain.Memory.Store</span>, another
                process. Provenance is collected per process, so nothing here could be recorded.
              </.empty_panel>
              <.empty_panel
                kind={:not_instrumented}
                entries={@unrecorded.provenance}
                expected={["Brain.Analysis.SemanticChunker", "Brain.ML.Tokenizer"]}
              />
              <.empty_panel kind={:could_not_ask}>
                A sample: MicroClassifiers exited before answering <span class="text-ref">stale/0</span>.
                Whether any model is stale is unknown.
                <:action>
                  <.btn size={:sm} variant={:outline}>Ask again</.btn>
                </:action>
              </.empty_panel>
              <.empty_panel kind={:plain} words="No saved cases for this subsystem yet" />
            </div>
          </.both>
        </.section>

        <.section title="Regression gate">
          <.both :let={theme}>
            <div class="space-y-space-lg">
              <div class="space-y-space-sm">
                <div
                  :for={{key, name} <- @gate_sample_order}
                  id={"design-gate-#{key}-#{theme}"}
                  class="flex flex-wrap items-center gap-space-md"
                >
                  <span class="w-40 shrink-0 text-ref text-ink-muted">{name}</span>
                  <.gate_verdict verdict={@gate_verdicts[key]} />
                </div>
              </div>
              <div>
                <h4 class="mb-space-sm text-label text-ink-muted">In a stat tile's verdict slot</h4>
                <div class="grid grid-cols-2 gap-space-sm">
                  <div id={"design-kpi-gate-pass-#{theme}"} class="min-w-0">
                    <.stat_kpi
                      label="Macro F1"
                      value="49.6%"
                      sublabel="macro-F1 · 4870 examples"
                      icon="hero-chart-bar"
                    >
                      <:verdict><.gate_verdict verdict={@gate_verdicts.pass} /></:verdict>
                    </.stat_kpi>
                  </div>
                  <div id={"design-kpi-gate-not-set-#{theme}"} class="min-w-0">
                    <.stat_kpi
                      label="Macro F1"
                      value="49.6%"
                      sublabel="macro-F1 · 4870 examples"
                      icon="hero-chart-bar"
                    >
                      <:verdict><.gate_verdict verdict={@gate_verdicts.not_set} /></:verdict>
                    </.stat_kpi>
                  </div>
                </div>
              </div>
              <p class="text-caption text-ink-muted">
                Samples: each is <span class="text-ref">Brain.Evaluation.Gate.verdict/3</span> on a sample
                baseline and result, not an evaluation.
              </p>
            </div>
          </.both>
        </.section>

        <.section title="Pagination">
          <.both :let={theme}>
            <div class="space-y-space-lg">
              <div :for={sample <- @page_sample_order}>
                <h4 class="mb-space-sm text-label text-ink-muted">{@page_samples[sample].title}</h4>
                <.page_bar
                  page={@page_samples[sample].page}
                  page_size={@page_samples[sample].page_size}
                  total={@page_samples[sample].total}
                  event={"design_page_#{sample}"}
                  label={"#{@page_samples[sample].title}, #{theme}"}
                />
              </div>
            </div>
          </.both>
        </.section>

        <.section title="Availability and status">
          <.both :let={_theme}>
            <div class="space-y-space-md">
              <div class="grid grid-cols-2 gap-space-sm sm:grid-cols-3">
                <div :for={status <- @statuses} class="flex items-center gap-space-sm">
                  <.status_dot status={status} />
                  <span class="text-ref text-ink-muted">{status}</span>
                </div>
              </div>
              <div class="flex items-center gap-space-sm">
                <.status_dot status={:initializing} pulse />
                <span class="text-ref text-ink-muted">initializing, pulse (still under reduced motion)</span>
              </div>
            </div>
          </.both>
        </.section>

        <.section title="Job progress">
          <.both :let={_theme}>
            <div class="space-y-space-md">
              <div class="h-space-sm w-64 rounded-sm bg-progress-track">
                <div class="h-full rounded-sm bg-progress-fill" style="width: 40%" />
              </div>
              <div class="flex flex-wrap gap-space-md">
                <.circular_progress
                  :for={variant <- [:primary, :success, :warning, :error, :info]}
                  value={65}
                  variant={variant}
                  size={:sm}
                >
                  <span class="text-offset text-ink">65</span>
                </.circular_progress>
              </div>
            </div>
          </.both>
        </.section>

        <.section title="Badges">
          <.both :let={_theme}>
            <div class="space-y-space-sm">
              <div :for={size <- [:xs, :sm]} class="flex flex-wrap items-center gap-space-sm">
                <.badge :for={variant <- [:default, :info, :success, :primary, :warning, :error]} variant={variant} size={size}>
                  {variant} {size}
                </.badge>
              </div>
              <div class="flex flex-wrap items-center gap-space-sm">
                <.badge mono>world default</.badge>
                <.badge mono>ChunkFeatures.entity_features/1</.badge>
                <.badge mono>runner.ex:222-281</.badge>
                <span class="text-caption text-ink-muted">mono</span>
              </div>
            </div>
          </.both>
        </.section>

        <.section title="Buttons">
          <.both :let={theme}>
            <div class="space-y-space-sm">
              <div
                :for={variant <- [:primary, :outline, :secondary, :ghost]}
                class="flex flex-wrap items-center gap-space-sm"
              >
                <.btn :for={size <- [:xs, :sm, :md, :lg]} variant={variant} size={size}>
                  {variant} {size}
                </.btn>
                <.btn variant={variant} disabled>disabled</.btn>
                <.icon_btn variant={variant} size={:sm} title="Refresh">
                  <.icon name="hero-arrow-path" class="size-3.5" />
                </.icon_btn>
                <.icon_btn variant={variant} size={:md} title="Refresh">
                  <.icon name="hero-arrow-path" class="size-4" />
                </.icon_btn>
                <.icon_btn variant={variant} size={:lg} title="Refresh">
                  <.icon name="hero-arrow-path" class="size-5" />
                </.icon_btn>
                <.icon_btn variant={variant} title="Refresh (disabled)" disabled>
                  <.icon name="hero-arrow-path" class="size-4" />
                </.icon_btn>
              </div>
              <div class="flex flex-wrap items-center gap-space-md">
                <.btn :for={size <- [:xs, :sm, :md]} variant={:link} size={size} navigate={~p"/design"}>
                  View details {size}
                </.btn>
                <span class="text-caption text-ink-muted">link: navigation, accent</span>
              </div>
              <div class="flex flex-wrap items-center gap-space-sm">
                <.btn icon="hero-play">Run</.btn>
                <.btn busy>Training</.btn>
                <.btn busy variant={:outline} busy_label="Saving">Save case</.btn>
                <.btn busy variant={:ghost} size={:sm}>Re-run</.btn>
                <span class="text-caption text-ink-muted">busy: spinner in the button's own color</span>
              </div>
              <div class="flex flex-wrap items-center gap-space-sm">
                <.btn
                  id={"design-cancel-run-#{theme}"}
                  variant={:outline}
                  size={:sm}
                  reach={:local}
                  target="run pos-0412"
                >
                  Cancel run
                </.btn>
                <span class="text-caption text-ink-muted">a Cancel is outline; this one writes local</span>
              </div>
              <form class="flex flex-wrap items-center gap-space-sm" id={"design-promote-#{theme}"}>
                <.btn size={:sm} name="target" value="test" type="button">Test</.btn>
                <.btn size={:sm} name="target" value="production" type="button">Production</.btn>
                <span class="text-caption text-ink-muted">name and value, for a form action</span>
              </form>
            </div>
          </.both>
        </.section>

        <.section title="Tabs, toggle and inputs">
          <.both :let={theme}>
            <div class="space-y-space-md">
              <.tabs>
                <.tab active>Selected</.tab>
                <.tab>Other</.tab>
                <.tab>Another</.tab>
              </.tabs>
              <div class="flex flex-wrap gap-space-lg">
                <.toggle checked label="on md" />
                <.toggle label="off md" />
                <.toggle checked size={:sm} label="on sm" />
                <.toggle size={:sm} label="off sm" />
              </div>
              <.text_input name={"design-text-#{theme}"} value="a value" />
              <.text_input name={"design-placeholder-#{theme}"} placeholder="placeholder text" />
              <.text_input name={"design-disabled-#{theme}"} value="disabled" disabled />
              <.input name={"design-input-#{theme}"} value="core input" label="Label" />
              <.input
                name={"design-input-error-#{theme}"}
                value="core input with an error"
                label="With an error"
                errors={["is invalid"]}
              />
              <.input
                type="select"
                name={"design-select-#{theme}"}
                value="b"
                label="Select"
                options={[{"Option a", "a"}, {"Option b", "b"}]}
              />
              <.input type="textarea" name={"design-textarea-#{theme}"} value="a textarea" label="Textarea" />
              <.input type="checkbox" name={"design-checkbox-#{theme}"} value={true} label="Checkbox" />
              <div>
                <h4 class="mb-space-sm text-label text-ink-muted">Inline, size :sm, in table rows</h4>
                <.table
                  id={"design-inline-table-#{theme}"}
                  rows={[%{id: 1, key: "access_token", errors: []}, %{id: 2, key: "refresh_token", errors: ["must not be empty"]}]}
                >
                  <:col :let={row} label="Credential"><span class="text-ref">{row.key}</span></:col>
                  <:col :let={row} label="Value">
                    <.input
                      name={"design-inline-#{row.key}-#{theme}"}
                      label={"#{row.key} value"}
                      placeholder="paste a token"
                      value=""
                      inline
                      size={:sm}
                      errors={row.errors}
                    />
                  </:col>
                  <:action>
                    <.btn size={:xs}>Save</.btn>
                  </:action>
                </.table>
              </div>
            </div>
          </.both>
        </.section>

        <.section title="Alerts, flash, cards and headers">
          <.both :let={theme}>
            <div class="space-y-space-md">
              <.alert :for={variant <- [:info, :success, :warning, :error]} variant={variant}>
                alert {variant}
              </.alert>
              <div class="relative h-40 contain-paint rounded-md border border-dashed border-border-strong">
                <.flash id={"design-flash-info-#{theme}"} kind={:info} title="Flash info">
                  An info flash, as it floats over content.
                </.flash>
              </div>
              <div class="relative h-40 contain-paint rounded-md border border-dashed border-border-strong">
                <.flash id={"design-flash-error-#{theme}"} kind={:error} title="Flash error">
                  An error flash.
                </.flash>
              </div>
              <div class="grid grid-cols-2 gap-space-sm">
                <.stat_kpi
                  :for={variant <- [:default, :info, :success, :primary, :warning, :error]}
                  label={"stat #{variant}"}
                  value="42"
                  sublabel="a sublabel"
                  icon="hero-check-badge"
                  variant={variant}
                />
              </div>
              <.card>
                <.card_body class="space-y-space-sm">
                  <.section_header icon="hero-cpu-chip">
                    Section header
                    <:actions><.btn size={:xs} variant={:outline}>Action</.btn></:actions>
                  </.section_header>
                  <.divider />
                  <p class="text-body text-ink">A card body, density :regular.</p>
                </.card_body>
              </.card>
              <.card>
                <.card_body density={:compact}>
                  <p class="text-body text-ink">A card body, density :compact.</p>
                </.card_body>
              </.card>
              <.card>
                <.card_body density={:flush}>
                  <.table
                    id={"design-flush-table-#{theme}"}
                    rows={[%{id: 1, path: "speech_act.category", value: "directive"}, %{id: 2, path: "score", value: "0.83"}]}
                  >
                    <:col :let={row} label="Path"><span class="text-ref">{row.path}</span></:col>
                    <:col :let={row} label="Value"><span class="text-value">{row.value}</span></:col>
                  </.table>
                </.card_body>
              </.card>
              <p class="text-caption text-ink-muted">density :flush: the table runs to the panel edge.</p>
              <.header>
                Core header
                <:subtitle>With a subtitle</:subtitle>
                <:actions><.btn size={:sm}>Act</.btn></:actions>
              </.header>
            </div>
          </.both>
        </.section>

        <.section title="Table and list">
          <.both :let={theme}>
            <div class="space-y-space-md">
              <div class="flex flex-wrap items-center justify-between gap-space-sm">
                <span class="text-subheading text-ink">Intent examples</span>
                <.reach_badge reach={:local} target="data/intents.json" />
              </div>
              <.table
                id={"design-table-#{theme}"}
                rows={[
                  %{id: 1, path: "speech_act.category", value: "directive", origin: :computed},
                  %{id: 2, path: "config.domain_lemmas", value: "%{}", origin: :default},
                  %{id: 3, path: "score", value: "0.83", origin: :computed},
                  %{id: 4, path: "config.gazetteer", value: "nil", origin: :unavailable},
                  %{id: 5, path: "text", value: "turn on the kitchen light", origin: :computed}
                ]}
                row_class={&Runner.origin_style(&1.origin).row}
              >
                <:col :let={row} label="Path"><span class="text-ref">{row.path}</span></:col>
                <:col :let={row} label="Value"><span class="text-value">{row.value}</span></:col>
                <:col :let={row} label="Came from"><Runner.origin origin={row.origin} /></:col>
                <:action>
                  <.icon_btn size={:sm} title="Edit record"><.icon name="hero-pencil" /></.icon_btn>
                  <.icon_btn size={:sm} title="Delete record"><.icon name="hero-trash" /></.icon_btn>
                </:action>
              </.table>
              <.page_bar
                page={@sample_page}
                page_size={50}
                total={1204}
                matching={312}
                event="design_page"
                label={"Sample pages, #{theme}"}
              />
              <.list>
                <:item title="World">default</:item>
                <:item title="Language">en</:item>
              </.list>
            </div>
          </.both>
        </.section>

        <.section title="Harness, rendered from sample calls">
          <.both :let={_theme}>
            <div class="space-y-space-xl">
              <Runner.result_panel
                outcome={@returned}
                label="ChatWeb.DesignLive.sample_with_every_origin/1"
                world_id={@current_world_id}
                expected_sources={["ChatWeb.DesignLive.sample_with_every_origin"]}
                comparison={@failing}
              />
              <Runner.result_panel
                outcome={@unrecorded}
                label="ChatWeb.DesignLive.sample_without_provenance/1"
                world_id={@current_world_id}
                expected_sources={["ChatWeb.DesignLive.sample_without_provenance"]}
              />
              <Runner.result_panel
                outcome={@raised}
                label="ChatWeb.DesignLive.sample_that_raises/1"
                world_id={@current_world_id}
                expected_sources={["ChatWeb.DesignLive.sample_that_raises"]}
              />
              <.card>
                <.card_body>
                  <Diff.diff comparison={@passing} />
                </.card_body>
              </.card>
              <.card>
                <.card_body class="space-y-space-sm">
                  <h3 class="text-heading text-ink">Raw term with stand-in paths, each by its origin</h3>
                  <Diff.term
                    term={
                      Comparison.normalize(%{
                        entity: %{familiarity: 0.5, count: 2},
                        memory: %{context: []},
                        config: %{gazetteer: nil, threshold: 0.4},
                        raw: {:ok, []}
                      })
                    }
                    stand_ins={[
                      %{path: ["entity", "familiarity"], origin: :default},
                      %{path: ["memory", "context"], origin: :absent},
                      %{path: ["config", "gazetteer"], origin: :unavailable},
                      %{path: ["config", "threshold"], origin: :default},
                      %{path: ["config", "threshold"], origin: :absent}
                    ]}
                  />
                </.card_body>
              </.card>
              <Runner.case_list cases={@cases} />
              <Runner.case_list cases={[]} />
            </div>
          </.both>
        </.section>
      </div>
    </.app_shell>
    """
  end

  attr :title, :string, required: true
  slot :inner_block, required: true

  defp section(assigns) do
    anchor = assigns.title |> String.downcase() |> String.replace(~r/[^a-z0-9]+/, "-")
    assigns = assign(assigns, :anchor, anchor)

    ~H"""
    <section id={@anchor} class="space-y-space-sm">
      <h2 class="text-heading text-ink">{@title}</h2>
      {render_slot(@inner_block)}
    </section>
    """
  end

  slot :inner_block, required: true

  defp both(assigns) do
    ~H"""
    <div class="grid gap-space-lg xl:grid-cols-2">
      <div
        :for={theme <- ["light", "dark"]}
        data-theme={theme}
        class="min-w-0 rounded-md border border-border-strong bg-ground p-space-lg text-ink"
      >
        <div class="mb-space-sm text-label text-ink-muted">{theme}</div>
        {render_slot(@inner_block, theme)}
      </div>
    </div>
    """
  end
end
