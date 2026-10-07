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
  `Atlas.Schemas.VerificationCase` structs. The samples are labeled as samples;
  nothing on this page is a verification result.
  """

  use ChatWeb, :live_view

  import ChatWeb.AppShell

  alias Atlas.Schemas.VerificationCase
  alias Atlas.Verification.Comparison
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

  @impl true
  def mount(_params, _session, socket) do
    returned = Runner.run(&sample_with_every_origin/1, "turn on the kitchen light")
    raised = Runner.run(&sample_that_raises/1, "turn on the kitchen light")

    {:ok,
     socket
     |> assign(:page_title, "Design Language")
     |> assign(:returned, returned)
     |> assign(:raised, raised)
     |> assign(:failing, Comparison.compare(%{intent: "greeting", score: 0.5}, returned.value))
     |> assign(:passing, Comparison.compare(%{intent: "directive"}, returned.value))
     |> assign(:cases, sample_cases())}
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

  defp sample_cases do
    run_at = ~U[2026-10-06 12:00:00.000000Z]

    for {status, name, last_run_at} <- [
          {"pass", "a plain directive", run_at},
          {"fail", "a greeting read as a directive", run_at},
          {"error", "an input that raises", run_at},
          {"pending", "a case never run", nil}
        ] do
      %VerificationCase{
        subsystem: "speech_act",
        name: name,
        world_id: "default",
        status: status,
        last_run_at: last_run_at
      }
    end
  end

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
            <div class="space-y-space-md">
              <div class="flex items-center gap-space-md">
                <span class="w-36 text-ref text-ink-muted">score-calibrated</span>
                <div class="h-space-sm w-40 rounded-sm bg-score-track">
                  <div class="h-full rounded-sm bg-score-calibrated" style="width: 72%" />
                </div>
                <span class="text-value text-ink">0.72 · model confidence</span>
              </div>
              <div class="flex items-center gap-space-md">
                <span class="w-36 text-ref text-ink-muted">score-relative</span>
                <div class="h-space-sm w-40 rounded-sm bg-score-track">
                  <div class="h-full rounded-sm border border-score-calibrated bg-score-relative" style="width: 12%" />
                </div>
                <span class="text-value text-ink">0.12 · softmax share</span>
              </div>
              <div class="flex items-center gap-space-md">
                <span class="w-36 text-ref text-ink-muted">score-heuristic</span>
                <div class="h-space-sm w-40 rounded-sm bg-score-track">
                  <div class="h-full rounded-sm border border-score-heuristic" style="width: 58%" />
                </div>
                <span class="text-value text-ink">0.58 · weighted vote</span>
              </div>
              <div class="flex items-center gap-space-md">
                <span class="w-36 text-ref text-ink-muted">score-unbounded</span>
                <div class="relative h-space-sm w-40 border-b border-score-axis">
                  <span class="absolute -bottom-1 size-2 rounded-pip bg-score-unbounded" style="left: 35%" />
                </div>
                <span class="text-value text-ink">2.41 · hyperbolic distance</span>
              </div>
              <div class="flex items-center gap-space-md">
                <span class="w-36 text-ref text-ink-muted">score-count</span>
                <span class="text-value text-score-count">17</span>
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
          <.both :let={_theme}>
            <div class="space-y-space-md">
              <div class="flex flex-wrap items-center gap-space-sm">
                <.btn>Run</.btn>
                <span class="text-caption text-ink-muted">read-only: no badge</span>
              </div>
              <div class="flex flex-wrap items-center gap-space-sm">
                <.btn>Retrain</.btn>
                <span class="inline-flex items-center gap-space-xs rounded-sm border border-reach-local px-space-xs text-caption font-semibold text-reach-local">
                  <.icon name="hero-circle-stack-micro" class="size-3" /> writes local
                </span>
              </div>
              <div class="flex flex-wrap items-center gap-space-md rounded-md border border-border-strong bg-surface-raised p-space-md shadow-overlay">
                <span class="inline-flex items-center gap-space-xs rounded-sm border border-reach-shared px-space-xs text-caption font-semibold text-reach-shared">
                  <.icon name="hero-share-micro" class="size-3" /> writes shared
                </span>
                <button class="h-control-md rounded-md bg-reach-shared-confirm px-space-md text-body font-semibold text-on-reach-shared outline-mark outline-reach-shared">
                  Confirm save
                </button>
              </div>
              <div class="flex flex-wrap items-center gap-space-md rounded-md border border-border-strong bg-surface-raised p-space-md shadow-overlay">
                <span class="inline-flex items-center gap-space-xs rounded-sm border border-reach-device px-space-xs text-caption font-semibold text-reach-device">
                  <.icon name="hero-bolt-micro" class="size-3" /> actuates light.kitchen
                </span>
                <button class="h-control-md rounded-md bg-reach-device-confirm px-space-md text-body font-semibold text-on-reach-device outline-mark outline-reach-device">
                  Confirm turn on
                </button>
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
              <Diff.coverage checked={7} total={143} />
              <div><Diff.coverage checked={3} total={3} /></div>
            </div>
          </.both>
        </.section>

        <.section title="Availability and status">
          <.both :let={_theme}>
            <div class="grid grid-cols-2 gap-space-sm sm:grid-cols-3">
              <div :for={status <- @statuses} class="flex items-center gap-space-sm">
                <.status_dot status={status} />
                <span class="text-ref text-ink-muted">{status}</span>
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
            </div>
          </.both>
        </.section>

        <.section title="Buttons">
          <.both :let={_theme}>
            <div class="space-y-space-sm">
              <div
                :for={variant <- [:primary, :outline, :secondary, :ghost, :danger]}
                class="flex flex-wrap items-center gap-space-sm"
              >
                <.btn :for={size <- [:xs, :sm, :md, :lg]} variant={variant} size={size}>
                  {variant} {size}
                </.btn>
                <.btn variant={variant} disabled>disabled</.btn>
                <.icon_btn :for={size <- [:sm, :md, :lg]} variant={variant} size={size} title="Refresh">
                  <.icon name="hero-arrow-path" class="size-4" />
                </.icon_btn>
              </div>
              <div class="flex flex-wrap items-center gap-space-sm">
                <.button variant="primary">core button primary</.button>
                <.button>core button default</.button>
              </div>
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
                  <p class="text-body text-ink">A card body.</p>
                </.card_body>
              </.card>
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
              <.table
                id={"design-table-#{theme}"}
                rows={[
                  %{id: 1, path: "speech_act.category", value: "directive"},
                  %{id: 2, path: "entities.0.type", value: "device"},
                  %{id: 3, path: "score", value: "0.83"},
                  %{id: 4, path: "text", value: "turn on the kitchen light"}
                ]}
              >
                <:col :let={row} label="Path"><span class="text-ref">{row.path}</span></:col>
                <:col :let={row} label="Value"><span class="text-value">{row.value}</span></:col>
              </.table>
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
                comparison={@failing}
              />
              <Runner.result_panel outcome={@raised} label="ChatWeb.DesignLive.sample_that_raises/1" />
              <.card>
                <.card_body>
                  <Diff.diff comparison={@passing} />
                </.card_body>
              </.card>
              <.card>
                <.card_body class="space-y-space-sm">
                  <h3 class="text-heading text-ink">Raw term with a stand-in path</h3>
                  <Diff.term
                    term={Comparison.normalize(%{entity: %{familiarity: 0.5, count: 2}, raw: {:ok, []}})}
                    defaulted={[["entity", "familiarity"]]}
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
    ~H"""
    <section class="space-y-space-sm">
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
