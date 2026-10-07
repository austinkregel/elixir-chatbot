defmodule ChatWeb.UITest do
  @moduledoc """
  The shared components in `ChatWeb.UI`, rendered, including every way each
  one refuses to render a state it has no treatment for.
  """
  use ExUnit.Case, async: true

  import Phoenix.Component, only: [sigil_H: 2]
  import Phoenix.LiveViewTest

  alias ChatWeb.UI
  alias Phoenix.LiveView.JS

  defp slot(text), do: [%{__slot__: :inner_block, inner_block: fn _, _ -> text end}]

  defp render_btn(attrs) do
    render_component(&UI.btn/1, Map.merge(%{inner_block: slot("Go")}, Map.new(attrs)))
  end

  describe "btn/1" do
    test "a primary action is a button with the primary fill" do
      html = render_btn(variant: :primary)

      assert html =~ "<button"
      assert html =~ "bg-primary"
      assert html =~ "text-on-primary"
    end

    test "the :danger variant is retired and raises" do
      assert_raise ArgumentError, ~r/no treatment for :danger/, fn -> render_btn(variant: :danger) end
    end

    test "a link renders <a> in accent, with no height or padding" do
      html = render_btn(variant: :link, navigate: "/design")

      assert html =~ "<a"
      assert html =~ ~s(href="/design")
      assert html =~ "text-accent"
      assert html =~ "underline"
      refute html =~ "h-control"
      refute html =~ "<button"
    end

    test "patch and href also render a link" do
      assert render_btn(variant: :link, patch: "/design?x=1") =~ ~s(data-phx-link="patch")
      assert render_btn(variant: :link, href: "https://example.com") =~ ~s(href="https://example.com")
    end

    test "a destination on any action variant raises" do
      for variant <- [:primary, :outline, :secondary, :ghost] do
        assert_raise ArgumentError, ~r/accepted only with variant :link/, fn ->
          render_btn(variant: variant, navigate: "/design")
        end
      end
    end

    test "a link with no destination raises" do
      assert_raise ArgumentError, ~r/needs navigate, patch or href/, fn -> render_btn(variant: :link) end
    end

    test "a link takes no disabled, busy or reach" do
      assert_raise ArgumentError, ~r/takes no disabled/, fn ->
        render_btn(variant: :link, navigate: "/", disabled: true)
      end

      assert_raise ArgumentError, ~r/takes no busy/, fn ->
        render_btn(variant: :link, navigate: "/", busy: true)
      end

      assert_raise ArgumentError, ~r/has no reach/, fn ->
        render_btn(variant: :link, navigate: "/", reach: :local, id: "x")
      end
    end

    test "name, value and form pass through" do
      html = render_btn(name: "target", value: "test", form: "promote-form", type: "submit")

      assert html =~ ~s(name="target")
      assert html =~ ~s(value="test")
      assert html =~ ~s(form="promote-form")
      assert html =~ ~s(type="submit")
    end

    test "class adds to the base classes instead of replacing them" do
      html = render_btn(class: "w-full")

      assert html =~ "w-full"
      assert html =~ "bg-primary"
      assert html =~ "rounded-md"
    end

    test "small ghost and small link sizes" do
      ghost = render_btn(variant: :ghost, size: :xs)
      assert ghost =~ "h-control-sm"
      assert ghost =~ "text-caption"

      link = render_btn(variant: :link, size: :sm, navigate: "/")
      assert link =~ "text-body-dense"
      refute link =~ "h-control-sm"
    end

    test "an unknown size raises" do
      assert_raise ArgumentError, ~r/no treatment for :huge/, fn -> render_btn(size: :huge) end
    end

    test "busy disables, sets aria-busy, and draws a spinner in currentColor held still under reduced motion" do
      html = render_btn(busy: true, icon: "hero-play", busy_label: "Training")

      assert html =~ "disabled"
      assert html =~ ~s(aria-busy="true")
      assert html =~ "data-spinner"
      assert html =~ ~s(stroke="currentColor")
      assert html =~ "motion-safe:animate-spin"
      refute html =~ ~r/class="[^"]*(?<!:)animate-spin/
      assert html =~ "motion-reduce:inline"
      assert html =~ "Training"
      refute html =~ "hero-play"
    end

    test "busy_label defaults to Working" do
      assert render_btn(busy: true) =~ "Working"
    end

    test "the icon shows when not busy" do
      assert render_btn(icon: "hero-play") =~ "hero-play"
    end

    test "a writing reach renders its badge as the button's description" do
      html = render_btn(id: "rebuild", reach: :local, target: "micro/intent.term")

      assert html =~ ~s(aria-describedby="rebuild-reach")
      assert html =~ ~s(id="rebuild-reach")
      assert html =~ "writes local"
      assert html =~ "micro/intent.term"
    end

    test "the trigger never wears the reach ring" do
      html = render_btn(id: "approve", reach: :shared, target: "candidate 41")

      refute html =~ "reach-ring"
      refute html =~ "outline-reach"
    end

    test "read-only renders no badge" do
      refute render_btn([]) =~ "data-reach"
    end

    test "a writing reach without an id raises" do
      assert_raise ArgumentError, ~r/needs an id/, fn -> render_btn(reach: :shared, target: "x") end
    end

    test "a device reach without a target raises" do
      assert_raise ArgumentError, ~r/needs its target/, fn -> render_btn(id: "on", reach: :device) end
    end

    test "an unknown reach raises" do
      assert_raise ArgumentError, ~r/no treatment for reach :cosmic/, fn ->
        render_btn(id: "x", reach: :cosmic)
      end
    end
  end

  describe "icon_btn/1" do
    defp render_icon_btn(attrs) do
      render_component(
        &UI.icon_btn/1,
        Map.merge(%{inner_block: slot("i"), title: "Delete record"}, Map.new(attrs))
      )
    end

    test "the title is its accessible name" do
      html = render_icon_btn([])

      assert html =~ ~s(title="Delete record")
      assert html =~ ~s(aria-label="Delete record")
    end

    test "disabled renders disabled at half opacity" do
      html = render_icon_btn(disabled: true)

      assert html =~ "disabled"
      assert html =~ "disabled:opacity-50"
    end

    test ":lg is larger than :md" do
      assert render_icon_btn(size: :md) =~ "size-control-md"
      assert render_icon_btn(size: :lg) =~ "size-row-relaxed"
      refute render_icon_btn(size: :lg) =~ "size-control-md"
    end

    test "a link variant is not an icon button variant" do
      assert_raise ArgumentError, ~r/no treatment for :link/, fn -> render_icon_btn(variant: :link) end
    end

    test "with no title it does not render" do
      assert_raise KeyError, fn ->
        render_component(&UI.icon_btn/1, %{inner_block: slot("i")})
      end
    end
  end

  describe "reach_badge/1" do
    test "each reach has its words, glyph and color" do
      for {reach, words, icon, color} <- [
            {:local, "writes local", "hero-circle-stack-micro", "text-reach-local"},
            {:shared, "writes shared", "hero-share-micro", "text-reach-shared"},
            {:device, "actuates", "hero-bolt-micro", "text-reach-device"}
          ] do
        html = render_component(&UI.reach_badge/1, reach: reach, target: "t")

        assert html =~ words
        assert html =~ icon
        assert html =~ color
        assert html =~ ~s(data-reach="#{reach}")
      end
    end

    test "the target is set in the ref type" do
      html = render_component(&UI.reach_badge/1, reach: :shared, target: "candidate 41")

      assert html =~ ~r/class="text-ref[^"]*">candidate 41/
    end

    test "a local badge may omit its target" do
      assert render_component(&UI.reach_badge/1, reach: :local, target: nil) =~ "writes local"
    end

    test "a device badge without a target raises" do
      assert_raise ArgumentError, ~r/needs its target/, fn ->
        render_component(&UI.reach_badge/1, reach: :device, target: nil)
      end
    end

    test "there is no read-only badge" do
      assert_raise ArgumentError, ~r/no treatment for :read/, fn ->
        render_component(&UI.reach_badge/1, reach: :read, target: nil)
      end
    end
  end

  describe "execute_confirm/1" do
    defp confirm(attrs) do
      Map.merge(
        %{
          id: "c",
          open: true,
          reach: :shared,
          verb: "Reject",
          target: "candidate 41",
          consequence: "Marks this candidate rejected.",
          on_confirm: "reject",
          on_cancel: "close",
          trigger_id: "trigger"
        },
        Map.new(attrs)
      )
    end

    defp render_confirm(attrs), do: render_component(&UI.execute_confirm/1, confirm(attrs))

    test "closed, it renders nothing" do
      refute render_confirm(open: false) =~ "dialog"
    end

    test "open, it is a raised panel with the reach badge first, then the consequence" do
      html = render_confirm([])

      assert html =~ ~s(role="dialog")
      assert html =~ "bg-surface-raised"
      assert html =~ "shadow-overlay"
      assert html =~ "border-border-strong"

      {badge_at, _} = :binary.match(html, "writes shared")
      {consequence_at, _} = :binary.match(html, "Marks this candidate rejected.")
      assert badge_at < consequence_at
      assert html =~ "candidate 41"
    end

    test "Cancel comes first, is outline, and takes focus when the panel mounts" do
      html = render_confirm([])

      {cancel_at, _} = :binary.match(html, ~s(id="c-cancel"))
      {confirm_at, _} = :binary.match(html, ~s(id="c-confirm"))
      assert cancel_at < confirm_at

      [cancel] = Regex.run(~r/<button[^>]*id="c-cancel"[^>]*>/, html)
      assert cancel =~ "border-primary"
      assert cancel =~ "phx-mounted"
      assert cancel =~ "focus"
    end

    test "Cancel and Escape run on_cancel and return focus to the trigger" do
      html = render_confirm([])

      [cancel] = Regex.run(~r/<button[^>]*id="c-cancel"[^>]*>/, html)
      assert cancel =~ "close"
      assert cancel =~ "#trigger"

      assert html =~ ~s(phx-key="Escape")
      assert html =~ "phx-window-keydown"
    end

    test "the confirm button is primary, labeled with the verb" do
      html = render_confirm([])

      [confirm] = Regex.run(~r/<button[^>]*id="c-confirm"[^>]*>/, html)
      assert confirm =~ "bg-primary"
      assert confirm =~ "reject"
      assert html =~ "Reject"
    end

    test "a shared or device confirm sits in a reach ring, not an outline on the button" do
      shared = render_confirm(reach: :shared)
      assert shared =~ ~r/class="reach-ring border-reach-shared"/

      device = render_confirm(reach: :device, target: "light.kitchen")
      assert device =~ ~r/class="reach-ring border-reach-device"/

      [confirm] = Regex.run(~r/<button[^>]*id="c-confirm"[^>]*>/, device)
      refute confirm =~ "outline-reach"
    end

    test "a local removal confirms with no ring" do
      html = render_confirm(reach: :local, removes: true, verb: "Unload world", target: "world x")

      assert html =~ "writes local"
      refute html =~ "reach-ring"
    end

    test "a local write that removes nothing does not confirm" do
      assert_raise ArgumentError, ~r/confirms only when it removes/, fn -> render_confirm(reach: :local) end
    end

    test "a read-only action never confirms" do
      assert_raise ArgumentError, ~r/no treatment for reach :read/, fn -> render_confirm(reach: :read) end
    end

    test "a device shows its current state" do
      html = render_confirm(reach: :device, target: "light.kitchen", current_state: "off, read 12:04")

      assert html =~ "current state: off, read 12:04"
    end

    test "current_state on a non-device reach raises" do
      assert_raise ArgumentError, ~r/current_state is a device's state/, fn ->
        render_confirm(current_state: "off")
      end
    end

    test "a failed confirmation shows its failure in the panel" do
      html = render_confirm(error: "The review store refused the write: timeout.")

      assert html =~ ~s(role="dialog")
      assert html =~ "The review store refused the write: timeout."
      assert html =~ ~s(role="alert")
    end

    test "events may be JS commands" do
      html = render_confirm(on_confirm: JS.push("reject", value: %{id: 41}), on_cancel: JS.push("close"))

      assert html =~ "reject"
    end

    test "an event that is neither a name nor a JS command raises" do
      assert_raise ArgumentError, ~r/event name or a Phoenix.LiveView.JS/, fn ->
        render_confirm(on_cancel: :close)
      end
    end

    test "with fields, the panel is a form whose confirm submits" do
      assigns = %{}

      html =
        rendered_to_string(~H"""
        <UI.execute_confirm
          id="c"
          open
          reach={:shared}
          verb="Start session"
          target="learning session"
          consequence="Saves a session."
          on_confirm="start"
          on_cancel="close"
          trigger_id="trigger"
        >
          <:fields><input name="topic" /></:fields>
        </UI.execute_confirm>
        """)

      assert html =~ ~s(phx-submit="start")
      assert html =~ ~s(name="topic")
      [confirm] = Regex.run(~r/<button[^>]*id="c-confirm"[^>]*>/, html)
      assert confirm =~ ~s(type="submit")
    end
  end

  describe "empty_panel/1" do
    test "observed and clean: filled circle, neutral edge" do
      html = render_component(&UI.empty_panel/1, kind: :observed_clean, inner_block: slot("11 values"))

      assert html =~ "Observed: nothing stood in"
      assert html =~ ~s(data-mark="filled_circle")
      assert html =~ "border-dashed"
    end

    test "not observable: dotted square, with its detail" do
      html = render_component(&UI.empty_panel/1, kind: :not_observable, inner_block: slot("Memory.Store"))

      assert html =~ "Not observable"
      assert html =~ ~s(data-mark="dotted_square")
      assert html =~ "Memory.Store"
    end

    test "not instrumented lists what Brain.Provenance.missing_sources/2 names" do
      entries = [%{path: ["a"], value: 1, origin: :computed, source: "Brain.ML.Tokenizer.tokenize/1", meta: %{}}]

      html =
        render_component(&UI.empty_panel/1,
          kind: :not_instrumented,
          entries: entries,
          expected: ["Brain.Analysis.SemanticChunker", "Brain.ML.Tokenizer"]
        )

      assert html =~ "Not instrumented"
      assert html =~ ~s(data-mark="dashed_circle")
      assert html =~ "Brain.Analysis.SemanticChunker"
      refute html =~ "Brain.ML.Tokenizer<"
    end

    test "not instrumented raises when every expected source reported" do
      entries = [%{path: ["a"], value: 1, origin: :computed, source: "Brain.ML.Tokenizer.tokenize/1", meta: %{}}]

      assert_raise ArgumentError, ~r/every expected source reported/, fn ->
        render_component(&UI.empty_panel/1, kind: :not_instrumented, entries: entries, expected: ["Brain.ML.Tokenizer"])
      end
    end

    test "not instrumented raises without entries and expected" do
      assert_raise ArgumentError, ~r/needs `entries`/, fn ->
        render_component(&UI.empty_panel/1, kind: :not_instrumented)
      end

      assert_raise ArgumentError, ~r/needs `entries`/, fn ->
        render_component(&UI.empty_panel/1, kind: :not_instrumented, entries: [], expected: [])
      end
    end

    test "could not ask is breakage: struck circle on the unavailable wash, solid edge" do
      html = render_component(&UI.empty_panel/1, kind: :could_not_ask, inner_block: slot("exited"))

      assert html =~ "Could not be asked"
      assert html =~ ~s(data-mark="struck_circle")
      assert html =~ "bg-origin-unavailable-wash"
      assert html =~ "border-solid border-origin-unavailable"
    end

    test "not observable and could not ask need their detail" do
      for kind <- [:not_observable, :could_not_ask] do
        assert_raise ArgumentError, ~r/needs its detail/, fn ->
          render_component(&UI.empty_panel/1, kind: kind)
        end
      end
    end

    test "entries on another kind raise" do
      assert_raise ArgumentError, ~r/only :not_instrumented takes entries/, fn ->
        render_component(&UI.empty_panel/1, kind: :observed_clean, entries: [], expected: ["x"])
      end
    end

    test "an unknown kind raises" do
      assert_raise ArgumentError, ~r/no treatment for :blank/, fn ->
        render_component(&UI.empty_panel/1, kind: :blank)
      end
    end

    test "plain: words of its own on the neutral edge, with no mark" do
      html =
        render_component(&UI.empty_panel/1,
          kind: :plain,
          words: "No saved cases for this subsystem yet",
          inner_block: slot("Save one from a run above.")
        )

      assert html =~ "No saved cases for this subsystem yet"
      assert html =~ "Save one from a run above."
      assert html =~ ~s(data-empty="plain")
      assert html =~ "border-dashed border-border-strong bg-surface"
      refute html =~ "data-mark"
      refute html =~ "data-origin"
    end

    test "plain needs no detail" do
      html = render_component(&UI.empty_panel/1, kind: :plain, words: "No records in this source")

      assert html =~ "No records in this source"
    end

    test "plain raises without its words" do
      assert_raise ArgumentError, ~r/:plain needs `words` of its own/, fn ->
        render_component(&UI.empty_panel/1, kind: :plain)
      end
    end

    test "plain raises on blank words" do
      assert_raise ArgumentError, ~r/:plain needs words of its own; got blank words/, fn ->
        render_component(&UI.empty_panel/1, kind: :plain, words: "   ")
      end
    end

    test "the four kinds keep their fixed words: words on them raise" do
      for {kind, detail} <- [observed_clean: [], not_observable: slot("x"), could_not_ask: slot("x")] do
        assert_raise ArgumentError, ~r/has fixed words, so it takes no `words`/, fn ->
          render_component(&UI.empty_panel/1, kind: kind, words: "Nothing here", inner_block: detail)
        end
      end
    end
  end

  describe "stat_kpi/1" do
    test "the verdict slot shows a judgment in the tile, below the value, leaving the value in ink" do
      assigns = %{}

      html =
        rendered_to_string(~H"""
        <UI.stat_kpi label="Failing" value="3" sublabel="cases · default world">
          <:verdict><ChatWeb.Harness.Diff.verdict status="fail" /></:verdict>
        </UI.stat_kpi>
        """)

      assert html =~ "data-kpi-verdict"
      assert html =~ ~s(data-verdict="fail")
      assert html =~ ~r/text-title tabular-nums text-ink">3</

      {value_at, _} = :binary.match(html, ">3<")
      {verdict_at, _} = :binary.match(html, "data-kpi-verdict")
      assert value_at < verdict_at
    end

    test "with no verdict, no verdict region renders" do
      html = render_component(&UI.stat_kpi/1, label: "Cases", value: "12")

      refute html =~ "data-kpi-verdict"
    end
  end

  describe "macro_f1_gate/1" do
    test "with no baseline saved, it is the not-run verdict worded gate not set" do
      assert Brain.Evaluation.Gate.read_baseline() == :not_set

      html = render_component(&UI.macro_f1_gate/1, task: "intent")

      assert html =~ ~s(data-gate="not_set")
      assert html =~ ~s(data-verdict="pending")
      assert html =~ "gate not set"
      refute html =~ "Not run"
    end

    test "a task the gate does not judge raises" do
      assert_raise ArgumentError, ~r/does not judge task "pos"/, fn ->
        render_component(&UI.macro_f1_gate/1, task: "pos")
      end
    end
  end

  describe "gate_verdict/1" do
    alias Brain.Evaluation.Gate

    defp baseline(f1), do: {:ok, %{"intent" => %{"macro_f1" => f1, "accuracy" => 0.5, "diagnostics" => nil}}}

    defp gate(verdict), do: render_component(&UI.gate_verdict/1, verdict: verdict)

    test "a pass shows its mark, the change in points and the allowance" do
      html = gate(Gate.verdict("intent", baseline(0.500), %{"macro_f1" => 0.496}))

      assert html =~ ~s(data-gate="pass")
      assert html =~ ~s(data-verdict="pass")
      assert html =~ "−0.4 pts against baseline · allows 2"
    end

    test "a gain is signed" do
      assert gate(Gate.verdict("intent", baseline(0.5), %{"macro_f1" => 0.512})) =~ "+1.2 pts against baseline"
    end

    test "a regression beyond the allowance is a fail with its mark" do
      html = gate(Gate.verdict("intent", baseline(0.5), %{"macro_f1" => 0.45}))

      assert html =~ ~s(data-verdict="fail")
      assert html =~ "−5.0 pts against baseline · allows 2"
    end

    test "a fail on new failed predictions says how many" do
      base = {:ok, %{"intent" => %{"macro_f1" => 0.5, "diagnostics" => %{"unknown" => 1, "errored" => 0}}}}
      current = %{"macro_f1" => 0.5, "diagnostics" => %{"unknown" => 3, "errored" => 1}}

      html = gate(Gate.verdict("intent", base, current))

      assert html =~ ~s(data-verdict="fail")
      assert html =~ "3 new unknown, errored or not-loaded predictions"
      refute html =~ "canary not measured"
    end

    test "a measured canary with no new failed predictions shows no canary line" do
      diagnostics = %{"unknown" => 2, "errored" => 0}
      base = {:ok, %{"intent" => %{"macro_f1" => 0.5, "diagnostics" => diagnostics}}}

      html = gate(Gate.verdict("intent", base, %{"macro_f1" => 0.5, "diagnostics" => diagnostics}))

      assert html =~ ~s(data-verdict="pass")
      refute html =~ "new unknown, errored or not-loaded predictions"
      refute html =~ "data-gate-canary"
    end

    test "a current result with no diagnostics leaves the canary unmeasured and says where" do
      base = {:ok, %{"intent" => %{"macro_f1" => 0.5, "diagnostics" => %{"unknown" => 1}}}}

      html = gate(Gate.verdict("intent", base, %{"macro_f1" => 0.496}))

      assert html =~ ~s(data-verdict="pass")
      assert html =~ ~s(data-gate-canary="not_measured")
      assert html =~ "error canary not measured · no diagnostics in the latest result"
      refute html =~ "new unknown, errored or not-loaded predictions"
    end

    test "a baseline with no diagnostics leaves the canary unmeasured and says where" do
      html = gate(Gate.verdict("intent", baseline(0.5), %{"macro_f1" => 0.5, "diagnostics" => %{"unknown" => 9}}))

      assert html =~ ~s(data-gate-canary="not_measured")
      assert html =~ "error canary not measured · no diagnostics in the baseline"
      refute html =~ "the latest result"
      refute html =~ "new unknown, errored or not-loaded predictions"
    end

    test "neither side with diagnostics names both" do
      html = gate(Gate.verdict("intent", baseline(0.5), %{"macro_f1" => 0.45}))

      assert html =~ ~s(data-verdict="fail")
      assert html =~ "error canary not measured · no diagnostics in the baseline or the latest result"
      refute html =~ "new unknown, errored or not-loaded predictions"
    end

    test "a baseline with no current result is a fail that says so" do
      html = gate(Gate.verdict("intent", baseline(0.5), nil))

      assert html =~ ~s(data-verdict="fail")
      assert html =~ "no current result to judge"
      refute html =~ "against baseline"
      refute html =~ "canary not measured"
    end

    test "no baseline is the not-run verdict worded gate not set" do
      html = gate(Gate.verdict("intent", :not_set, nil))

      assert html =~ ~s(data-verdict="pending")
      assert html =~ "gate not set"
    end

    test "an allowance with a fraction of a point keeps it" do
      assert gate(%{Gate.verdict("intent", baseline(0.5), %{"macro_f1" => 0.5}) | allowance: 0.015}) =~
               "allows 1.5"
    end

    test "a verdict with no status the page can show raises" do
      assert_raise ArgumentError, ~r/no treatment for a gate verdict/, fn -> gate(%{status: :maybe}) end
    end
  end

  describe "score_display/1" do
    defp score(attrs), do: render_component(&UI.score_display/1, Map.new(attrs))

    test "the text form is the value and its kind" do
      html = score(kind: :weighted_vote, value: 0.834)

      assert html =~ "0.83"
      assert html =~ "weighted vote"
      refute html =~ "bg-score-track"
    end

    test "every kind has a text form" do
      for attrs <- [
            [kind: :model_confidence, value: 0.7],
            [kind: :softmax_share, value: 0.1],
            [kind: :mapped_confidence, value: 0.7],
            [kind: :weighted_vote, value: 0.7],
            [kind: :reranked_confidence, value: 0.7],
            [kind: :match_confidence, value: 0.7],
            [kind: :completeness, value: 0.7, parts: ["actor"]],
            [kind: :belief_confidence, value: 0.7],
            [kind: :analyzer_activation, value: 0.7],
            [kind: :activation, value: 0.7],
            [kind: :accumulated_confidence, value: 0.7],
            [kind: :margin, value: 0.3],
            [kind: :entropy, value: 0.3],
            [kind: :cosine_similarity, value: -0.3],
            [kind: :distance, value: 2.4],
            [kind: :activation_sum, value: 1.4],
            [kind: :raw_score, value: -3.2, source: "log-probability"],
            [kind: :count, value: 3, noun: "voters"],
            [kind: :unestablished, value: 0.64]
          ] do
        assert score(attrs) =~ ~s(data-score="#{attrs[:kind]}")
      end
    end

    test "the new kinds carry their labels" do
      assert score(kind: :completeness, value: 0.9, parts: ["actor", "object", "verb"]) =~
               "completeness · actor, object, verb"

      assert score(kind: :belief_confidence, value: 0.7) =~ "belief confidence"
      assert score(kind: :match_confidence, value: 0.7) =~ "match confidence"
      assert score(kind: :reranked_confidence, value: 0.7) =~ "graph-reranked confidence"
      assert score(kind: :mapped_confidence, value: 0.7) =~ "mapped confidence"
    end

    test "a calibrated kind draws a solid bar" do
      html = score(kind: :model_confidence, value: 0.72, form: :bar)

      assert html =~ "bg-score-track"
      assert html =~ "bg-score-calibrated"
      assert html =~ "width: 72.0%"
      assert html =~ "model confidence"
    end

    test "a softmax share draws a light bar outlined in the calibrated blue" do
      html = score(kind: :softmax_share, value: 0.12, form: :bar)

      assert html =~ "bg-score-relative"
      assert html =~ "border-score-calibrated"
    end

    test "a heuristic kind draws an outline only" do
      for kind <- [:mapped_confidence, :weighted_vote, :reranked_confidence, :match_confidence, :belief_confidence] do
        html = score(kind: kind, value: 0.5, form: :bar)

        assert html =~ "border-score-heuristic"
        refute html =~ "bg-score-calibrated"
      end
    end

    test "cosine similarity draws a diverging bar about a zero line" do
      positive = score(kind: :cosine_similarity, value: 0.6, form: :bar)
      assert positive =~ "left: 50.0%; width: 30.0%"
      assert positive =~ "border-score-axis"

      negative = score(kind: :cosine_similarity, value: -0.4, form: :bar)
      assert negative =~ "left: 30.0%; width: 20.0%"
    end

    test "a distance is a dot on its own axis, never a bar" do
      html = score(kind: :distance, value: 2.0, axis_max: 8, form: :bar)

      assert html =~ "bg-score-unbounded"
      assert html =~ "left: 25.0%"
      refute html =~ "bg-score-track"
      assert html =~ "axis 0–8.00"
    end

    test "a distance bar needs its axis and must lie on it" do
      assert_raise ArgumentError, ~r/needs a positive `axis_max`/, fn ->
        score(kind: :distance, value: 2.0, form: :bar)
      end

      assert_raise ArgumentError, ~r/beyond its axis_max/, fn ->
        score(kind: :distance, value: 9.0, axis_max: 8, form: :bar)
      end
    end

    test "kinds with no bar raise when asked for one" do
      for attrs <- [
            [kind: :raw_score, value: 1.0, source: "s"],
            [kind: :count, value: 1, noun: "n"],
            [kind: :activation_sum, value: 1.2],
            [kind: :unestablished, value: 0.5]
          ] do
        assert_raise ArgumentError, ~r/has no bar form/, fn -> score([{:form, :bar} | attrs]) end
      end
    end

    test "an unestablished confidence is text, the not-observed mark, and no bar" do
      html = score(kind: :unestablished, value: 0.64, candidates: "model confidence or weighted vote")

      assert html =~ ~s(data-mark="dotted_square")
      assert html =~ "text-origin-unobserved"
      assert html =~ "confidence, kind not established: model confidence or weighted vote"
      refute html =~ "bg-score-track"
    end

    test "an unknown kind raises" do
      assert_raise ArgumentError, ~r/no treatment for :probability/, fn ->
        score(kind: :probability, value: 0.5)
      end
    end

    test "a non-number raises" do
      for value <- [nil, "0.5", :high] do
        assert_raise ArgumentError, ~r/must be a number/, fn -> score(kind: :model_confidence, value: value) end
      end
    end

    test "a value outside its kind's range raises" do
      assert_raise ArgumentError, ~r/lies in 0..1/, fn -> score(kind: :model_confidence, value: 1.2) end
      assert_raise ArgumentError, ~r/lies in 0..1/, fn -> score(kind: :weighted_vote, value: -0.1) end
      assert_raise ArgumentError, ~r/lies in -1..1/, fn -> score(kind: :cosine_similarity, value: -1.5) end
      assert_raise ArgumentError, ~r/lies in 0.5..1/, fn -> score(kind: :completeness, value: 0.2, parts: ["a"]) end
      assert_raise ArgumentError, ~r/lies in 0 upward/, fn -> score(kind: :distance, value: -0.1) end
    end

    test "a count is a non-negative integer in ink, with its noun and optional denominator" do
      html = score(kind: :count, value: 3, of: 12, noun: "voters")

      assert html =~ ~r/text-score-count">\s*3\s*</
      assert html =~ "of 12 voters"

      assert_raise ArgumentError, ~r/non-negative integer/, fn -> score(kind: :count, value: 2.5, noun: "n") end
      assert_raise ArgumentError, ~r/non-negative integer/, fn -> score(kind: :count, value: -1, noun: "n") end
      assert_raise ArgumentError, ~r/names what it counts/, fn -> score(kind: :count, value: 1) end
    end

    test "completeness names its parts and a raw score its source" do
      assert_raise ArgumentError, ~r/names the parts found/, fn -> score(kind: :completeness, value: 0.9) end
      assert_raise ArgumentError, ~r/names the `source`/, fn -> score(kind: :raw_score, value: 1.0) end
    end
  end

  describe "no_confidence/1" do
    test "says no confidence and the method, never a dash" do
      html = render_component(&UI.no_confidence/1, method: "speech act fallback")

      assert html =~ "no confidence · speech act fallback"
      refute html =~ "—"
    end
  end

  describe "card_body/1" do
    test "each density sets its padding" do
      for {density, class} <- [regular: "p-space-lg", compact: "p-space-md", flush: "p-0"] do
        html = render_component(&UI.card_body/1, density: density, inner_block: slot("x"))
        assert html =~ class
      end
    end

    test "an unknown density raises" do
      assert_raise ArgumentError, ~r/no treatment for :roomy/, fn ->
        render_component(&UI.card_body/1, density: :roomy, inner_block: slot("x"))
      end
    end
  end

  describe "badge/1" do
    test "mono sets the ref type at the xs padding, without the semibold weight" do
      html = render_component(&UI.badge/1, mono: true, inner_block: slot("world default"))

      assert html =~ "text-ref"
      assert html =~ "px-space-xs"
      refute html =~ "font-semibold"
      refute html =~ "py-px"
    end

    test "mono combines with :default only" do
      for variant <- [:info, :success, :primary, :warning, :error] do
        assert_raise ArgumentError, ~r/combines with :default only/, fn ->
          render_component(&UI.badge/1, mono: true, variant: variant, inner_block: slot("x"))
        end
      end
    end
  end

  describe "status_dot/1" do
    test "pulse animates only when motion is welcome" do
      html = render_component(&UI.status_dot/1, status: :initializing, pulse: true)

      assert html =~ "motion-safe:animate-pulse"
      refute html =~ ~r/class="[^"]*(?<!:)animate-pulse/
    end
  end

  describe "alert/1" do
    test "only warning and error interrupt with role=alert" do
      for variant <- [:info, :success] do
        refute render_component(&UI.alert/1, variant: variant, inner_block: slot("x")) =~ "role="
      end

      for variant <- [:warning, :error] do
        assert render_component(&UI.alert/1, variant: variant, inner_block: slot("x")) =~ ~s(role="alert")
      end
    end
  end

  describe "page_bar/1" do
    defp page_bar(attrs) do
      render_component(
        &UI.page_bar/1,
        Map.merge(%{page: 2, page_size: 50, total: 1204, event: "page"}, Map.new(attrs))
      )
    end

    test "the current page is the accent segment with aria-current" do
      html = page_bar([])

      [current] = Regex.run(~r/<button[^>]*aria-current="page"[^>]*>/, html)
      assert current =~ "bg-accent"
      assert current =~ "text-on-accent"
      assert current =~ ~s(phx-value-page="2")
      refute current =~ "bg-primary"
    end

    test "it is a navigation landmark, not a tablist" do
      html = page_bar(label: "Records")

      assert html =~ ~s(<nav aria-label="Records")
      refute html =~ "tablist"
    end

    test "it says which rows show, how many exist and how many match" do
      html = page_bar(matching: 312)

      assert html =~ "Showing 51–100 of 1,204"
      assert html =~ "312 match the filter"
    end

    test "Prev is disabled on the first page and Next on the last" do
      assert page_bar(page: 1) =~ ~r/<button[^>]*\sdisabled[\s>][^>]*phx-value-page="0"/
      assert page_bar(page: 25) =~ ~r/<button[^>]*\sdisabled[\s>][^>]*phx-value-page="26"/
      refute page_bar(page: 2) =~ ~r/<button[^>]*\sdisabled[\s>]/
    end

    test "a page outside the range raises" do
      assert_raise ArgumentError, ~r/outside 1..25/, fn -> page_bar(page: 26) end
    end

    test "an empty set raises" do
      assert_raise ArgumentError, ~r/no rows to page/, fn -> page_bar(total: 0, page: 1) end
    end

    # The numbered segments and gaps in order, Prev and Next left out.
    defp window(html) do
      ~r/<button[^>]*phx-value-page="(\d+)"[^>]*>\s*(?:Prev|Next|\d+)\s*<\/button>|data-page-gap/
      |> Regex.scan(html)
      |> Enum.flat_map(fn
        ["data-page-gap" | _] -> [:gap]
        [segment, page | _] -> if segment =~ ~r/>\s*\d+\s*</, do: [String.to_integer(page)], else: []
      end)
    end

    test "a middle page shows the first, the last, itself with its neighbors, and a gap each side" do
      assert window(page_bar(page: 12)) == [1, :gap, 11, 12, 13, :gap, 25]
    end

    test "near the start there is a gap only before the last page" do
      assert window(page_bar(page: 1)) == [1, 2, :gap, 25]
      assert window(page_bar(page: 3)) == [1, 2, 3, 4, :gap, 25]
    end

    test "near the end there is a gap only after the first page" do
      assert window(page_bar(page: 25)) == [1, :gap, 24, 25]
      assert window(page_bar(page: 23)) == [1, :gap, 22, 23, 24, 25]
    end

    test "a single skipped page is still a gap" do
      assert window(page_bar(page: 4)) == [1, :gap, 3, 4, 5, :gap, 25]
    end

    test "few pages show every page and no gap" do
      assert window(page_bar(page: 2, total: 150)) == [1, 2, 3]
      assert window(page_bar(page: 1, total: 10)) == [1]
    end

    test "the window divides the matching rows, not all rows" do
      assert window(page_bar(page: 1, matching: 312)) == [1, 2, :gap, 7]
    end

    test "the current page in the window keeps aria-current and the accent segment" do
      html = page_bar(page: 12)

      assert [[current]] = Regex.scan(~r/<button[^>]*aria-current="page"[^>]*>/, html)
      assert current =~ ~s(phx-value-page="12")
      assert current =~ "bg-accent text-on-accent"
    end

    test "a gap is not a control" do
      html = page_bar(page: 12)

      refute html =~ ~r/<button[^>]*data-page-gap/
      assert html =~ ~r/<span[^>]*data-page-gap[^>]*>…<\/span>/
    end

    test "an explicit pages list replaces the window" do
      assert window(page_bar(page: 12, pages: [10, 11, 12, 13, 14])) == [10, 11, 12, 13, 14]
    end
  end
end
