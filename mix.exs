defmodule ChatBot.Umbrella.MixProject do
  use Mix.Project

  def project do
    [
      apps_path: "apps",
      version: "0.1.0",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      aliases: aliases(),
      releases: releases(),
      listeners: if(Mix.env() == :dev, do: [Phoenix.CodeReloader], else: []),
      name: "chat_bot",
      source_url: "https://github.com/austinkregel/elixir-chatbot",
      docs: docs()
    ]
  end

  def cli do
    [
      preferred_envs: [precommit: :test]
    ]
  end

  # Dependencies listed here are available only for this
  # project and cannot be accessed from applications inside
  # the apps folder.
  #
  # Run "mix help deps" for examples and options.
  defp deps do
    [
      # Code quality
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      # Documentation site (`mix docs`) — see docs()/0 below
      {:ex_doc, "~> 0.40", only: :dev, runtime: false},
      # Shared test dependencies
      {:excoveralls, "~> 0.18", only: :test},
      # .env file loading
      {:dotenvy, "~> 1.1"},
      {:xla, "~> 0.9.0", override: true}
    ]
  end

  # Aliases are shortcuts or tasks specific to the current project.
  defp aliases do
    [
      # Run setup in all child apps
      setup: [
        "cmd mix setup",
        "atlas.setup",
        "atlas.seed",
        "download_speech_act_corpus",
        "download_sentiment_corpus",
        "setup_lexicon",
        "ingest_framing_corpus",
        "gen_micro_data",
        "gen_framing_data --corpus gvfc",
        "gen_lattice_data",
        "train"
      ],

      # Precommit runs format check, Credo, and tests
      precommit: ["format --check-formatted", "credo --strict", "test"],

      # Browsable API docs for all umbrella apps -> doc/index.html
      "docs.open": ["docs", "cmd open doc/index.html"],

      # Test coverage
      "test.coverage": ["coveralls.html"],
      "test.coverage.json": ["coveralls.json"],

      # Training world shortcuts (using mix do --app instead of deprecated cmd --app)
      "world.list": ["do --app world training_world.list"],
      "world.status": ["do --app world training_world.metrics default"],
      "world.setup": ["do --app world training_world.create default --mode=persistent"],
      "world.clear": ["do --app world training_world.destroy default"],
      "world.reset": [
        "do --app world training_world.destroy default",
        "do --app world training_world.create default --mode=persistent"
      ],

      # Atlas database shortcuts
      "atlas.setup": [
        "do --app atlas ecto.create",
        "do --app atlas atlas.bootstrap_age",
        "do --app atlas ecto.migrate"
      ],
      "atlas.reset": ["do --app atlas ecto.drop", "atlas.setup"],
      "atlas.migrate": ["do --app atlas ecto.migrate"],

      # ML training shortcuts (using mix do --app instead of deprecated cmd --app)
      # Master training pipeline - trains ALL models
      train: ["do --app brain train"],
      # Quick training - skip slow optional models
      "train.quick": ["do --app brain train --quick"],
      # Fast TF-IDF only (legacy)
      "train.tfidf": ["do --app brain train_models"],

      # S3 model store
      "models.upload": ["do --app brain models.upload"],
      "models.download": ["do --app brain models.download"]
    ]
  end

  # ExDoc supports umbrella projects natively and emits one site covering every
  # child app. Order matters in `groups_for_modules` — the first pattern that
  # matches a module wins, so the catch-all per-app groups come last.
  defp docs do
    [
      main: "readme",
      output: "doc",
      formatters: ["html"],
      # ExDoc has no built-in mermaid renderer; this injects mermaid.js so the
      # ```mermaid fences in docs/ARCHITECTURE.md (and any @moduledoc) render as
      # diagrams instead of code blocks. GitHub renders those fences natively,
      # so the same source works in both places.
      before_closing_body_tag: &mermaid_script/1,
      skip_code_autolink_to: skip_autolink(),
      # Sidebar entries drop the shared prefix, so `Brain.Analysis.Pipeline`
      # renders as `Pipeline` under its group instead of the full path.
      nest_modules_by_prefix: [
        Brain.Analysis,
        Brain.Response,
        Brain.ML,
        Brain.Epistemic,
        Brain.Knowledge,
        Brain.Memory,
        Brain.Lexicon,
        Brain.Code,
        Brain.Graph,
        Brain.Services,
        Brain.Lattice,
        Atlas.Schemas,
        Atlas.Graph,
        Fleet,
        World,
        ChatWeb
      ],
      extras: extras(),
      groups_for_extras: [
        Overview: ["README.md", "CLAUDE.md"],
        Guides: ["docs/ARCHITECTURE.md", "docs/IMPLEMENTING_A_MODULE.md"],
        Subsystems: Path.wildcard("docs/*.md"),
        Audits: Path.wildcard("docs/internal/*.md")
      ],
      groups_for_modules: [
        # Mix tasks first: they live under Mix.Tasks.* and would otherwise be
        # scattered across the per-app catch-alls below.
        "Mix Tasks": ~r/^Mix\.Tasks\./,
        "Brain · Analysis": ~r/^Brain\.Analysis/,
        "Brain · Response": ~r/^Brain\.Response/,
        "Brain · ML": ~r/^Brain\.ML/,
        "Brain · Epistemic": ~r/^Brain\.Epistemic/,
        "Brain · Knowledge": ~r/^Brain\.Knowledge/,
        "Brain · Memory": ~r/^Brain\.Memory/,
        "Brain · Lexicon": ~r/^Brain\.(Lexicon|LinguisticData)/,
        "Brain · Code Intelligence": ~r/^Brain\.Code/,
        "Brain · Graph": ~r/^Brain\.(Graph|Atlas|Fact)/,
        "Brain · Services": ~r/^Brain\.Services/,
        "Brain · Lattice": ~r/^Brain\.Lattice/,
        "Brain · Runtime": ~r/^Brain/,
        "Atlas · Schemas": ~r/^Atlas\.Schemas/,
        "Atlas · Graph": ~r/^Atlas\.Graph/,
        "Atlas · Core": ~r/^Atlas/,
        Fleet: ~r/^Fleet/,
        World: ~r/^World/,
        Web: ~r/^ChatWeb/,
        Tasks: ~r/^Tasks/,
        FourthWall: ~r/^FourthWall/
      ]
    ]
  end

  # Module names in backticks get autolinked, and ExDoc warns when the target
  # can't be linked. Every term here is referenced *on purpose* by docs that
  # describe internals, so the reference should render as plain code rather than
  # be reworded to appease the linker.
  defp skip_autolink do
    [
      # Deliberate counter-example in docs/internal/apps_dead_code_audit.md,
      # which documents that the real module is `Brain.ML.NLPPipeline` and NOT
      # this spelling. Autolinking it is actively harmful on macOS: the beam
      # lookup for `Elixir.Brain.ML.NlpPipeline.beam` is case-insensitive, so
      # APFS returns `Elixir.Brain.ML.NLPPipeline.beam`, which loads and then
      # reports `module name in object code is ...` on every docs build.
      "Brain.ML.NlpPipeline",

      # `@moduledoc false` by convention (OTP application callbacks, test
      # support, macro-generated types) but still worth naming in audit docs.
      "Atlas.Application",
      "Atlas.PostgrexTypes",
      "Brain.Application",
      "Brain.Test.AtlasSandbox",
      "ChatWeb.Application",
      "Mix.Tasks.Split.HeldOut",
      "World.Application",

      # Function references are matched as their own terms, so listing the
      # parent module above is not enough to cover `Mod.fun/arity` mentions.
      "Brain.Application.start/2",
      "Fleet.Telemetry.handle_officer_event/4"
    ]
  end

  # ExDoc emits ```mermaid fences as <pre><code class="mermaid">. Mermaid expects
  # the raw graph source, so unwrap those into <pre class="mermaid"> before
  # initializing, and re-render on theme change so diagrams follow dark mode.
  defp mermaid_script(:html) do
    """
    <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
    <script>
      (function () {
        function theme() {
          return document.body.className.includes("dark") ? "dark" : "default";
        }

        function render() {
          document.querySelectorAll("pre > code.mermaid").forEach(function (code) {
            var pre = code.parentElement;
            pre.className = "mermaid";
            pre.textContent = code.textContent;
          });

          mermaid.initialize({ startOnLoad: false, theme: theme() });
          mermaid.run({ querySelector: "pre.mermaid" });
        }

        if (document.readyState === "loading") {
          document.addEventListener("DOMContentLoaded", render);
        } else {
          render();
        }
      })();
    </script>
    """
  end

  defp mermaid_script(_other), do: ""

  # Guides are listed explicitly first so they lead the sidebar; the wildcards
  # then pick up everything else. ExDoc rejects duplicate extras outright, so
  # dedupe rather than relying on it to collapse the overlap.
  defp extras do
    (["README.md", "CLAUDE.md", "docs/ARCHITECTURE.md", "docs/IMPLEMENTING_A_MODULE.md"] ++
       Path.wildcard("docs/*.md") ++
       Path.wildcard("docs/internal/*.md"))
    |> Enum.uniq()
  end

  defp releases do
    [
      chat_bot: [
        applications: [
          atlas: :permanent,
          brain: :permanent,
          world: :permanent,
          fleet: :permanent,
          tasks: :permanent,
          chat_web: :permanent
        ]
      ]
    ]
  end
end
