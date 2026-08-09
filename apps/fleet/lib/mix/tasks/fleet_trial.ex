defmodule Mix.Tasks.Fleet.Trial do
  @moduledoc """
  Run the fleet's red-team trials and print the result table.

      mix fleet.trial                      # every harness trial (no model needed)
      mix fleet.trial --list               # list scenarios and the failure taxonomy
      mix fleet.trial --only frame_escape  # one scenario
      mix fleet.trial --cognition          # include model-driven trials (slow)
      mix fleet.trial --soul ensign-jj7    # which soul crews the cognition trials

  Harness trials assert properties of the deterministic layer — gates, framing,
  isolation, review — and are cheap enough to run on every change. Cognition
  trials commission a real officer against a rigged mind-world and cost a
  mission each.

  A `pass` is an *honourable exit*, not task success: an agent that reports
  honestly that it could not do the impossible has passed.
  """
  @shortdoc "Run the fleet red-team trials"

  use Mix.Task

  alias Fleet.Trial.{Runner, Scenario, Taxonomy}

  @impl Mix.Task
  def run(argv) do
    {opts, _, _} =
      OptionParser.parse(argv,
        strict: [list: :boolean, only: :string, cognition: :boolean, soul: :string],
        aliases: [l: :list, o: :only]
      )

    if opts[:list] do
      list()
    else
      Mix.Task.run("app.start")
      run_trials(opts)
    end
  end

  defp list do
    Mix.shell().info("\nFailure taxonomy\n" <> String.duplicate("-", 60))

    Mix.shell().info("\n  Inherited from LCARS §7:")

    for {id, desc} <- Taxonomy.inherited() do
      Mix.shell().info("    #{String.pad_trailing(to_string(id), 5)} #{desc}")
    end

    Mix.shell().info("\n  Specific to this implementation's subsystems:")

    for {id, desc} <- Taxonomy.ours() do
      Mix.shell().info("    #{String.pad_trailing(to_string(id), 5)} #{desc}")
    end

    Mix.shell().info("\n\nScenarios\n" <> String.duplicate("-", 60))

    for scenario <- Scenario.all() do
      Mix.shell().info(
        "\n  #{scenario.id}  [#{scenario.kind}]  targets: #{Enum.join(scenario.targets, ", ")}\n" <>
          "    #{scenario.summary}"
      )
    end

    Mix.shell().info("")
  end

  defp run_trials(opts) do
    only = opts[:only] && String.split(opts[:only], ",") |> Enum.map(&String.to_atom/1)
    run_opts = if only, do: [only: only], else: []

    harness = Runner.run_harness(run_opts)

    cognition =
      if opts[:cognition] do
        soul = opts[:soul] || "ensign-jj7"

        :cognition
        |> Scenario.by_kind()
        |> then(fn s -> if only, do: Enum.filter(s, &(&1.id in only)), else: s end)
        |> Enum.map(&Runner.run_scenario(&1, soul_id: soul))
      else
        []
      end

    results = harness ++ cognition

    Mix.shell().info("\n" <> Runner.summarize(results) <> "\n")

    unless opts[:cognition] do
      Mix.shell().info("(cognition trials skipped — pass --cognition to include them)\n")
    end

    failures = Enum.reject(results, &Taxonomy.pass?(&1.verdict))

    if failures != [] do
      Mix.shell().error("#{length(failures)} trial(s) did not pass")
      exit({:shutdown, 1})
    end
  end
end
