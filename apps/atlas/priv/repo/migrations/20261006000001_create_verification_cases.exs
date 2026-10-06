defmodule Atlas.Repo.Migrations.CreateVerificationCases do
  use Ecto.Migration

  def change do
    # One row per human-curated verification scenario: an input to a subsystem,
    # what a person decided the right answer is, and what the subsystem last
    # actually produced.
    #
    # These are deliberately NOT a gold standard. A gold standard is a bulk
    # corpus for training and evaluation (tasks 013-015); these are a handful of
    # scenarios a person checked by hand, most of which have no gold-standard
    # analogue. Where the two overlap -- the response page -- the export is
    # explicit rather than two copies being maintained.
    create table(:atlas_verification_cases, primary_key: false) do
      add :id, :binary_id, primary_key: true

      # Which page/subsystem the case belongs to, as declared in
      # ChatWeb.Harness.Subsystems. Validated against that declaration rather
      # than being free text, so a case cannot be saved against a subsystem
      # that has no page to exercise it.
      add :subsystem, :string, null: false

      # A human-readable name, because a list of cases identified only by their
      # input is unusable once there are more than three of them.
      add :name, :string, null: false

      add :input, :map, null: false, default: %{}

      # What a person decided is correct. A map of the parts of the output they
      # chose to assert, not the whole term -- a full-term expectation breaks on
      # every unrelated change and so gets deleted rather than fixed. The
      # partiality is made visible on the page instead of being silent: the
      # diff reports how many keys were checked against how many the subsystem
      # returned.
      add :expected, :map, null: false, default: %{}

      add :last_actual, :map

      # Float comparison tolerance for this case. NULL means "use
      # Atlas.Verification.Comparison.default_tolerance/0", which is tight
      # (1.0e-6) so that a regression cannot hide inside it. A case comparing a
      # value this repo is known to drift on widens its own tolerance
      # deliberately, which keeps the loosening attached to the one case that
      # needs it rather than applied to all of them.
      add :tolerance, :float

      # "pending" until the case has been run even once. Never "pass" by
      # default: a case that has not run has not passed, and a store that says
      # otherwise would make /verify's counts a lie on the day it ships.
      add :status, :string, null: false, default: "pending"

      # When last_actual was produced. Deliberately NOT used to derive
      # staleness by comparing against updated_at: Ecto stamps updated_at at
      # write time, microseconds after the clock read that fills this column, so
      # that comparison reports every freshly-run case as stale. An edit resets
      # `status` to "pending" instead, which states the same fact once.
      add :last_run_at, :utc_datetime_usec

      # A subsystem's answer depends on which world's models are loaded, so a
      # case is only meaningful within the world it was recorded in.
      add :world_id, :string, null: false

      timestamps(type: :utc_datetime_usec)
    end

    # A case is identified by what it exercises, in which world, under what
    # name. Saving the same name twice for one subsystem and world is an edit,
    # not a second case.
    create unique_index(:atlas_verification_cases, [:subsystem, :world_id, :name],
             name: :atlas_verification_cases_identity_index
           )

    # The query /verify is built on: pass/fail counts per subsystem.
    create index(:atlas_verification_cases, [:subsystem, :status])

    create index(:atlas_verification_cases, [:world_id])
  end
end
