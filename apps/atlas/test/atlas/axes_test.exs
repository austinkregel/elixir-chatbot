defmodule Atlas.AxesTest do
  use Atlas.DataCase, async: false

  # These count and list whole tables, so they start from an empty one.
  # Emptied inside each test's transaction and rolled back with it.
  @moduletag :blank_slate

  alias Atlas.Axes
  alias Atlas.Schemas.{AxisObservation, AxisRun}

  defp run_attrs(overrides \\ %{}) do
    Map.merge(
      %{
        provenance: %{
          "git" => %{"sha" => "341b3d31fb646616fc3dce5ed1b81ad5fa244b9e", "dirty" => false},
          "extractor" => %{"schema_fingerprint" => "bc289842ba5ccb3d", "vector_dimension" => 343}
        },
        schema_fingerprint: "bc289842ba5ccb3d",
        corpus_sha256: String.duplicate("a", 64),
        utterance_count: 2
      },
      overrides
    )
  end

  defp observation(overrides \\ %{}) do
    Map.merge(
      %{utterance_id: "u1", axis: "polarity", value: "1.0", status: "computed"},
      overrides
    )
  end

  describe "AxisRun changeset" do
    test "requires the fields a run is selected and compared by" do
      changeset = AxisRun.changeset(%AxisRun{}, %{})

      assert %{
               schema_fingerprint: ["can't be blank"],
               corpus_sha256: ["can't be blank"],
               utterance_count: ["can't be blank"]
             } = errors_on(changeset)
    end

    test "rejects a schema fingerprint that is not 16 hex characters" do
      for bad <- ["not-hex-at-all!!", "bc289842ba5ccb3", "BC289842BA5CCB3D", ""] do
        changeset = AxisRun.changeset(%AxisRun{}, run_attrs(%{schema_fingerprint: bad}))
        refute changeset.valid?, "#{inspect(bad)} was accepted as a schema fingerprint"
      end
    end

    test "a tag without a note is rejected" do
      changeset = AxisRun.changeset(%AxisRun{}, run_attrs(%{tag: "baseline"}))

      refute changeset.valid?
      assert %{tag_note: ["is required when a run is tagged"]} = errors_on(changeset)

      blank = AxisRun.changeset(%AxisRun{}, run_attrs(%{tag: "baseline", tag_note: "   "}))
      refute blank.valid?
    end

    test "an untagged run needs no note" do
      assert AxisRun.changeset(%AxisRun{}, run_attrs()).valid?
    end

    test "two runs cannot share a tag" do
      {:ok, _} = Axes.record_run(run_attrs(%{tag: "baseline", tag_note: "first"}), [observation()])

      assert {:error, {:run, changeset}} =
               Axes.record_run(run_attrs(%{tag: "baseline", tag_note: "second"}), [observation()])

      assert %{tag: ["has already been taken"]} = errors_on(changeset)

      # The second run's observations must not be left behind either.
      assert Repo.aggregate(AxisRun, :count) == 1
      assert Repo.aggregate(AxisObservation, :count) == 1
    end

    test "many runs can be untagged" do
      assert {:ok, _} = Axes.record_run(run_attrs(), [observation()])
      assert {:ok, _} = Axes.record_run(run_attrs(), [observation()])

      assert length(Axes.list_runs()) == 2
    end
  end

  describe "AxisObservation changeset" do
    test "a defaulted observation must say why" do
      changeset =
        AxisObservation.changeset(
          %AxisObservation{},
          observation(%{run_id: Ecto.UUID.generate(), status: "defaulted", value: "0.0"})
        )

      refute changeset.valid?
      assert %{reason: ["is required when status is defaulted"]} = errors_on(changeset)
    end

    test "a defaulted observation with a reason is valid" do
      changeset =
        AxisObservation.changeset(
          %AxisObservation{},
          observation(%{
            run_id: Ecto.UUID.generate(),
            status: "defaulted",
            value: "0.0",
            reason: "no_negation_particle_matched"
          })
        )

      assert changeset.valid?
    end

    test "a computed observation needs no reason" do
      changeset =
        AxisObservation.changeset(%AxisObservation{}, observation(%{run_id: Ecto.UUID.generate()}))

      assert changeset.valid?
    end

    test "status is restricted to computed and defaulted" do
      changeset =
        AxisObservation.changeset(
          %AxisObservation{},
          observation(%{run_id: Ecto.UUID.generate(), status: "maybe"})
        )

      refute changeset.valid?
      assert %{status: ["is invalid"]} = errors_on(changeset)
    end
  end

  describe "record_run/2" do
    test "writes the run and every observation" do
      observations = [
        observation(%{utterance_id: "u1", axis: "polarity", value: "1.0"}),
        observation(%{utterance_id: "u1", axis: "tense", value: "present"}),
        observation(%{
          utterance_id: "u2",
          axis: "polarity",
          value: "0.0",
          status: "defaulted",
          reason: "no_negation_particle_matched"
        })
      ]

      assert {:ok, %{run: run, observations: 3}} = Axes.record_run(run_attrs(), observations)
      assert run.id
      assert Repo.aggregate(AxisObservation.for_run(AxisObservation, run.id), :count) == 3
    end

    test "one invalid observation writes nothing at all" do
      observations = [
        observation(),
        # defaulted with no reason
        observation(%{utterance_id: "u2", status: "defaulted", value: "0.0"}),
        observation(%{utterance_id: "u3"})
      ]

      assert {:error, {:observation, 1, changeset}} = Axes.record_run(run_attrs(), observations)
      assert %{reason: ["is required when status is defaulted"]} = errors_on(changeset)

      # The run must not exist either -- a run with no observations reads as a
      # measurement that found nothing.
      assert Repo.aggregate(AxisRun, :count) == 0
      assert Repo.aggregate(AxisObservation, :count) == 0
    end

    test "an invalid run writes nothing and names the run" do
      assert {:error, {:run, changeset}} =
               Axes.record_run(run_attrs(%{utterance_count: 0}), [observation()])

      refute changeset.valid?
      assert Repo.aggregate(AxisRun, :count) == 0
    end

    test "stores a full feature vector when one is given" do
      vector = Enum.map(1..343, fn i -> i / 343 end)

      {:ok, %{run: run}} =
        Axes.record_run(run_attrs(), [observation(%{feature_vector: vector})])

      [stored] = Repo.all(AxisObservation.for_run(AxisObservation, run.id))
      assert length(stored.feature_vector) == 343
      assert Enum.at(stored.feature_vector, 0) == 1 / 343
    end
  end

  describe "census/1" do
    test "counts computed and defaulted per axis" do
      observations = [
        observation(%{utterance_id: "u1", axis: "polarity", value: "1.0"}),
        observation(%{utterance_id: "u2", axis: "polarity", value: "1.0"}),
        observation(%{
          utterance_id: "u3",
          axis: "novelty_score",
          value: "0.0",
          status: "defaulted",
          reason: "absent"
        })
      ]

      {:ok, %{run: run}} = Axes.record_run(run_attrs(%{utterance_count: 3}), observations)

      assert Axes.census(run.id) == %{
               "polarity" => %{computed: 2, defaulted: 0},
               "novelty_score" => %{computed: 0, defaulted: 1}
             }
    end

    test "an axis that was never observed is absent, not zero" do
      {:ok, %{run: run}} = Axes.record_run(run_attrs(), [observation()])

      census = Axes.census(run.id)
      assert Map.has_key?(census, "polarity")
      refute Map.has_key?(census, "self_disclosure_level")
    end
  end

  describe "tag_run/3 and get_by_tag/1" do
    test "promotes an untagged run to a reference point" do
      {:ok, %{run: run}} = Axes.record_run(run_attrs(), [observation()])
      refute run.tag

      assert {:ok, tagged} = Axes.tag_run(run.id, "baseline-2026-09-25", "first honest measurement")
      assert tagged.tag == "baseline-2026-09-25"
      assert Axes.get_by_tag("baseline-2026-09-25").id == run.id
    end

    test "tagging still requires a note" do
      {:ok, %{run: run}} = Axes.record_run(run_attrs(), [observation()])

      assert {:error, changeset} = Axes.tag_run(run.id, "baseline", "")
      assert %{tag_note: ["is required when a run is tagged"]} = errors_on(changeset)
    end

    test "tagging a run that does not exist reports it" do
      assert {:error, :not_found} = Axes.tag_run(Ecto.UUID.generate(), "t", "n")
    end
  end

  describe "prune_untagged/1 " do
    test "deletes untagged runs and their observations, keeping tagged ones" do
      {:ok, %{run: untagged}} = Axes.record_run(run_attrs(), [observation(), observation(%{utterance_id: "u2"})])
      {:ok, %{run: tagged}} = Axes.record_run(run_attrs(%{tag: "keep", tag_note: "a reference"}), [observation()])

      assert Repo.aggregate(AxisObservation, :count) == 3

      cutoff = DateTime.utc_now() |> DateTime.add(60, :second)
      assert Axes.prune_untagged(cutoff) == 1

      assert Repo.aggregate(AxisRun, :count) == 1
      assert Repo.get(AxisRun, tagged.id)
      refute Repo.get(AxisRun, untagged.id)

      # The observations went with the run, via on_delete: :delete_all, so a run
      # and its rows cannot be separated.
      assert Repo.aggregate(AxisObservation, :count) == 1
    end

    test "leaves untagged runs newer than the cutoff alone" do
      {:ok, _} = Axes.record_run(run_attrs(), [observation()])

      cutoff = DateTime.utc_now() |> DateTime.add(-3600, :second)
      assert Axes.prune_untagged(cutoff) == 0
      assert Repo.aggregate(AxisRun, :count) == 1
    end
  end

  describe "comparability" do
    test "runs are queryable by the schema fingerprint they were taken under" do
      {:ok, _} = Axes.record_run(run_attrs(%{schema_fingerprint: "bc289842ba5ccb3d"}), [observation()])
      {:ok, _} = Axes.record_run(run_attrs(%{schema_fingerprint: "fb3ce7f1e4739cb3"}), [observation()])

      comparable = AxisRun.comparable_to(AxisRun, "bc289842ba5ccb3d") |> Repo.all()

      assert length(comparable) == 1
      assert hd(comparable).schema_fingerprint == "bc289842ba5ccb3d"
    end
  end
end
