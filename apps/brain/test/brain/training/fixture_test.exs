defmodule Brain.Training.FixtureTest do
  @moduledoc """
  The task 086 fixture format is enforced at load: every rule breach fails
  loudly, naming the file and the record. Also checks the committed EWT
  fixtures against the sources they were built from.
  """
  use ExUnit.Case, async: true

  alias Brain.Training.Fixture

  @moduletag :tmp_dir

  @pos_dir Brain.priv_path("training/pos")

  defp record(overrides \\ %{}) do
    Map.merge(
      %{
        "id" => "test/0001",
        "text" => "turn it off",
        "origin" => "authored",
        "source_id" => "fixture_test#1",
        "license" => "apache-2.0",
        "produced_at" => "2026-09-19T00:00:00Z",
        "producer" => %{
          "name" => "FixtureTest",
          "version" => "1",
          "inputs" => [%{"name" => "hand", "version" => "1", "sha256" => String.duplicate("a", 64)}]
        },
        "tokens" => ["turn", "it", "off"],
        "layers" => %{
          "pos" => %{
            "tags" => ["VERB", "PRON", "ADP"],
            "vocabulary" => "ud_v1",
            "resolution" => ["authored", "authored", "authored"]
          }
        }
      },
      overrides
    )
  end

  defp write(dir, records, version \\ 1) do
    path = Path.join(dir, "fixture.json")
    File.write!(path, Jason.encode!(%{"format_version" => version, "records" => records}))
    path
  end

  defp pos_layer(overrides), do: %{"pos" => Map.merge(record()["layers"]["pos"], overrides)}

  describe "load!/1 accepts" do
    test "a well-formed file, returning its records", %{tmp_dir: dir} do
      assert [%{"id" => "test/0001"}] = Fixture.load!(write(dir, [record()]))
    end
  end

  describe "load!/1 refuses, naming the file and record" do
    test "a missing required field", %{tmp_dir: dir} do
      path = write(dir, [Map.delete(record(), "license")])
      assert_raise RuntimeError, ~r/test\/0001.*missing required field "license"/, fn -> Fixture.load!(path) end
    end

    test "tags whose count differs from the tokens", %{tmp_dir: dir} do
      path = write(dir, [record(%{"layers" => pos_layer(%{"tags" => ["VERB", "PRON"]})})])
      assert_raise RuntimeError, ~r/2 tags for 3 tokens/, fn -> Fixture.load!(path) end
    end

    test "a resolution list whose count differs from the tokens", %{tmp_dir: dir} do
      path = write(dir, [record(%{"layers" => pos_layer(%{"resolution" => ["authored"]})})])
      assert_raise RuntimeError, ~r/1 resolutions for 3 tokens/, fn -> Fixture.load!(path) end
    end

    test "a tag outside the declared vocabulary", %{tmp_dir: dir} do
      # CCONJ is UD v2; ud_v1 has CONJ.
      path = write(dir, [record(%{"layers" => pos_layer(%{"tags" => ["VERB", "PRON", "CCONJ"]})})])
      assert_raise RuntimeError, ~r/tags outside ud_v1: \["CCONJ"\]/, fn -> Fixture.load!(path) end
    end

    test "an unknown vocabulary", %{tmp_dir: dir} do
      path = write(dir, [record(%{"layers" => pos_layer(%{"vocabulary" => "ud_v2"})})])
      assert_raise RuntimeError, ~r/unknown vocabulary "ud_v2"/, fn -> Fixture.load!(path) end
    end

    test "an unknown resolution tier", %{tmp_dir: dir} do
      path = write(dir, [record(%{"layers" => pos_layer(%{"resolution" => ["authored", "guessed", "authored"]})})])
      assert_raise RuntimeError, ~r/unknown resolution tiers \["guessed"\]/, fn -> Fixture.load!(path) end
    end

    test "a producer input without a sha256", %{tmp_dir: dir} do
      producer = %{"name" => "x", "version" => "1", "inputs" => [%{"name" => "hand", "version" => "1"}]}
      path = write(dir, [record(%{"producer" => producer})])
      assert_raise RuntimeError, ~r/64-hex sha256/, fn -> Fixture.load!(path) end
    end

    test "a producer with no inputs", %{tmp_dir: dir} do
      path = write(dir, [record(%{"producer" => %{"name" => "x", "version" => "1", "inputs" => []}})])
      assert_raise RuntimeError, ~r/at least one input/, fn -> Fixture.load!(path) end
    end

    test "a produced_at that is not a datetime", %{tmp_dir: dir} do
      path = write(dir, [record(%{"produced_at" => "yesterday"})])
      assert_raise RuntimeError, ~r/not an ISO 8601 datetime/, fn -> Fixture.load!(path) end
    end

    test "empty tokens", %{tmp_dir: dir} do
      path = write(dir, [record(%{"tokens" => []})])
      assert_raise RuntimeError, ~r/non-empty list/, fn -> Fixture.load!(path) end
    end

    test "a repeated id", %{tmp_dir: dir} do
      path = write(dir, [record(), record(%{"text" => "turn it on"})])
      assert_raise RuntimeError, ~r/repeated ids: \["test\/0001"\]/, fn -> Fixture.load!(path) end
    end

    test "another format version", %{tmp_dir: dir} do
      path = write(dir, [record()], 2)
      assert_raise RuntimeError, ~r/format_version 2, expected 1/, fn -> Fixture.load!(path) end
    end

    test "a file with no records", %{tmp_dir: dir} do
      assert_raise RuntimeError, ~r/holds no records/, fn -> Fixture.load!(write(dir, [])) end
    end

    test "a bare list, which has no format version", %{tmp_dir: dir} do
      path = Path.join(dir, "bare.json")
      File.write!(path, Jason.encode!([record()]))
      assert_raise RuntimeError, ~r/not a fixture file/, fn -> Fixture.load!(path) end
    end
  end

  describe "pos_sequences/1" do
    test "gives the tokens and tags the tagger trains on", %{tmp_dir: dir} do
      records = Fixture.load!(write(dir, [record()]))

      assert Fixture.pos_sequences(records) == [
               %{tokens: ["turn", "it", "off"], tags: ["VERB", "PRON", "ADP"]}
             ]
    end
  end

  describe "the committed EWT fixtures" do
    setup do
      sources = @pos_dir |> Path.join("sources.json") |> File.read!() |> Jason.decode!()
      {:ok, ewt: sources["ud_ewt"]}
    end

    test "each split is valid and records the pinned source it came from", %{ewt: ewt} do
      for {split, %{"file" => file, "sha256" => sha}} <- ewt["files"] do
        records = Fixture.load!(Path.join(@pos_dir, "ud_ewt.#{split}.json"))

        assert records != []
        assert Enum.all?(records, &(&1["origin"] == "corpus:ud_ewt"))
        assert Enum.all?(records, &(&1["license"] == ewt["license"]))

        inputs = records |> Enum.flat_map(& &1["producer"]["inputs"]) |> Enum.uniq()
        assert %{"name" => "ud_ewt/#{file}", "version" => ewt["version"], "sha256" => sha} in inputs
      end
    end

    test "the mapping they were built with is the one committed beside them" do
      mapping = Path.join(@pos_dir, "ud_v2_to_ud_v1.json")
      sha = :crypto.hash(:sha256, File.read!(mapping)) |> Base.encode16(case: :lower)

      [first | _] = Fixture.load!(Path.join(@pos_dir, "ud_ewt.dev.json"))
      assert Enum.any?(first["producer"]["inputs"], &(&1["name"] == "ud_v2_to_ud_v1" and &1["sha256"] == sha))
    end
  end
end
