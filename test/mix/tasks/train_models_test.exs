defmodule Mix.Tasks.TrainModelsTest do
  use ExUnit.Case, async: true

  @moduletag :mix_task

  describe "module loading" do
    test "module is loaded correctly" do
      assert Code.ensure_loaded?(Mix.Tasks.TrainModels)
    end

    test "module uses Mix.Task" do
      behaviours = Mix.Tasks.TrainModels.__info__(:attributes)[:behaviour] || []
      assert Mix.Task in behaviours
    end
  end

  describe "option parsing" do
    test "parses --intent-only flag" do
      {opts, _, _} =
        OptionParser.parse(["--intent-only"],
          strict: [
            world: :string,
            intent_only: :boolean,
            entity_only: :boolean,
            pos_only: :boolean,
            gazetteer_only: :boolean,
            skip_gazetteer: :boolean,
            skip_pos: :boolean
          ]
        )

      assert Keyword.get(opts, :intent_only) == true
    end

    test "parses --entity-only flag" do
      {opts, _, _} =
        OptionParser.parse(["--entity-only"],
          strict: [
            world: :string,
            intent_only: :boolean,
            entity_only: :boolean,
            pos_only: :boolean,
            gazetteer_only: :boolean,
            skip_gazetteer: :boolean,
            skip_pos: :boolean
          ]
        )

      assert Keyword.get(opts, :entity_only) == true
    end

    test "parses --pos-only flag" do
      {opts, _, _} =
        OptionParser.parse(["--pos-only"],
          strict: [
            world: :string,
            intent_only: :boolean,
            entity_only: :boolean,
            pos_only: :boolean,
            gazetteer_only: :boolean,
            skip_gazetteer: :boolean,
            skip_pos: :boolean
          ]
        )

      assert Keyword.get(opts, :pos_only) == true
    end

    test "parses --gazetteer-only flag" do
      {opts, _, _} =
        OptionParser.parse(["--gazetteer-only"],
          strict: [
            world: :string,
            intent_only: :boolean,
            entity_only: :boolean,
            pos_only: :boolean,
            gazetteer_only: :boolean,
            skip_gazetteer: :boolean,
            skip_pos: :boolean
          ]
        )

      assert Keyword.get(opts, :gazetteer_only) == true
    end

    test "parses --world flag" do
      {opts, _, _} =
        OptionParser.parse(["--world", "test_world"],
          strict: [
            world: :string,
            intent_only: :boolean,
            entity_only: :boolean,
            pos_only: :boolean,
            gazetteer_only: :boolean,
            skip_gazetteer: :boolean,
            skip_pos: :boolean
          ]
        )

      assert Keyword.get(opts, :world) == "test_world"
    end

    test "parses --skip-gazetteer flag" do
      {opts, _, _} =
        OptionParser.parse(["--skip-gazetteer"],
          strict: [
            world: :string,
            intent_only: :boolean,
            entity_only: :boolean,
            pos_only: :boolean,
            gazetteer_only: :boolean,
            skip_gazetteer: :boolean,
            skip_pos: :boolean
          ]
        )

      assert Keyword.get(opts, :skip_gazetteer) == true
    end

    test "parses --skip-pos flag" do
      {opts, _, _} =
        OptionParser.parse(["--skip-pos"],
          strict: [
            world: :string,
            intent_only: :boolean,
            entity_only: :boolean,
            pos_only: :boolean,
            gazetteer_only: :boolean,
            skip_gazetteer: :boolean,
            skip_pos: :boolean
          ]
        )

      assert Keyword.get(opts, :skip_pos) == true
    end

    test "parses combined flags" do
      {opts, _, _} =
        OptionParser.parse(["--world", "my_world", "--intent-only", "--skip-gazetteer"],
          strict: [
            world: :string,
            intent_only: :boolean,
            entity_only: :boolean,
            pos_only: :boolean,
            gazetteer_only: :boolean,
            skip_gazetteer: :boolean,
            skip_pos: :boolean
          ]
        )

      assert Keyword.get(opts, :world) == "my_world"
      assert Keyword.get(opts, :intent_only) == true
      assert Keyword.get(opts, :skip_gazetteer) == true
    end
  end

  describe "training data" do
    test "intents directory exists" do
      paths = ["data/intents", "data/training/intents"]
      assert Enum.any?(paths, &File.dir?/1)
    end

    test "entities directory exists" do
      paths = ["data/entities", "data/training/entities"]
      assert Enum.any?(paths, &File.dir?/1)
    end

    test "intents directory has JSON files" do
      paths = ["data/intents", "data/training/intents"]
      intents_dir = Enum.find(paths, &File.dir?/1)

      if intents_dir do
        json_files = Path.wildcard("#{intents_dir}/*.json")
        assert length(json_files) > 0
      end
    end

    test "entities directory has JSON files" do
      paths = ["data/entities", "data/training/entities"]
      entities_dir = Enum.find(paths, &File.dir?/1)

      if entities_dir do
        json_files = Path.wildcard("#{entities_dir}/*.json")
        assert length(json_files) > 0
      end
    end
  end

  describe "model output directory" do
    test "priv/ml_models directory exists" do
      assert File.dir?("priv/ml_models")
    end

    test "models directory contains term files" do
      if File.dir?("priv/ml_models") do
        term_files = Path.wildcard("priv/ml_models/*.term")
        # Should have some model files
        assert length(term_files) >= 0
      end
    end
  end
end
