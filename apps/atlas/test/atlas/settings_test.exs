defmodule Atlas.SettingsTest do
  @moduledoc """
  The store behind node-wide runtime settings: a value written by name is read
  back by that name, a later write replaces it, a setting never written is
  reported as absent, and a blank name or value is refused.
  """
  use Atlas.DataCase, async: false

  alias Atlas.Schemas.Setting
  alias Atlas.Settings

  describe "fetch/1" do
    test "reports a setting that was never written as absent" do
      assert Settings.fetch("test_never_written") == :error
    end

    test "reads back the written value" do
      Settings.put!("test_setting", "test_world")

      assert Settings.fetch("test_setting") == {:ok, "test_world"}
    end
  end

  describe "put!/2" do
    test "replaces the earlier value and keeps one row" do
      Settings.put!("test_setting", "test_first")
      Settings.put!("test_setting", "test_second")

      assert Settings.fetch("test_setting") == {:ok, "test_second"}
      assert Repo.aggregate(from(s in Setting, where: s.name == "test_setting"), :count) == 1
    end

    test "raises on a blank name or value and stores nothing" do
      assert_raise Ecto.InvalidChangesetError, fn -> Settings.put!("", "test_world") end
      assert_raise Ecto.InvalidChangesetError, fn -> Settings.put!("test_setting", "   ") end

      assert Settings.fetch("test_setting") == :error
    end
  end
end
