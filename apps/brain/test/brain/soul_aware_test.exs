defmodule Brain.SoulAwareTest do
  @moduledoc """
  Proves the soul-aware generation seam: a Soul residing in a world renders its
  constitution as the system prompt, and generation stays backward-compatible
  (the generic default) when no soul resides.
  """
  use ExUnit.Case, async: false

  alias Brain.Response.RealizationPacket
  alias Brain.Analysis.ChunkAnalysis
  alias Brain.Soul

  setup do
    dir = Path.join(System.tmp_dir!(), "souls_test_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)

    File.write!(
      Path.join(dir, "ensign-test.json"),
      Jason.encode!(%{
        "id" => "ensign-test",
        "name" => "Ensign Test",
        "constitution" => "You are Ensign Test, a careful officer who never fabricates."
      })
    )

    Application.put_env(:brain, :souls_dir, dir)
    on_exit(fn ->
      File.rm_rf(dir)
      Application.delete_env(:brain, :souls_dir)
    end)

    :ok
  end

  test "a Soul loads from file and renders its constitution" do
    assert {:ok, soul} = Soul.get("ensign-test")
    assert soul.name == "Ensign Test"
    assert Soul.system_prompt(soul) =~ "never fabricates"
  end

  test "the acting soul's constitution becomes the system message" do
    {:ok, soul} = Soul.get("ensign-test")

    [system | _] =
      RealizationPacket.build([], %ChunkAnalysis{}, unified_context: %{soul: soul})

    assert system.role == "system"
    assert system.content =~ "You are Ensign Test"
    # the generic identity is replaced...
    refute system.content =~ "You are a conversational assistant"
    # ...but the mechanical output instructions are retained
    assert system.content =~ "Output only the final response text"
  end

  test "falls back to the generic default when no soul resides (backward compatible)" do
    [system | _] = RealizationPacket.build([], %ChunkAnalysis{}, unified_context: %{})

    assert system.role == "system"
    assert system.content =~ "You are a conversational assistant"
  end
end
