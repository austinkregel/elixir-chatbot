defmodule Brain.ML.GenerationTest do
  @moduledoc "Unit tests for the pluggable generation seam (no live backend needed)."
  use ExUnit.Case, async: true

  alias Brain.ML.Generation
  alias Brain.ML.Generation.{OpenAICompatible, OuroSidecar, Null}

  test "backend keys resolve to their implementation modules" do
    assert Generation.backend_module(:openai_compatible) == OpenAICompatible
    assert Generation.backend_module(:ouro_sidecar) == OuroSidecar
    assert Generation.backend_module(:null) == Null
  end

  test "a custom backend module is passed through" do
    assert Generation.backend_module(SomeCustom.Backend) == SomeCustom.Backend
  end

  test "only the ouro_sidecar backend contributes (heavy) children" do
    assert OpenAICompatible.children() == []
    assert Null.children() == []
    assert OuroSidecar.children() == [Brain.ML.Ouro.Model, Brain.ML.Ouro.SidecarLauncher]
  end

  test "the null backend is honest: no generation, cleanly reported" do
    assert Null.generate([%{role: "user", content: "hi"}]) == {:error, :no_backend}
    refute Null.ready?()
    assert Null.name() == :null
  end

  test "BackendError names the outage as a backend problem, not comprehension" do
    msg = Exception.message(%Brain.ML.Generation.BackendError{backend: :openai_compatible, reason: :econnrefused})
    assert msg =~ "openai_compatible"
    assert msg =~ "not a comprehension failure"
  end
end
