defmodule Brain.ML.Ouro.ModelGenerationTest do
  use ExUnit.Case, async: false

  alias Brain.ML.Ouro.Model, as: OuroModel
  alias Brain.ML.Ouro.SidecarLauncher

  describe "generate/3" do
    @tag timeout: 120_000
    test "generates text from a ChatML plan realization prompt" do
      SidecarLauncher.ensure_ready!(timeout: 90_000)

      assert OuroModel.ready?(),
        "Ouro model not ready — sidecar is healthy but Model GenServer has not detected it yet"

      messages = [
        %{
          role: "system",
          content: "You are a response realization engine. Output only the response text."
        },
        %{
          role: "user",
          content:
            Jason.encode!(%{
              mode: "plan_realization",
              tone: "neutral",
              plan: [
                %{
                  type: "acknowledgment",
                  variant: "social",
                  payload: %{speech_act_sub_type: "greeting"}
                },
                %{type: "follow_up", variant: "elaboration", payload: %{}}
              ]
            })
        }
      ]

      result = OuroModel.generate(messages, max_new_tokens: 50, temperature: 0.7)

      assert {:ok, text} = result,
        "Expected {:ok, text}, got #{inspect(result)}"

      assert text == "Hello! How can I assist you today?"
      assert is_binary(text) and text != "",
        "Expected non-empty generated text, got: #{inspect(text)}"

      assert String.length(text) < 200,
        "Expected concise greeting, got #{String.length(text)} chars: #{inspect(text)}"

      meta_terms = ["JSON", "structure", "plan", "payload", "user provided"]
      lowercase = String.downcase(text)

      refute Enum.any?(meta_terms, &String.contains?(lowercase, String.downcase(&1))),
        "Response contains meta-commentary about the plan instead of realizing it: #{inspect(text)}"
    end
  end

end
