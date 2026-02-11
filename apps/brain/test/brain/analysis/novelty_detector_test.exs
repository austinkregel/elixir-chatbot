defmodule Brain.Analysis.NoveltyDetectorTest do
  use ExUnit.Case, async: true

  alias Brain.Analysis.NoveltyDetector

  describe "is_novel?/3" do
    test "returns novel for low confidence" do
      assert {:novel, _score} = NoveltyDetector.is_novel?(0.3, 0.5)
    end

    test "returns novel for small margin" do
      assert {:novel, _score} = NoveltyDetector.is_novel?(0.7, 0.1)
    end

    test "returns not_novel for high confidence and large margin" do
      assert :not_novel = NoveltyDetector.is_novel?(0.8, 0.5)
    end

    test "respects custom thresholds" do
      assert :not_novel = NoveltyDetector.is_novel?(0.6, 0.3, novelty_threshold: 0.4, margin_threshold: 0.2)
      assert {:novel, _score} = NoveltyDetector.is_novel?(0.3, 0.1, novelty_threshold: 0.5, margin_threshold: 0.2)
    end
  end

  describe "is_substantive?/2" do
    test "returns true for directive speech acts" do
      speech_act = %{category: :directive, sub_type: :command}
      assert NoveltyDetector.is_substantive?(speech_act, "smarthome.device.switch.on")
    end

    test "returns true for assertive speech acts" do
      speech_act = %{category: :assertive, sub_type: :statement}
      assert NoveltyDetector.is_substantive?(speech_act, "unknown")
    end

    test "returns false for well-handled expressives" do
      speech_act = %{category: :expressive, sub_type: :greeting}
      refute NoveltyDetector.is_substantive?(speech_act, "smalltalk.greetings.hello")
    end

    test "returns true for other expressives" do
      speech_act = %{category: :expressive, sub_type: :general}
      assert NoveltyDetector.is_substantive?(speech_act, "unknown")
    end
  end
end
