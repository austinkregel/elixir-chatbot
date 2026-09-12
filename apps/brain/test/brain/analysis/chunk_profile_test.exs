defmodule Brain.Analysis.ChunkProfileTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.{ChunkProfile, ChunkAnalysis, SpeechActResult, DiscourseResult}

  describe "new/0" do
    test "creates a default profile with all expected fields" do
      profile = ChunkProfile.new()

      assert profile.domain == :unknown
      assert profile.speech_act_category == :unknown
      assert profile.speech_act_subtype == :unknown
      assert profile.target == :ambiguous
      assert profile.modality == :declarative
      assert profile.polarity == :affirmative
      assert profile.tense == :present
      assert profile.aspect == :simple
      assert profile.addressee == :unknown
      assert profile.urgency == :low
      assert profile.certainty == :committed
      assert profile.sentiment_alignment == :neutral
      assert profile.slot_completeness == 1.0
      assert profile.novelty_score == 0.0
      assert profile.feature_provenance == %{}
      assert profile.confidence == 0.0
      assert profile.derived_label == ""
      assert profile.response_posture == :direct
      assert profile.engagement_level == :casual_engagement
      assert profile.self_disclosure_level == :none
      assert profile.temporal_framing == :timeless
      assert profile.feature_vector == []
    end
  end

  describe "materialize/2" do
    test "projects speech act fields from analysis" do
      analysis = %ChunkAnalysis{
        chunk_index: 0,
        text: "hello there",
        speech_act:
          SpeechActResult.new(:expressive, :greeting, 0.9,
            is_question: false,
            is_imperative: false
          ),
        discourse: DiscourseResult.new(:bot, 0.8),
        confidence: 0.85,
        pos_tags: [{"hello", "INTJ"}, {"there", "ADV"}]
      }

      profile = ChunkProfile.materialize(analysis, [])

      assert profile.speech_act_category == :expressive
      assert profile.speech_act_subtype == :greeting
      assert profile.addressee == :bot
      assert is_float(profile.confidence)
    end

    test "derives label from domain and speech_act_subtype" do
      profile = %ChunkProfile{domain: :weather, speech_act_subtype: :question_factual}
      assert ChunkProfile.derived_label(profile) == "weather.question_factual"
    end

    test "handles nil speech_act gracefully" do
      analysis = %ChunkAnalysis{
        chunk_index: 0,
        text: "test",
        speech_act: nil,
        discourse: nil,
        confidence: 0.5,
        pos_tags: []
      }

      profile = ChunkProfile.materialize(analysis, [])
      assert profile.speech_act_category == :unknown
      assert profile.addressee == :unknown
    end

    test "stores feature vector" do
      analysis = %ChunkAnalysis{
        chunk_index: 0,
        text: "what is the weather",
        speech_act:
          SpeechActResult.new(:directive, :question_factual, 0.8, is_question: true),
        discourse: DiscourseResult.new(:bot, 0.7),
        confidence: 0.7,
        pos_tags: [{"what", "PRON"}, {"is", "AUX"}, {"the", "DET"}, {"weather", "NOUN"}]
      }

      feature_vector = List.duplicate(0.5, 140)
      profile = ChunkProfile.materialize(analysis, feature_vector)

      assert length(profile.feature_vector) == 140
    end

    test "projects modality from speech act" do
      question_analysis = %ChunkAnalysis{
        chunk_index: 0,
        text: "what time is it",
        speech_act:
          SpeechActResult.new(:directive, :question_factual, 0.9, is_question: true),
        discourse: DiscourseResult.new(:bot, 0.8),
        confidence: 0.8,
        pos_tags: []
      }

      profile = ChunkProfile.materialize(question_analysis, [])
      assert profile.modality == :interrogative
    end
  end

  describe "interaction axes" do
    test "response_posture is :direct for high confidence committed utterances" do
      profile = %ChunkProfile{
        confidence: 0.9,
        certainty: :committed,
        slot_completeness: 1.0,
        novelty_score: 0.1
      }

      assert profile.confidence >= 0.7
      assert profile.certainty == :committed
    end

    test "derived_label format" do
      profile = %ChunkProfile{
        domain: :smarthome,
        speech_act_subtype: :command
      }

      assert ChunkProfile.derived_label(profile) == "smarthome.command"
    end
  end

  describe "axis_manifest/0" do
    test "declares exactly the 18 axes, and every one is a real struct field" do
      axes = ChunkProfile.axes()

      assert length(axes) == 18
      assert axes == Enum.uniq(axes)

      struct_fields = %ChunkProfile{} |> Map.from_struct() |> Map.keys()

      Enum.each(axes, fn axis ->
        assert axis in struct_fields,
               "axis #{inspect(axis)} is declared in the manifest but is not a " <>
                 "%ChunkProfile{} field — the manifest would describe a value " <>
                 "that is never set"
      end)
    end

    test "every axis declares a kind and a source we know how to fill" do
      Enum.each(ChunkProfile.axis_manifest(), fn {axis, kind, _domain, source} ->
        assert kind in [:categorical, :continuous], "#{axis} has kind #{inspect(kind)}"

        assert source in [:analysis, :micro_classifier, :composed, :derived],
               "#{axis} has source #{inspect(source)}"
      end)
    end

    test "axis_default/1 agrees with the struct, so defaults have one definition" do
      defaults = Map.from_struct(%ChunkProfile{})

      Enum.each(ChunkProfile.axes(), fn axis ->
        assert ChunkProfile.axis_default(axis) == Map.fetch!(defaults, axis)
      end)
    end

    test "axis_domain/1 resolves every axis to a usable domain" do
      Enum.each(ChunkProfile.axes(), fn axis ->
        case ChunkProfile.axis_domain(axis) do
          {:enum, values} ->
            assert values != [], "#{axis} resolved to an empty value set"
            assert Enum.all?(values, &is_atom/1)

          {:range, lo, hi} ->
            assert lo < hi

          {:open, _reason} ->
            :ok

          {:error, reason} ->
            flunk(
              "#{axis} has no resolvable value domain (#{inspect(reason)}). A " <>
                "model-backed axis whose classifier is not loaded cannot be " <>
                "validated — every value would appear to pass."
            )
        end
      end)
    end

    test "a categorical axis's declared default is inside its own domain, or is a defaulted marker" do
      # Two axes intentionally default to a value their classifier cannot emit:
      # `domain` defaults to :unknown and intent_domain has no "unknown" label.
      # That is deliberate and useful — observing :unknown then *proves* the
      # axis was defaulted — but it has to be declared, not discovered.
      known_out_of_domain_defaults = [:domain]

      Enum.each(ChunkProfile.axis_manifest(), fn {axis, kind, _d, _s} ->
        with :categorical <- kind,
             {:enum, permitted} <- ChunkProfile.axis_domain(axis) do
          default = ChunkProfile.axis_default(axis)

          if axis in known_out_of_domain_defaults do
            refute default in permitted,
                   "#{axis} is listed as having an out-of-domain default but " <>
                     "#{inspect(default)} is now inside #{inspect(permitted)} — " <>
                     "remove it from known_out_of_domain_defaults"
          else
            assert default in permitted,
                   "#{axis} defaults to #{inspect(default)}, which is not in its " <>
                     "domain #{inspect(permitted)}. Either the domain is wrong or " <>
                     "the default is unreachable."
          end
        end
      end)
    end
  end

  describe "feature_provenance" do
    setup do
      analysis = %ChunkAnalysis{
        chunk_index: 0,
        text: "turn off the kitchen lights",
        speech_act: SpeechActResult.new(:directive, :command, 0.9, is_imperative: true),
        discourse: DiscourseResult.new(:bot, 0.8),
        confidence: 0.8,
        pos_tags: [
          {"turn", "VERB"},
          {"off", "ADP"},
          {"the", "DET"},
          {"kitchen", "NOUN"},
          {"lights", "NOUN"}
        ]
      }

      %{profile: ChunkProfile.materialize(analysis, [])}
    end

    test "records an entry for every declared axis", %{profile: profile} do
      missing = Enum.reject(ChunkProfile.axes(), &Map.has_key?(profile.feature_provenance, &1))

      assert missing == [],
             "no provenance recorded for #{inspect(missing)}. An axis with no " <>
               "provenance cannot be told apart from one that was defaulted."
    end

    test "every entry carries a source and a status", %{profile: profile} do
      Enum.each(ChunkProfile.axes(), fn axis ->
        entry = Map.fetch!(profile.feature_provenance, axis)

        assert is_map(entry), "#{axis} provenance is #{inspect(entry)}, expected a map"

        assert entry[:status] in [:computed, :defaulted],
               "#{axis} has status #{inspect(entry[:status])}"

        assert entry[:source] in [:analysis, :micro_classifier, :composed, :derived],
               "#{axis} has source #{inspect(entry[:source])}"
      end)
    end

    test "a defaulted axis holds exactly its declared default", %{profile: profile} do
      # This is the invariant task 083's corpus-selection rule rests on: "keep
      # the sentence if at least one axis is off its default AND computed". If a
      # defaulted axis could hold a non-default value the rule would silently
      # admit placeholder values as demonstrators.
      Enum.each(ChunkProfile.axes(), fn axis ->
        case Map.fetch!(profile.feature_provenance, axis) do
          %{status: :defaulted} ->
            assert Map.get(profile, axis) == ChunkProfile.axis_default(axis),
                   "#{axis} is marked :defaulted but holds " <>
                     "#{inspect(Map.get(profile, axis))} rather than its default " <>
                     "#{inspect(ChunkProfile.axis_default(axis))}"

          _ ->
            :ok
        end
      end)
    end

    test "a defaulted entry says why", %{profile: profile} do
      Enum.each(ChunkProfile.axes(), fn axis ->
        case Map.fetch!(profile.feature_provenance, axis) do
          %{status: :defaulted} = entry ->
            assert entry[:reason] != nil,
                   "#{axis} is defaulted with no :reason — the point of recording " <>
                     "the default is knowing what to fix"

          _ ->
            :ok
        end
      end)
    end

    test "derived axes name their parents and which of them were defaulted", %{profile: profile} do
      derived =
        ChunkProfile.axis_manifest()
        |> Enum.filter(fn {_a, _k, _d, source} -> source == :derived end)
        |> Enum.map(&elem(&1, 0))

      assert length(derived) == 4

      Enum.each(derived, fn axis ->
        entry = Map.fetch!(profile.feature_provenance, axis)

        assert is_list(entry[:depends_on]) and entry[:depends_on] != [],
               "#{axis} is derived but names no parents"

        assert is_list(entry[:parents_defaulted])

        assert Enum.all?(entry[:parents_defaulted], &(&1 in entry[:depends_on])),
               "#{axis} lists a defaulted parent it does not depend on"
      end)
    end

    test "materialized values stay inside their declared domains", %{profile: profile} do
      # A computed value must be in the axis's domain. A defaulted value is
      # allowed to sit outside it — `domain` deliberately defaults to :unknown,
      # which intent_domain cannot emit, and that is what makes :unknown a
      # reliable marker of a defaulted domain rather than a 14th class.
      Enum.each(ChunkProfile.axes(), fn axis ->
        value = Map.get(profile, axis)
        defaulted? = Map.fetch!(profile.feature_provenance, axis)[:status] == :defaulted

        case ChunkProfile.axis_domain(axis) do
          {:enum, permitted} ->
            assert value in permitted or (defaulted? and value == ChunkProfile.axis_default(axis)),
                   "#{axis} produced #{inspect(value)}, outside its declared domain " <>
                     "#{inspect(permitted)}, and it is not a defaulted value " <>
                     "(status #{inspect(Map.fetch!(profile.feature_provenance, axis)[:status])})"

          {:range, lo, hi} ->
            assert is_number(value) and value >= lo and value <= hi,
                   "#{axis} produced #{inspect(value)}, outside #{lo}..#{hi}"

          _ ->
            :ok
        end
      end)
    end

    test "polarity records both tag spellings so tasks 074/081 stay separable", %{
      profile: profile
    } do
      entry = Map.fetch!(profile.feature_provenance, :polarity)
      evidence = entry[:evidence]

      assert evidence[:pos_tag_count] == 5
      assert Map.has_key?(evidence, :atom_part)
      assert Map.has_key?(evidence, :string_part)
    end
  end
end
