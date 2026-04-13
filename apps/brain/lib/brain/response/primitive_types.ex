defmodule Brain.Response.PrimitiveTypes do
  @moduledoc """
  Content schemas and validation for each primitive type+variant.

  Each primitive type handles one aspect of turning analysis into language.
  The `valid?/1` function checks that required content fields are present.
  The `required_content/2` function returns the required fields for a type+variant.

  ## Type taxonomy

  Types are organized by **communicative function**:

    - `:acknowledgment` -- recognizing what the user said or did
    - `:framing` -- setting the pragmatic frame for what follows
    - `:hedging` -- epistemic calibration
    - `:content` -- the substantive body of the response
    - `:attunement` -- emotional/relational engagement
    - `:follow_up` -- driving the conversation forward
    - `:contradiction_response` -- handling conflicting information
    - `:transition` -- connecting parts of multi-part responses
  """

  alias Brain.Response.Primitive

  @acknowledgment_variants [:social, :action, :learning, :repair, :general]
  @framing_variants [:affirmative, :negative, :informative, :boundary, :reframe]
  @content_variants [:factual, :explanatory, :reflective, :narrative, :creative, :action_result]
  @attunement_variants [:empathy, :validation, :interest, :concern]
  @follow_up_variants [:clarification, :elaboration, :correction_invite, :continuation, :context_probe]

  @doc "Returns all valid variants for a given primitive type."
  def variants(:acknowledgment), do: @acknowledgment_variants
  def variants(:framing), do: @framing_variants
  def variants(:hedging), do: [nil]
  def variants(:content), do: @content_variants
  def variants(:attunement), do: @attunement_variants
  def variants(:follow_up), do: @follow_up_variants
  def variants(:contradiction_response), do: [nil]
  def variants(:transition), do: [nil]
  def variants(_), do: []

  @doc "Returns the list of required content fields for a type+variant."
  def required_content(:acknowledgment, :social), do: [:speech_act_sub_type]
  def required_content(:acknowledgment, :action), do: [:action, :capability]
  def required_content(:acknowledgment, :learning), do: [:learned_fact]
  def required_content(:acknowledgment, :repair), do: [:what_went_wrong]
  def required_content(:acknowledgment, :general), do: []

  def required_content(:framing, :affirmative), do: [:confirmed_fact]
  def required_content(:framing, :negative), do: [:actual_fact, :user_claim]
  def required_content(:framing, :informative), do: [:topic]
  def required_content(:framing, :boundary), do: [:capability]
  def required_content(:framing, :reframe), do: [:original_question_type, :offered_alternative]

  def required_content(:hedging, _), do: [:confidence_level]

  def required_content(:content, :factual), do: [:fact]
  def required_content(:content, :explanatory), do: [:topic]
  def required_content(:content, :reflective), do: [:understood_meaning]
  def required_content(:content, :narrative), do: [:beliefs]
  def required_content(:content, :creative), do: [:prompt_type]
  def required_content(:content, :action_result), do: [:action, :result]

  def required_content(:attunement, :empathy), do: [:sentiment_label]
  def required_content(:attunement, :validation), do: [:experience_summary]
  def required_content(:attunement, :interest), do: [:topic]
  def required_content(:attunement, :concern), do: [:frustration_source]

  def required_content(:follow_up, :clarification), do: []
  def required_content(:follow_up, :elaboration), do: [:topic]
  def required_content(:follow_up, :correction_invite), do: []
  def required_content(:follow_up, :continuation), do: []
  def required_content(:follow_up, :context_probe), do: []

  def required_content(:contradiction_response, _), do: [:existing_belief, :new_claim]

  def required_content(:transition, _), do: [:from_speech_act, :to_speech_act]

  def required_content(_, _), do: []

  @doc "Checks whether a primitive has valid type, variant, and required content fields."
  def valid?(%Primitive{type: type, variant: variant, content: content}) do
    valid_variant?(type, variant) and has_required_content?(type, variant, content)
  end

  def valid?(_), do: false

  @doc "Returns all known primitive types."
  def all_types do
    [:acknowledgment, :framing, :hedging, :content, :attunement, :follow_up,
     :contradiction_response, :transition]
  end

  defp valid_variant?(type, variant) do
    variant in variants(type)
  end

  defp has_required_content?(type, variant, content) when is_map(content) do
    required = required_content(type, variant)
    Enum.all?(required, &Map.has_key?(content, &1))
  end

  defp has_required_content?(_, _, _), do: false
end
