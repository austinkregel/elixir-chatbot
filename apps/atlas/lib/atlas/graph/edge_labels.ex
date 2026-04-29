defmodule Atlas.Graph.EdgeLabels do
  @moduledoc """
  Canonical edge/relationship labels for all Atlas graphs.

  All labels use SCREAMING_SNAKE_CASE. Every writer and reader must
  reference these constants instead of inline strings. PredicateNormalizer
  maps all variants (PascalCase, snake_case) to these canonical forms.

  Module attributes (@is_a etc.) are available for pattern matching
  within this module. Other modules use the zero-arity functions.

  `all/0` covers only statically-defined labels. Dynamic labels (SRL edges,
  event frame edges like "HAS_" <> arg_role, epistemic rel_type passthrough,
  and the open-set upcase fallback in preference_rel_type/1) are not
  enumerable at compile time.
  """

  # -- Knowledge Graph --
  @is_a "IS_A"
  @has_subtype "HAS_SUBTYPE"
  @instance_of "INSTANCE_OF"
  @alias_of "ALIAS_OF"
  @co_occurs_with "CO_OCCURS_WITH"
  @actor "ACTOR"
  @acts_on "ACTS_ON"
  @has_lexicon_facet "HAS_LEXICON_FACET"

  # -- Semantic Graph --
  @evidence_for "EVIDENCE_FOR"

  # -- Conversation Graph --
  @contains "CONTAINS"
  @follows "FOLLOWS"
  @has_topic "HAS_TOPIC"
  @topic_transition "TOPIC_TRANSITION"

  # -- Epistemic Graph --
  @supports "SUPPORTS"
  @requires_in "REQUIRES_IN"
  @requires_out "REQUIRES_OUT"
  @contradicts "CONTRADICTS"

  # -- POS Graph --
  @has_tag "HAS_TAG"
  @followed_by "FOLLOWED_BY"

  # -- User Graph --
  @likes "LIKES"
  @wants "WANTS"
  @interested_in "INTERESTED_IN"
  @needs "NEEDS"
  @dislikes "DISLIKES"
  @asked_about "ASKED_ABOUT"

  def is_a, do: @is_a
  def has_subtype, do: @has_subtype
  def instance_of, do: @instance_of
  def alias_of, do: @alias_of
  def co_occurs_with, do: @co_occurs_with
  def actor, do: @actor
  def acts_on, do: @acts_on
  def has_lexicon_facet, do: @has_lexicon_facet
  def evidence_for, do: @evidence_for
  def contains, do: @contains
  def follows, do: @follows
  def has_topic, do: @has_topic
  def topic_transition, do: @topic_transition
  def supports, do: @supports
  def requires_in, do: @requires_in
  def requires_out, do: @requires_out
  def contradicts, do: @contradicts
  def has_tag, do: @has_tag
  def followed_by, do: @followed_by
  def likes, do: @likes
  def wants, do: @wants
  def interested_in, do: @interested_in
  def needs, do: @needs
  def dislikes, do: @dislikes
  def asked_about, do: @asked_about

  @doc "All statically-defined labels. Does NOT include dynamic SRL/event/epistemic labels."
  def all do
    [@is_a, @has_subtype, @instance_of, @alias_of, @co_occurs_with,
     @actor, @acts_on, @has_lexicon_facet, @evidence_for, @contains,
     @follows, @has_topic, @topic_transition, @supports, @requires_in,
     @requires_out, @contradicts, @has_tag, @followed_by, @likes,
     @wants, @interested_in, @needs, @dislikes, @asked_about]
  end

  @doc """
  KG-relevant relation labels for triple scorer training.
  Excludes conversation-graph, POS-graph, and epistemic-graph labels
  that never appear in knowledge graph triples.
  """
  def kg_relations do
    [@is_a, @has_subtype, @instance_of, @co_occurs_with,
     @actor, @acts_on, @likes, @wants, @interested_in,
     @needs, @dislikes]
  end
end
