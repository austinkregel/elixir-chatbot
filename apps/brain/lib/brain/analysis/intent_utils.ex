defmodule Brain.Analysis.IntentUtils do
  @moduledoc """
  Shared helpers for comparing intent labels (exact match, domain prefix).
  """

  @doc """
  True when both intent strings share the same domain prefix (first dot segment).
  """
  def same_domain_prefix?(intent_a, intent_b)
      when is_binary(intent_a) and is_binary(intent_b) and intent_a != "" and intent_b != "" do
    domain_a = domain_prefix(intent_a)
    domain_b = domain_prefix(intent_b)
    domain_a != "" and domain_a == domain_b
  end

  def same_domain_prefix?(_, _), do: false

  @doc "Returns the first segment of an intent label (e.g. `weather` from `weather.query`)."
  def domain_prefix(intent) when is_binary(intent) do
    case String.split(intent, ".", parts: 2) do
      [domain, _] -> domain
      [domain] -> domain
      _ -> ""
    end
  end

  def domain_prefix(intent) when is_atom(intent), do: domain_prefix(Atom.to_string(intent))
  def domain_prefix(_), do: ""
end
