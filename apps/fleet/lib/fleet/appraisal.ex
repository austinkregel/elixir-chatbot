defmodule Fleet.Appraisal do
  @moduledoc """
  Value-grounded DISSENT judgment: does the agent's SOUL permit carrying out this
  order? Runs inside the cognition Task (so it never blocks the mailbox), after
  ACK and before `Brain.evaluate`. Returns `:proceed` or `{:dissent, verdict}`.

  Two tiers, leading with the real, structured, deterministic mechanism:

    * **Tier 1 (deterministic):** the soul's `genome` declares a real value
      vocabulary — `prohibited_terms` (token-matched), `prohibited_speech_acts`
      (checked against `Brain.Analysis.SpeechActClassifier`'s typed category /
      sub_type), and data-provenance refusal. No fragile free-text parsing.
    * **Tier 2 (constrained Brain verdict, opt-in):** when the soul's genome sets
      `deep_appraisal: true`, the constitution + order are put to the Brain with a
      single leading-token contract (`PROCEED` | `DISSENT: <reason>`), parsed
      deterministically, defaulting on ambiguity per the soul's `deference`.

  Tier 2 is gated behind `deep_appraisal` rather than run on every order — value
  appraisal via cognition would otherwise double every order's cognition cost.
  """

  require Logger
  alias Fleet.Order
  alias Brain.Analysis.SpeechActClassifier

  @doc "Appraise an order against a soul. `:proceed | {:dissent, verdict_map}`."
  def appraise(%Order{} = order, soul) do
    genome = genome_of(soul)
    directive = to_string(order.directive || "")

    cond do
      data_provenance?(order) ->
        {:dissent, %{basis: :provenance, rule: :data_embedded,
                     reason: "directive provenance is data, not a command"}}

      term = prohibited_term(directive, genome) ->
        {:dissent, %{basis: :value, rule: :prohibited_term,
                     reason: "directive contains prohibited term: #{term}"}}

      sa = prohibited_speech_act(directive, genome) ->
        {:dissent, %{basis: :value, rule: :prohibited_speech_act,
                     reason: "directive is a prohibited speech act: #{sa}"}}

      deep_appraisal?(genome) ->
        cognition_verdict(order, soul, genome)

      true ->
        :proceed
    end
  end

  # ── Tier 1 helpers ────────────────────────────────────────────────────────

  defp data_provenance?(%Order{grant: grant}) when is_map(grant),
    do: Map.get(grant, :provenance) == :data

  defp data_provenance?(_), do: false

  defp prohibited_term(directive, genome) do
    lowered = String.downcase(directive)

    genome
    |> list_field("prohibited_terms")
    |> Enum.find(fn term -> String.contains?(lowered, String.downcase(to_string(term))) end)
  end

  defp prohibited_speech_act(directive, genome) do
    case list_field(genome, "prohibited_speech_acts") do
      [] ->
        nil

      prohibited ->
        result = SpeechActClassifier.classify(directive)
        cat = result.category |> to_string()
        sub = result.sub_type |> to_string()
        prohibited = Enum.map(prohibited, &to_string/1)

        cond do
          cat in prohibited -> cat
          sub in prohibited -> sub
          true -> nil
        end
    end
  end

  # ── Tier 2: constrained Brain verdict ─────────────────────────────────────

  defp cognition_verdict(%Order{} = order, soul, genome) do
    prompt = verdict_prompt(soul, order)

    with {:ok, conv} <- Brain.create_conversation(world_id: order.world_id) do
      text = order |> run_eval(conv, prompt) |> extract_text()
      parse_verdict(text, genome)
    else
      other ->
        Logger.warning("Fleet.Appraisal: verdict cognition setup failed", reason: inspect(other))
        # Conservative default per deference when we cannot obtain a verdict.
        default_on_ambiguity(genome, "verdict cognition unavailable")
    end
  end

  defp run_eval(_order, conv, prompt), do: Brain.evaluate(conv, prompt, [])

  defp verdict_prompt(soul, %Order{} = order) do
    constitution = (soul && Brain.Soul.system_prompt(soul)) || ""

    """
    #{constitution}

    You have received the following ORDER:
    "#{order.directive}"

    Decide, strictly against your constitution above, whether you may lawfully and
    ethically carry it out. Answer with a SINGLE leading token:
    - Reply exactly `PROCEED` if you may carry it out.
    - Reply `DISSENT: <reason>` if you must refuse.
    """
  end

  defp parse_verdict(text, genome) do
    trimmed = text |> to_string() |> String.trim()

    cond do
      String.starts_with?(String.upcase(trimmed), "PROCEED") ->
        :proceed

      String.starts_with?(String.upcase(trimmed), "DISSENT") ->
        reason = trimmed |> String.split(":", parts: 2) |> List.last() |> String.trim()
        {:dissent, %{basis: :value, rule: :cognition_verdict, reason: reason}}

      true ->
        default_on_ambiguity(genome, "verdict ambiguous: #{String.slice(trimmed, 0, 80)}")
    end
  end

  # Low deference → refuse on ambiguity; high deference → proceed.
  defp default_on_ambiguity(genome, reason) do
    deference = number_field(genome, "deference", 0.5)

    if deference < 0.5 do
      {:dissent, %{basis: :value, rule: :ambiguous_verdict, reason: reason}}
    else
      :proceed
    end
  end

  # ── genome accessors ──────────────────────────────────────────────────────

  defp genome_of(%{genome: g}) when is_map(g), do: g
  defp genome_of(_), do: %{}

  defp deep_appraisal?(genome), do: Map.get(genome, "deep_appraisal") == true

  defp list_field(genome, key) do
    case Map.get(genome, key) do
      list when is_list(list) -> list
      _ -> []
    end
  end

  defp number_field(genome, key, default) do
    case Map.get(genome, key) do
      n when is_number(n) -> n
      _ -> default
    end
  end

  defp extract_text({:ok, %{response: r}}) when is_binary(r), do: r
  defp extract_text({:ok, r}) when is_binary(r), do: r
  defp extract_text(%{response: r}) when is_binary(r), do: r
  defp extract_text(r) when is_binary(r), do: r
  defp extract_text(other), do: inspect(other)
end
