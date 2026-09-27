defmodule Brain.Response.ResponseSystemRouter do
  @moduledoc """
  Selects which response system to use based on per-domain configuration
  stored in system_config.json.

  Each domain can be configured to use :lattice, :ouro, or :template
  as its primary response system, with a fallback.
  """

  alias Brain.Response.{PhraseInventory, LatticeScorer}

  require Logger

  @config_path "priv/response/system_config.json"

  @doc """
  Routes a response realization request to the appropriate system.

  Returns `{system, domain_config}` where:
  - system is :lattice, :ouro, or :template
  - domain_config includes tone_bias and mirror_coefficient
  """
  def route(domain, _opts \\ []) do
    config = get_domain_config(domain)
    system = resolve_system(config)
    {system, config}
  end

  @doc "Returns the domain configuration map."
  def get_domain_config(domain) do
    all_config = load_config()
    domains = Map.get(all_config, "domains", %{})
    domain_str = to_string(domain)

    domain_config = Map.get(domains, domain_str, Map.get(domains, "_default", %{}))
    default_config = Map.get(domains, "_default", %{})
    tone_config = Map.get(all_config, "tone", %{})

    %{
      system: Map.get(domain_config, "system", "template") |> String.to_atom(),
      fallback: Map.get(domain_config, "fallback", "template") |> String.to_atom(),
      tone_bias: Map.get(domain_config, "tone_bias", Map.get(default_config, "tone_bias", "calm")),
      mirror_coefficient: Map.get(domain_config, "mirror_coefficient",
        Map.get(tone_config, "mirror_coefficient", 0.4)),
      tone_vectors: Map.get(tone_config, "tone_vectors", %{})
    }
  end

  @doc "Returns the 10-dim tone bias vector for a domain."
  def get_tone_bias_vector(domain) do
    config = get_domain_config(domain)
    tone_name = config.tone_bias
    vectors = config.tone_vectors
    Map.get(vectors, tone_name, List.duplicate(0.5, 10))
  end

  @doc "Computes the desired tone vector for a domain given input sentiment."
  def compute_tone(domain, input_sentiment) do
    config = get_domain_config(domain)
    tone_bias_vector = Map.get(config.tone_vectors, config.tone_bias, List.duplicate(0.5, 10))
    LatticeScorer.compute_desired_tone(input_sentiment, tone_bias_vector, config.mirror_coefficient)
  end

  @doc "Updates a domain's config and persists to disk."
  def update_domain_config(domain, changes) when is_map(changes) do
    all_config = load_config()
    domains = Map.get(all_config, "domains", %{})
    domain_str = to_string(domain)

    current = Map.get(domains, domain_str, %{})
    updated = Map.merge(current, changes)

    new_domains = Map.put(domains, domain_str, updated)
    new_config = Map.put(all_config, "domains", new_domains)

    save_config(new_config)
  end

  @doc "Lists all configured domains with their settings."
  def list_domains do
    all_config = load_config()
    domains = Map.get(all_config, "domains", %{})

    Enum.map(domains, fn {domain, config} ->
      %{
        domain: domain,
        system: Map.get(config, "system", "template"),
        fallback: Map.get(config, "fallback", "template"),
        tone_bias: Map.get(config, "tone_bias", "calm"),
        mirror_coefficient: Map.get(config, "mirror_coefficient", 0.4)
      }
    end)
    |> Enum.reject(fn d -> d.domain == "_default" end)
    |> Enum.sort_by(& &1.domain)
  end

  defp resolve_system(%{system: :lattice} = config) do
    if PhraseInventory.ready?() do
      :lattice
    else
      Logger.debug("ResponseSystemRouter: lattice requested but PhraseInventory not ready, falling back")
      config.fallback
    end
  end

  defp resolve_system(%{system: system}), do: system

  defp load_config do
    path = brain_priv(@config_path)

    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, config} -> config
          _ -> default_config()
        end

      {:error, _} ->
        default_config()
    end
  end

  defp save_config(config) do
    path = brain_priv(@config_path)
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Jason.encode!(config, pretty: true) <> "\n")
    :ok
  end

  defp default_config do
    %{
      "version" => 1,
      "tone" => %{
        "mirror_coefficient" => 0.4,
        "default_tone_bias" => "calm",
        "tone_vectors" => %{}
      },
      "domains" => %{
        "_default" => %{
          "system" => "template",
          "fallback" => "template",
          "tone_bias" => "calm",
          "mirror_coefficient" => 0.4
        }
      }
    }
  end

  defp brain_priv(relative) do
    case :code.priv_dir(:brain) do
      {:error, _} -> Path.join("apps/brain", relative)
      priv_dir -> Path.join(priv_dir, Path.relative_to(relative, "priv"))
    end
  end
end
