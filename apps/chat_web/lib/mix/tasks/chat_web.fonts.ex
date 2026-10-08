defmodule Mix.Tasks.ChatWeb.Fonts do
  @shortdoc "Publishes the IBM Plex fonts from assets/node_modules into priv/static/fonts"

  @moduledoc """
  Publishes the IBM Plex font files the design tokens use, and their license,
  from the npm packages installed under `assets/node_modules` into
  `priv/static/fonts`, where the `@font-face` rules in `assets/css/app.css`
  load them from `/fonts/`.

      mix chat_web.fonts

  The packages are `@ibm/plex-sans` and `@ibm/plex-mono`, pinned in
  `assets/package.json` and installed by `mix assets.setup`. Every source file
  must exist; a missing one raises naming each absent path, so a build never
  publishes a partial font set.

  Both packages ship the same SIL Open Font License text. It is published once,
  as `IBM-Plex-OFL.txt`, and the task raises if the two packages' copies ever
  differ.
  """

  use Mix.Task

  @packages_dir "assets/node_modules/@ibm"
  @output_dir "priv/static/fonts"

  @fonts [
    {"plex-sans/fonts/complete/woff2/IBMPlexSans-Regular.woff2", "IBMPlexSans-Regular.woff2"},
    {"plex-sans/fonts/complete/woff2/IBMPlexSans-SemiBold.woff2", "IBMPlexSans-SemiBold.woff2"},
    {"plex-mono/fonts/complete/woff2/IBMPlexMono-Regular.woff2", "IBMPlexMono-Regular.woff2"},
    {"plex-mono/fonts/complete/woff2/IBMPlexMono-Medium.woff2", "IBMPlexMono-Medium.woff2"},
    {"plex-mono/fonts/complete/woff2/IBMPlexMono-SemiBold.woff2", "IBMPlexMono-SemiBold.woff2"}
  ]

  @licenses ["plex-sans/LICENSE.txt", "plex-mono/LICENSE.txt"]
  @license_output "IBM-Plex-OFL.txt"

  @impl Mix.Task
  def run(_args) do
    # The chat_web app root, wherever the task is invoked from (an umbrella
    # `mix do --app chat_web` runs it with the umbrella as the current project).
    root = Path.expand("../../..", __DIR__)
    packages = Path.join(root, @packages_dir)
    output = Path.join(root, @output_dir)

    fonts = Enum.map(@fonts, fn {source, name} -> {Path.join(packages, source), name} end)
    licenses = Enum.map(@licenses, &Path.join(packages, &1))

    missing = Enum.reject(Enum.map(fonts, &elem(&1, 0)) ++ licenses, &File.regular?/1)

    if missing != [] do
      Mix.raise("""
      Cannot publish fonts: these files are missing.

      #{Enum.map_join(missing, "\n", &("  " <> &1))}

      They come from the npm packages pinned in assets/package.json. Run `mix assets.setup` to install them.
      """)
    end

    [license | other_licenses] = licenses
    license_text = File.read!(license)

    differing = Enum.reject(other_licenses, &(File.read!(&1) == license_text))

    if differing != [] do
      Mix.raise("""
      Cannot publish fonts: the packages ship different license texts, and only one is published as #{@license_output}.

        #{license}
      differs from
      #{Enum.map_join(differing, "\n", &("  " <> &1))}
      """)
    end

    File.mkdir_p!(output)

    Enum.each(fonts, fn {source, name} -> File.cp!(source, Path.join(output, name)) end)
    File.cp!(license, Path.join(output, @license_output))

    Mix.shell().info("Published #{length(fonts)} fonts and #{@license_output} to #{output}")
  end
end
