defmodule ChatBot.Knowledge.HtmlProcessor do
  @moduledoc """
  Processes HTML content into clean text or markdown.

  Used by the Research Agent to clean web content before extracting
  factual claims. Removes scripts, styles, navigation, and other
  non-content elements while preserving meaningful text structure.
  """

  require Logger

  # Tags that contain no useful content
  @remove_tags ~w(script style noscript iframe svg canvas audio video 
                  nav header footer aside form button input select 
                  textarea label)

  # Tags that typically contain navigation/boilerplate
  @boilerplate_classes ~w(nav navigation menu sidebar footer header 
                          advertisement ad-container cookie-banner
                          social-share comment comments related-posts
                          breadcrumb pagination search)

  # Block elements that should have paragraph breaks
  @block_tags ~w(p div section article h1 h2 h3 h4 h5 h6 li blockquote 
                 tr td th dt dd figcaption)

  # Minimum content length to consider valid
  @min_content_length 50

  @doc """
  Converts HTML content to clean, readable text.

  Removes scripts, styles, and other non-content elements.
  Preserves paragraph structure with newlines.

  ## Options
    - :preserve_links - If true, converts links to markdown format (default: false)
    - :preserve_headers - If true, converts headers to markdown format (default: true)
    - :min_length - Minimum length for content to be returned (default: 50)
  """
  @spec html_to_text(String.t(), keyword()) :: {:ok, String.t()} | {:error, :no_content}
  def html_to_text(html, opts \\ []) when is_binary(html) do
    preserve_links = Keyword.get(opts, :preserve_links, false)
    preserve_headers = Keyword.get(opts, :preserve_headers, true)
    min_length = Keyword.get(opts, :min_length, @min_content_length)

    try do
      # Parse HTML
      {:ok, document} = Floki.parse_document(html)

      # Remove unwanted elements
      cleaned = remove_unwanted_elements(document)

      # Extract main content area if possible
      main_content = extract_main_content(cleaned)

      # Convert to text
      text =
        main_content
        |> extract_text(preserve_links, preserve_headers)
        |> clean_whitespace()
        |> String.trim()

      if String.length(text) >= min_length do
        {:ok, text}
      else
        {:error, :no_content}
      end
    rescue
      e ->
        Logger.warning("Failed to parse HTML", error: Exception.message(e))
        # Fallback to basic tag stripping
        fallback_strip(html, min_length)
    end
  end

  @doc """
  Converts HTML to markdown format.

  Preserves structure like headers, lists, links, and emphasis.
  """
  @spec html_to_markdown(String.t(), keyword()) :: {:ok, String.t()} | {:error, :no_content}
  def html_to_markdown(html, opts \\ []) do
    html_to_text(html, Keyword.merge([preserve_links: true, preserve_headers: true], opts))
  end

  @doc """
  Checks if content appears to be HTML.
  """
  @spec is_html?(String.t()) :: boolean()
  def is_html?(content) when is_binary(content) do
    # Check for common HTML indicators
    String.contains?(content, "<") and
      (String.contains?(content, "</") or
         String.contains?(content, "/>") or
         String.contains?(content, "<!DOCTYPE") or
         String.contains?(content, "<html") or
         String.contains?(content, "<body") or
         String.contains?(content, "<div") or
         String.contains?(content, "<p>"))
  end

  def is_html?(_), do: false

  @doc """
  Extracts just the main article/content text from HTML, aggressively
  filtering out boilerplate.
  """
  @spec extract_article_text(String.t()) :: {:ok, String.t()} | {:error, :no_content}
  def extract_article_text(html) when is_binary(html) do
    try do
      {:ok, document} = Floki.parse_document(html)

      # Try to find the main content area
      main_content =
        find_main_content(document) ||
          find_article_content(document) ||
          remove_unwanted_elements(document)

      text =
        main_content
        |> extract_text(false, true)
        |> clean_whitespace()
        |> filter_short_lines()
        |> String.trim()

      if String.length(text) >= @min_content_length do
        {:ok, text}
      else
        {:error, :no_content}
      end
    rescue
      _ -> {:error, :no_content}
    end
  end

  # ============================================================================
  # Private - Element Removal
  # ============================================================================

  defp remove_unwanted_elements(document) do
    document
    |> remove_by_tags(@remove_tags)
    |> remove_by_classes(@boilerplate_classes)
    |> remove_by_ids(~w(nav navigation menu sidebar footer header comments))
  end

  defp remove_by_tags(document, tags) do
    Enum.reduce(tags, document, fn tag, doc ->
      Floki.filter_out(doc, tag)
    end)
  end

  defp remove_by_classes(document, classes) do
    Enum.reduce(classes, document, fn class, doc ->
      Floki.filter_out(doc, "[class*='#{class}']")
    end)
  end

  defp remove_by_ids(document, ids) do
    Enum.reduce(ids, document, fn id, doc ->
      Floki.filter_out(doc, "##{id}")
    end)
  end

  # ============================================================================
  # Private - Content Extraction
  # ============================================================================

  defp find_main_content(document) do
    # Try common main content selectors
    selectors = ["main", "[role='main']", "#content", "#main-content", ".main-content"]

    Enum.find_value(selectors, fn selector ->
      case Floki.find(document, selector) do
        [] -> nil
        found -> found
      end
    end)
  end

  defp find_article_content(document) do
    # Try article-specific selectors
    selectors = ["article", ".article", ".post-content", ".entry-content", ".article-body"]

    Enum.find_value(selectors, fn selector ->
      case Floki.find(document, selector) do
        [] -> nil
        found -> found
      end
    end)
  end

  defp extract_main_content(document) do
    # If we find a main content area, use it; otherwise use the whole cleaned doc
    find_main_content(document) ||
      find_article_content(document) ||
      # Try body if nothing else
      case Floki.find(document, "body") do
        [] -> document
        body -> body
      end
  end

  defp extract_text(nodes, preserve_links, preserve_headers) do
    nodes
    |> Floki.traverse_and_update(fn
      # Handle headers
      {tag, attrs, children} when tag in ~w(h1 h2 h3 h4 h5 h6) ->
        if preserve_headers do
          level = String.to_integer(String.last(tag))
          prefix = String.duplicate("#", level) <> " "
          text = Floki.text(children, sep: " ")
          {"span", attrs, ["\n\n#{prefix}#{text}\n\n"]}
        else
          {tag, attrs, children}
        end

      # Handle links
      {"a", attrs, children} ->
        if preserve_links do
          href = attrs |> Enum.find_value(fn {k, v} -> if k == "href", do: v end)
          text = Floki.text(children, sep: " ")

          if href && String.starts_with?(href, "http") do
            {"span", [], ["[#{text}](#{href})"]}
          else
            {"span", [], [text]}
          end
        else
          {"a", attrs, children}
        end

      # Handle list items
      {"li", attrs, children} ->
        text = Floki.text(children, sep: " ")
        {"span", attrs, ["\n• #{text}"]}

      # Handle paragraphs and block elements
      {tag, attrs, children} when tag in @block_tags ->
        {tag, attrs, children ++ ["\n\n"]}

      # Handle line breaks
      {"br", _, _} ->
        {"span", [], ["\n"]}

      # Pass through other elements
      other ->
        other
    end)
    |> Floki.text(sep: " ")
  end

  # ============================================================================
  # Private - Text Cleaning
  # ============================================================================

  defp clean_whitespace(text) do
    text
    # Normalize various whitespace
    |> String.replace(~r/[\t\r]+/, " ")
    # Multiple spaces to single
    |> String.replace(~r/ {2,}/, " ")
    # Multiple newlines to double
    |> String.replace(~r/\n{3,}/, "\n\n")
    # Space at start of line
    |> String.replace(~r/\n +/, "\n")
    # Trim lines
    |> String.split("\n")
    |> Enum.map(&String.trim/1)
    |> Enum.join("\n")
  end

  defp filter_short_lines(text) do
    # Filter out very short lines that are likely navigation/boilerplate
    text
    |> String.split("\n")
    |> Enum.filter(fn line ->
      trimmed = String.trim(line)
      # Keep if: header, list item, or substantial text
      String.starts_with?(trimmed, "#") or
        String.starts_with?(trimmed, "•") or
        String.length(trimmed) > 30
    end)
    |> Enum.join("\n")
  end

  defp fallback_strip(html, min_length) do
    # Simple regex-based fallback for malformed HTML
    text =
      html
      # Remove script/style content
      |> String.replace(~r/<script[^>]*>.*?<\/script>/is, " ")
      |> String.replace(~r/<style[^>]*>.*?<\/style>/is, " ")
      # Remove all tags
      |> String.replace(~r/<[^>]+>/, " ")
      # Decode common entities
      |> String.replace("&nbsp;", " ")
      |> String.replace("&amp;", "&")
      |> String.replace("&lt;", "<")
      |> String.replace("&gt;", ">")
      |> String.replace("&quot;", "\"")
      |> String.replace("&#39;", "'")
      # Clean whitespace
      |> String.replace(~r/\s+/, " ")
      |> String.trim()

    if String.length(text) >= min_length do
      {:ok, text}
    else
      {:error, :no_content}
    end
  end
end
