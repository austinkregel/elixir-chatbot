defmodule Brain.Knowledge.HtmlProcessorTest do
  use ExUnit.Case, async: false

  alias Brain.Knowledge.HtmlProcessor

  describe "is_html?/1" do
    test "detects basic HTML" do
      assert HtmlProcessor.is_html?("<html><body>Hello</body></html>")
      assert HtmlProcessor.is_html?("<div>Content</div>")
      assert HtmlProcessor.is_html?("<p>Paragraph</p>")
      assert HtmlProcessor.is_html?("<!DOCTYPE html><html>")
    end

    test "returns false for plain text" do
      refute HtmlProcessor.is_html?("Just plain text")
      refute HtmlProcessor.is_html?("No tags here, just words.")
    end

    test "returns false for nil" do
      refute HtmlProcessor.is_html?(nil)
    end
  end

  describe "html_to_text/2" do
    test "extracts text from simple HTML" do
      html = "<html><body><p>Hello World, this is a longer sentence with enough content.</p></body></html>"
      assert {:ok, text} = HtmlProcessor.html_to_text(html, min_length: 20)
      assert text =~ "Hello World"
    end

    test "removes script and style content" do
      html = """
      <html>
        <head>
          <script>var x = 'should be removed';</script>
          <style>.class { color: red; }</style>
        </head>
        <body>
          <p>This is the actual content that should remain and it has enough text to pass the minimum length check.</p>
        </body>
      </html>
      """

      assert {:ok, text} = HtmlProcessor.html_to_text(html, min_length: 20)
      assert text =~ "actual content"
      refute text =~ "should be removed"
      refute text =~ "color: red"
    end

    test "removes navigation elements" do
      html = """
      <html>
        <body>
          <nav><a href="/">Home</a><a href="/about">About</a></nav>
          <main>
            <article>
              <p>This is the main article content that we want to extract from the page.</p>
            </article>
          </main>
          <footer>Copyright 2024</footer>
        </body>
      </html>
      """

      assert {:ok, text} = HtmlProcessor.html_to_text(html)
      assert text =~ "main article content"
      # Nav content should be removed
      refute text =~ "Home"
    end

    test "preserves headers when requested" do
      html = "<h1>Main Title</h1><p>Content here that is long enough to pass validation checks.</p>"
      assert {:ok, text} = HtmlProcessor.html_to_text(html, preserve_headers: true, min_length: 20)
      assert text =~ "# Main Title"
    end

    test "returns error for content below min length" do
      html = "<p>Hi</p>"
      assert {:error, :no_content} = HtmlProcessor.html_to_text(html, min_length: 50)
    end
  end

  describe "html_to_markdown/2" do
    test "converts links to markdown format" do
      html = """
      <p>Check out <a href="https://example.com">this link</a> for more info.</p>
      """

      assert {:ok, text} = HtmlProcessor.html_to_markdown(html, min_length: 10)
      assert text =~ "[this link](https://example.com)"
    end

    test "converts headers to markdown" do
      html = "<h2>Section Title</h2><p>Section content goes here.</p>"
      assert {:ok, text} = HtmlProcessor.html_to_markdown(html, min_length: 10)
      assert text =~ "## Section Title"
    end
  end

  describe "extract_article_text/1" do
    test "extracts main content from article" do
      html = """
      <html>
        <body>
          <header>Site Header</header>
          <nav>Menu Items</nav>
          <main>
            <article>
              <h1>Article Title</h1>
              <p>This is a detailed article about an important topic that contains substantial content.</p>
            </article>
          </main>
          <aside>Sidebar content</aside>
          <footer>Footer stuff</footer>
        </body>
      </html>
      """

      assert {:ok, text} = HtmlProcessor.extract_article_text(html)
      assert text =~ "important topic"
      assert text =~ "Article Title"
    end

    test "handles HTML with class-based content areas" do
      html = """
      <div class="navigation">Skip this</div>
      <div class="article-body">
        This is the real content of the article that should be extracted.
      </div>
      <div class="comments">Comment section to ignore</div>
      """

      assert {:ok, text} = HtmlProcessor.extract_article_text(html)
      assert text =~ "real content"
    end
  end

  describe "cleaning real-world HTML fragments" do
    test "cleans britannica-style HTML" do
      # Simulating the kind of HTML fragments seen in the review queue
      html = """
      <div class="mb-45 RESULT-5" data-topic-id="195686">
        <a class="font-weight-bold font-18" href="/place/Europe">Europe (continent)</a>
        <div class="mt-5 font-weight-normal">
          Europe is the second smallest continent, comprising the western fifth of the Eurasian landmass.
        </div>
      </div>
      """

      assert {:ok, text} = HtmlProcessor.html_to_text(html, min_length: 20)
      assert text =~ "Europe"
      assert text =~ "second smallest continent"
      refute text =~ "RESULT-5"
      refute text =~ "data-topic-id"
    end

    test "cleans form and search elements" do
      html = """
      <form method="get" action="/search">
        <input name="query" placeholder="Search...">
        <button>Search</button>
      </form>
      <div class="content-area">
        <p>Paris is the capital of France and its largest city. This is a longer paragraph with more content to meet the minimum length requirement.</p>
      </div>
      """

      assert {:ok, text} = HtmlProcessor.html_to_text(html, min_length: 20)
      assert text =~ "Paris is the capital"
      # Form elements should be removed
      refute text =~ "placeholder"
    end

    test "handles XMLHttpRequest and JavaScript fragments" do
      # These should be rejected entirely
      html = "r.load(e)}}else{var o=new XMLHttpRequest"
      assert {:error, :no_content} = HtmlProcessor.html_to_text(html)
    end
  end
end
