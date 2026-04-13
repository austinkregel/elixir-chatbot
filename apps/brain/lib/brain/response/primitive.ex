defmodule Brain.Response.Primitive do
  @moduledoc """
  Intermediate representation between structured analysis and realized language.

  A Primitive represents a single communicative function in a response -- one
  "thing the system needs to express." Responses are composed as ordered
  sequences of primitives, each grounded in the analysis context.

  The lifecycle of a primitive:

    1. **Discourse Planner** sets `type` and `variant` based on analysis signals.
    2. **Content Specifier** fills `content` with grounded data from knowledge stores.
    3. **Surface Realizer** produces `rendered` text from `content` + phrasing pools.

  ## Types

  Primitives are organized by communicative function, not input type.
  Different user inputs produce different *combinations* of the same primitives.

    - `:acknowledgment` -- recognizing what the user said or did
    - `:framing` -- setting the pragmatic frame for what follows
    - `:hedging` -- epistemic calibration of confidence
    - `:content` -- the substantive body of the response
    - `:attunement` -- emotional/relational engagement
    - `:follow_up` -- driving the conversation forward
    - `:contradiction_response` -- handling conflicting information
    - `:transition` -- connecting parts of multi-part responses
  """

  @type primitive_type ::
          :acknowledgment
          | :framing
          | :hedging
          | :content
          | :attunement
          | :follow_up
          | :contradiction_response
          | :transition

  @type t :: %__MODULE__{
          type: primitive_type(),
          variant: atom() | nil,
          content: map(),
          rendered: String.t() | nil,
          source: atom() | nil,
          confidence: float()
        }

  defstruct [
    :type,
    :variant,
    :rendered,
    :source,
    content: %{},
    confidence: 1.0
  ]

  @doc "Creates a new primitive with the given type and variant."
  def new(type, variant \\ nil, content \\ %{}) do
    %__MODULE__{
      type: type,
      variant: variant,
      content: content
    }
  end

  @doc "Sets the rendered text on a primitive."
  def render(%__MODULE__{} = primitive, text) when is_binary(text) do
    %{primitive | rendered: text}
  end

  @doc "Updates the content map of a primitive."
  def put_content(%__MODULE__{} = primitive, key, value) do
    %{primitive | content: Map.put(primitive.content, key, value)}
  end

  @doc "Merges additional content into a primitive's content map."
  def merge_content(%__MODULE__{} = primitive, new_content) when is_map(new_content) do
    %{primitive | content: Map.merge(primitive.content, new_content)}
  end

  @doc "Returns true if the primitive has been rendered to text."
  def rendered?(%__MODULE__{rendered: nil}), do: false
  def rendered?(%__MODULE__{rendered: text}) when is_binary(text), do: true
  def rendered?(_), do: false

  @doc "Extracts rendered text from a list of primitives, joining with spaces."
  def join_rendered(primitives) when is_list(primitives) do
    primitives
    |> Enum.filter(&rendered?/1)
    |> Enum.map(& &1.rendered)
    |> Enum.reject(&(&1 == ""))
    |> Enum.join(" ")
    |> clean_punctuation()
    |> String.trim()
  end

  defp clean_punctuation(text) do
    text
    |> String.replace(~r/([.!?])\s*\1+/, "\\1")
  end
end
