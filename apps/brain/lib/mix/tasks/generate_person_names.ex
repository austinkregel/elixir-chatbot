defmodule Mix.Tasks.GeneratePersonNames do
  @moduledoc """
  Mix task to generate person name entity data from the US Social Security Administration
  baby names dataset.

  ## Usage

      mix generate_person_names [options]

  ## Options

    --min-count N        Minimum total occurrences across all years (default: 1000)
    --min-length N       Minimum name length (default: 3)
    --years N            Number of years to include, counting back from most recent (default: 40)
    --download           Force re-download of the SSA names.zip file
    --output PATH        Output path for the entries JSON (default: data/entities/person_entries_en.json)

  ## Data Source

  Downloads from: https://www.ssa.gov/oact/babynames/names.zip

  The SSA dataset contains baby name popularity data from 1880 to present, with names
  that have at least 5 occurrences per year.

  ## Filtering

  Names are filtered to exclude:
  - Names shorter than --min-length characters
  - Names that are common English words (months, days, pronouns, verbs, etc.)
  - Names with total count below --min-count across all included years

  This reduces false positives when used for always-on entity extraction.
  """

  use Mix.Task
  require Logger

  @shortdoc "Generate person name entities from SSA baby names data"

  @ssa_url "https://www.ssa.gov/oact/babynames/names.zip"
  @cache_dir "priv/data_cache"
  @cache_file "names.zip"
  @default_output "data/entities/person_entries_en.json"

  # Names that are common English words - will cause false positives in entity extraction
  # This list includes months, days, common verbs, pronouns, and other function words
  @stoplist MapSet.new([
              # Months
              "january",
              "february",
              "march",
              "april",
              "may",
              "june",
              "july",
              "august",
              "september",
              "october",
              "november",
              "december",
              # Days
              "monday",
              "tuesday",
              "wednesday",
              "thursday",
              "friday",
              "saturday",
              "sunday",
              # Common verbs/words that are also names
              "will",
              "bill",
              "bob",
              "sue",
              "sue",
              "pat",
              "mark",
              "jack",
              "rob",
              "skip",
              "chase",
              "joy",
              "hope",
              "faith",
              "grace",
              "grant",
              "drew",
              "wade",
              "wade",
              "brook",
              "brooks",
              "autumn",
              "summer",
              "winter",
              "spring",
              "dawn",
              "eve",
              "ivy",
              "iris",
              "rose",
              "lily",
              "violet",
              "daisy",
              "olive",
              "hazel",
              "amber",
              "coral",
              "jade",
              "pearl",
              "ruby",
              "crystal",
              # Directional/positional
              "north",
              "south",
              "east",
              "west",
              # Common function words that happen to be names
              "art",
              "don",
              "gene",
              "guy",
              "les",
              "ray",
              "rex",
              "roy",
              "van",
              # Very short common words
              "al",
              "bo",
              "ed",
              "jo",
              "jo",
              "ty",
              "mo",
              # Other problematic words
              "love",
              "star",
              "sky",
              "ocean",
              "river",
              "storm",
              "rain",
              "angel",
              "angel",
              "candy",
              "cherry",
              "ginger",
              "king",
              "queen",
              "prince",
              "duke",
              "earl",
              "baron",
              "major",
              "judge",
              "bishop",
              "canon",
              # Tech/common terms
              "page",
              "web",
              "chip",
              "cash",
              "rich",
              "sterling",
              # Actions that could be names
              "skip",
              "trip",
              "chase",
              "rush",
              "chance"
            ])

  # Additional names to keep even if they match stoplist patterns
  # These are very common names that should be recognized
  @allowlist MapSet.new([
               "james",
               "john",
               "robert",
               "michael",
               "david",
               "william",
               "richard",
               "joseph",
               "thomas",
               "christopher",
               "charles",
               "daniel",
               "matthew",
               "anthony",
               "mark",
               "donald",
               "steven",
               "paul",
               "andrew",
               "joshua",
               "mary",
               "patricia",
               "jennifer",
               "linda",
               "elizabeth",
               "barbara",
               "susan",
               "jessica",
               "sarah",
               "karen",
               "nancy",
               "lisa",
               "betty",
               "margaret",
               "sandra",
               "ashley",
               "dorothy",
               "kimberly",
               "emily",
               "emma",
               "olivia",
               "ava",
               "sophia",
               "isabella",
               "mia",
               "charlotte",
               "amelia",
               "harper",
               "evelyn",
               "abigail",
               "ella",
               "scarlett",
               "liam",
               "noah",
               "oliver",
               "elijah",
               "lucas",
               "mason",
               "logan",
               "alexander",
               "ethan",
               "jacob",
               "aiden",
               "jackson",
               "sebastian"
             ])

  def run(args) do
    # Parse arguments
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          min_count: :integer,
          min_length: :integer,
          years: :integer,
          download: :boolean,
          use_builtin: :boolean,
          output: :string
        ]
      )

    min_count = Keyword.get(opts, :min_count, 1000)
    min_length = Keyword.get(opts, :min_length, 3)
    years = Keyword.get(opts, :years, 40)
    force_download = Keyword.get(opts, :download, false)
    use_builtin = Keyword.get(opts, :use_builtin, false)
    output_path = Keyword.get(opts, :output, @default_output)

    Mix.shell().info("Generating person name entities...")
    Mix.shell().info("  Min count: #{min_count}")
    Mix.shell().info("  Min length: #{min_length}")
    Mix.shell().info("  Years: #{years}")
    Mix.shell().info("")

    if use_builtin do
      # Use built-in curated list of popular names
      generate_from_builtin(output_path, min_length)
    else
      # Ensure cache directory exists
      File.mkdir_p!(@cache_dir)

      # Download or use cached data
      zip_path = Path.join(@cache_dir, @cache_file)

      zip_path =
        if force_download or not File.exists?(zip_path) do
          download_ssa_data(zip_path)
        else
          Mix.shell().info("Using cached SSA data: #{zip_path}")
          zip_path
        end

      # Parse the zip file and aggregate names
      case parse_ssa_zip(zip_path, years) do
        {:ok, name_counts} ->
          # Filter and format names
          entries = build_entries(name_counts, min_count, min_length)

          # Write output
          write_entries(entries, output_path)

          display_success(entries, output_path)

        {:error, reason} ->
          Mix.shell().error("Failed to parse SSA data: #{inspect(reason)}")
          System.halt(1)
      end
    end
  end

  defp generate_from_builtin(output_path, min_length) do
    Mix.shell().info("Using built-in curated name list...")

    entries =
      builtin_popular_names()
      |> Enum.filter(fn name ->
        normalized = String.downcase(name)

        String.length(normalized) >= min_length and
          (MapSet.member?(@allowlist, normalized) or
             not MapSet.member?(@stoplist, normalized))
      end)
      |> Enum.map(fn name ->
        %{"value" => name, "synonyms" => [name]}
      end)
      |> Enum.sort_by(fn %{"value" => name} -> String.downcase(name) end)

    write_entries(entries, output_path)
    display_success(entries, output_path)
  end

  defp display_success(entries, output_path) do
    Mix.shell().info("")
    Mix.shell().info("Successfully generated #{length(entries)} person name entries")
    Mix.shell().info("Output written to: #{output_path}")
    Mix.shell().info("")
    Mix.shell().info("Next steps:")
    Mix.shell().info("  1. Run `mix train_models --gazetteer-only` to rebuild the gazetteer")
    Mix.shell().info("  2. Restart the application to use the new names")
  end

  defp download_ssa_data(dest_path) do
    Mix.shell().info("Downloading SSA baby names data...")
    Mix.shell().info("  URL: #{@ssa_url}")

    # Use Req for HTTP requests (more reliable than :httpc)
    case Req.get(@ssa_url, receive_timeout: 60_000, connect_options: [timeout: 30_000]) do
      {:ok, %{status: 200, body: body}} ->
        File.write!(dest_path, body)
        Mix.shell().info("  Downloaded #{byte_size(body)} bytes")
        dest_path

      {:ok, %{status: status}} ->
        Mix.shell().error("HTTP request failed with status: #{status}")
        Mix.shell().info("")
        Mix.shell().info("If download is blocked, you can manually download the file from:")
        Mix.shell().info("  #{@ssa_url}")
        Mix.shell().info("And place it at: #{dest_path}")
        Mix.shell().info("")
        Mix.shell().info("Alternatively, run with --use-builtin to use the built-in name list.")
        System.halt(1)

      {:error, reason} ->
        Mix.shell().error("Download failed: #{inspect(reason)}")
        Mix.shell().info("")
        Mix.shell().info("Run with --use-builtin to use the built-in name list instead.")
        System.halt(1)
    end
  end

  defp parse_ssa_zip(zip_path, years_to_include) do
    current_year = Date.utc_today().year
    # SSA data typically lags by about a year
    most_recent_year = current_year - 1
    earliest_year = most_recent_year - years_to_include + 1

    Mix.shell().info("Parsing SSA data for years #{earliest_year}-#{most_recent_year}...")

    case :zip.unzip(String.to_charlist(zip_path), [:memory]) do
      {:ok, files} ->
        # Filter to only yobXXXX.txt files in our year range
        name_counts =
          files
          |> Enum.filter(fn {filename, _content} ->
            name = to_string(filename)
            String.match?(name, ~r/yob\d{4}\.txt$/)
          end)
          |> Enum.filter(fn {filename, _content} ->
            case extract_year(to_string(filename)) do
              {:ok, year} -> year >= earliest_year and year <= most_recent_year
              :error -> false
            end
          end)
          |> Enum.reduce(%{}, fn {filename, content}, acc ->
            parse_year_file(to_string(filename), content, acc)
          end)

        Mix.shell().info("  Parsed #{map_size(name_counts)} unique names")
        {:ok, name_counts}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp extract_year(filename) do
    case Regex.run(~r/yob(\d{4})\.txt$/, filename) do
      [_, year_str] -> {:ok, String.to_integer(year_str)}
      _ -> :error
    end
  end

  defp parse_year_file(_filename, content, acc) do
    content
    |> to_string()
    |> String.split("\n", trim: true)
    |> Enum.reduce(acc, fn line, inner_acc ->
      case String.split(line, ",") do
        [name, _gender, count_str] ->
          count = String.to_integer(String.trim(count_str))
          normalized = String.downcase(String.trim(name))

          Map.update(inner_acc, normalized, %{name: name, count: count}, fn existing ->
            # Keep the most common capitalization (highest count for that spelling)
            if count > existing.count do
              %{existing | name: name, count: existing.count + count}
            else
              %{existing | count: existing.count + count}
            end
          end)

        _ ->
          inner_acc
      end
    end)
  end

  defp build_entries(name_counts, min_count, min_length) do
    name_counts
    |> Enum.filter(fn {normalized, %{count: count}} ->
      # Filter by count
      # Filter by length
      # Filter out stoplist words (unless in allowlist)
      count >= min_count and
        String.length(normalized) >= min_length and
        (MapSet.member?(@allowlist, normalized) or
           not MapSet.member?(@stoplist, normalized))
    end)
    |> Enum.map(fn {_normalized, %{name: name}} ->
      %{
        "value" => name,
        "synonyms" => [name]
      }
    end)
    |> Enum.sort_by(fn %{"value" => name} -> String.downcase(name) end)
  end

  defp write_entries(entries, output_path) do
    # Ensure directory exists
    output_path |> Path.dirname() |> File.mkdir_p!()

    json = Jason.encode!(entries, pretty: true)
    File.write!(output_path, json)
  end

  # Curated list of popular US names from SSA data (1985-2024)
  # This is a fallback when the SSA download is unavailable
  # Includes top ~500 names for each gender, filtered for common ones
  defp builtin_popular_names do
    [
      # Top male names (last 40 years)
      "Aaron", "Abraham", "Adam", "Adrian", "Aiden", "Alan", "Albert", "Alejandro",
      "Alex", "Alexander", "Alexis", "Alfred", "Allen", "Alvin", "Andre", "Andrew",
      "Andy", "Angel", "Anthony", "Antonio", "Arthur", "Asher", "Austin", "Axel",
      "Barry", "Benjamin", "Bernard", "Billy", "Blake", "Bobby", "Bradley", "Brady",
      "Brandon", "Brayden", "Brendan", "Brett", "Brian", "Brody", "Bruce", "Bryan",
      "Bryce", "Caleb", "Calvin", "Cameron", "Carl", "Carlos", "Carson", "Carter",
      "Casey", "Cesar", "Chad", "Charles", "Charlie", "Chase", "Chris", "Christian",
      "Christopher", "Clayton", "Clifford", "Clinton", "Cody", "Cole", "Colin",
      "Colton", "Connor", "Cooper", "Corey", "Craig", "Curtis", "Dakota", "Dale",
      "Dallas", "Dalton", "Damian", "Damon", "Daniel", "Danny", "Darius", "Darrell",
      "Darren", "David", "Dean", "Dennis", "Derek", "Derrick", "Devin", "Diego",
      "Dillon", "Dominic", "Donald", "Douglas", "Drew", "Dustin", "Dylan", "Earl",
      "Eddie", "Edgar", "Eduardo", "Edward", "Edwin", "Eli", "Elijah", "Elliot",
      "Elliott", "Emanuel", "Emilio", "Emmanuel", "Eric", "Erik", "Ernest", "Ethan",
      "Eugene", "Evan", "Everett", "Ezra", "Felix", "Fernando", "Finn", "Francisco",
      "Frank", "Franklin", "Frederick", "Gabriel", "Gage", "Garrett", "Gary", "Gavin",
      "Geoffrey", "George", "Gerald", "Gilbert", "Giovanni", "Glen", "Glenn", "Gordon",
      "Graham", "Grant", "Grayson", "Gregory", "Hank", "Harold", "Harrison", "Harry",
      "Harvey", "Hayden", "Hector", "Henry", "Howard", "Hudson", "Hugh", "Hunter",
      "Ian", "Isaac", "Isaiah", "Ivan", "Jack", "Jackson", "Jacob", "Jaden", "Jake",
      "James", "Jamie", "Jared", "Jason", "Javier", "Jay", "Jayden", "Jeffrey",
      "Jeremiah", "Jeremy", "Jerome", "Jerry", "Jesse", "Jesus", "Jimmy", "Joel",
      "Joey", "John", "Johnathan", "Johnny", "Jon", "Jonathan", "Jordan", "Jorge",
      "Jose", "Joseph", "Joshua", "Josiah", "Juan", "Julian", "Julio", "Justin",
      "Kai", "Keith", "Kelly", "Kenneth", "Kevin", "Kirk", "Kyle", "Landon", "Larry",
      "Lawrence", "Lee", "Leo", "Leon", "Leonard", "Leonardo", "Levi", "Lewis",
      "Liam", "Lincoln", "Logan", "Lorenzo", "Louis", "Lucas", "Luis", "Luke",
      "Malcolm", "Manuel", "Marco", "Marcus", "Mario", "Mark", "Marshall", "Martin",
      "Marvin", "Mason", "Mathew", "Matthew", "Maurice", "Max", "Maxwell", "Melvin",
      "Michael", "Miguel", "Miles", "Mitchell", "Mohamed", "Muhammad", "Nathan",
      "Nathaniel", "Neil", "Nelson", "Nicholas", "Nicolas", "Noah", "Noel", "Norman",
      "Oliver", "Omar", "Orlando", "Oscar", "Owen", "Parker", "Patrick", "Paul",
      "Pedro", "Perry", "Peter", "Philip", "Phillip", "Preston", "Rafael", "Ralph",
      "Ramon", "Randall", "Randy", "Raul", "Raymond", "Reginald", "Ricardo", "Richard",
      "Ricky", "Riley", "Robert", "Roberto", "Rodney", "Roger", "Roland", "Roman",
      "Ronald", "Ronnie", "Ross", "Roy", "Ruben", "Russell", "Ryan", "Salvador",
      "Sam", "Samuel", "Santiago", "Saul", "Scott", "Sean", "Sebastian", "Sergio",
      "Seth", "Shane", "Shawn", "Sidney", "Simon", "Spencer", "Stanley", "Stephen",
      "Steve", "Steven", "Stuart", "Taylor", "Terry", "Theodore", "Thomas", "Timothy",
      "Todd", "Tommy", "Tony", "Travis", "Trent", "Trevor", "Tristan", "Troy", "Tyler",
      "Victor", "Vincent", "Walter", "Warren", "Wayne", "Wesley", "William", "Willie",
      "Wyatt", "Xavier", "Zachary", "Zane",
      # Top female names (last 40 years)
      "Aaliyah", "Abigail", "Ada", "Addison", "Adelaide", "Adeline", "Adriana",
      "Aisha", "Alexa", "Alexandra", "Alexis", "Alice", "Alicia", "Alina", "Alison",
      "Allison", "Alyssa", "Amanda", "Amara", "Amelia", "Amy", "Ana", "Andrea",
      "Angela", "Angelina", "Anita", "Ann", "Anna", "Annabelle", "Anne", "Annie",
      "Ariana", "Arianna", "Ashley", "Aubrey", "Audrey", "Aurora", "Autumn", "Ava",
      "Avery", "Bailey", "Barbara", "Beatrice", "Bella", "Beth", "Bethany", "Betty",
      "Beverly", "Bianca", "Bonnie", "Brenda", "Brianna", "Bridget", "Brittany",
      "Brooke", "Brooklyn", "Caitlin", "Camila", "Candace", "Carly", "Carmen", "Carol",
      "Caroline", "Carolyn", "Carrie", "Cassandra", "Cassidy", "Catherine", "Cecilia",
      "Celeste", "Charlotte", "Chelsea", "Cheryl", "Chloe", "Christina", "Christine",
      "Cindy", "Claire", "Clara", "Claudia", "Colleen", "Constance", "Cora", "Courtney",
      "Crystal", "Cynthia", "Daisy", "Dakota", "Dana", "Danielle", "Darlene", "Dawn",
      "Deborah", "Debra", "Delilah", "Denise", "Destiny", "Diana", "Diane", "Dolores",
      "Donna", "Dora", "Dorothy", "Eden", "Edith", "Eileen", "Elaine", "Eleanor",
      "Elena", "Eliana", "Elizabeth", "Ella", "Ellen", "Ellie", "Eloise", "Elsie",
      "Emery", "Emilia", "Emily", "Emma", "Erica", "Erika", "Erin", "Esther", "Eva",
      "Evelyn", "Faith", "Fatima", "Felicia", "Fiona", "Florence", "Frances",
      "Francesca", "Gabriella", "Gabrielle", "Gail", "Gemma", "Genesis", "Georgia",
      "Gianna", "Gina", "Giselle", "Gloria", "Grace", "Gracie", "Gwendolyn", "Hailey",
      "Hannah", "Harmony", "Harper", "Hazel", "Heather", "Heidi", "Helen", "Holly",
      "Ida", "Imani", "Irene", "Iris", "Isabel", "Isabella", "Isabelle", "Isla", "Ivy",
      "Jackie", "Jacqueline", "Jade", "Jamie", "Jane", "Janet", "Janice", "Jasmine",
      "Jean", "Jeanette", "Jeanne", "Jenna", "Jennifer", "Jenny", "Jessica", "Jill",
      "Jillian", "Joan", "Joanna", "Jocelyn", "Jodi", "Jodie", "Jolene", "Jordan",
      "Josephine", "Joy", "Joyce", "Judith", "Judy", "Julia", "Juliana", "Julie",
      "Juliet", "June", "Kaitlyn", "Karen", "Karina", "Kate", "Katelyn", "Katherine",
      "Kathleen", "Kathryn", "Katie", "Kay", "Kayla", "Kaylee", "Kelly", "Kelsey",
      "Kendra", "Kennedy", "Kimberly", "Kinsley", "Kira", "Kristen", "Kristin",
      "Kristina", "Kylie", "Lacey", "Laila", "Lana", "Laura", "Lauren", "Layla",
      "Leah", "Leila", "Lena", "Leslie", "Lillian", "Lily", "Linda", "Lindsay",
      "Lindsey", "Lisa", "Lois", "Lola", "Loretta", "Lori", "Lorraine", "Louise",
      "Lucia", "Lucille", "Lucy", "Luna", "Lydia", "Lynda", "Lynn", "Mabel",
      "Mackenzie", "Macy", "Madeleine", "Madeline", "Madelyn", "Madison", "Maggie",
      "Makayla", "Mallory", "Mandy", "Mara", "Marcia", "Margaret", "Maria", "Mariah",
      "Marianne", "Marie", "Marilyn", "Marina", "Marissa", "Marjorie", "Marlene",
      "Martha", "Mary", "Maryann", "Matilda", "Maureen", "Maya", "Megan", "Melanie",
      "Melinda", "Melissa", "Melody", "Mercedes", "Meredith", "Mia", "Michaela",
      "Michelle", "Mikayla", "Mildred", "Millie", "Mindy", "Miranda", "Miriam",
      "Molly", "Monica", "Morgan", "Mya", "Nadia", "Nancy", "Naomi", "Natalia",
      "Natalie", "Natasha", "Nevaeh", "Nicole", "Nina", "Nora", "Norma", "Olive",
      "Olivia", "Paige", "Paisley", "Pamela", "Paris", "Patricia", "Paula", "Pauline",
      "Payton", "Peggy", "Penelope", "Penny", "Peyton", "Phoebe", "Piper", "Priscilla",
      "Quinn", "Rachel", "Raquel", "Reagan", "Rebecca", "Regina", "Renee", "Riley",
      "Rita", "Roberta", "Robin", "Rosa", "Rose", "Rosemary", "Rowan", "Ruby", "Ruth",
      "Rylee", "Sabrina", "Sadie", "Sally", "Samantha", "Sandra", "Sara", "Sarah",
      "Savannah", "Scarlett", "Selena", "Serena", "Shannon", "Sharon", "Sheila",
      "Shelby", "Sherry", "Shirley", "Sierra", "Sienna", "Silvia", "Skylar", "Sofia",
      "Sonia", "Sophia", "Sophie", "Stacey", "Stacy", "Stella", "Stephanie", "Sue",
      "Summer", "Susan", "Suzanne", "Sydney", "Sylvia", "Tabitha", "Tamara", "Tammy",
      "Tanya", "Tara", "Tatiana", "Taylor", "Teresa", "Tessa", "Theresa", "Tiffany",
      "Tina", "Tracy", "Trinity", "Valerie", "Vanessa", "Vera", "Veronica", "Vicky",
      "Victoria", "Violet", "Virginia", "Vivian", "Wendy", "Whitney", "Willow",
      "Willa", "Wilma", "Ximena", "Yasmin", "Yolanda", "Yvonne", "Zara", "Zoe", "Zoey"
    ]
  end
end
