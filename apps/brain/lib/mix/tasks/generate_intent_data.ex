defmodule Mix.Tasks.GenerateIntentData do
  @moduledoc """
  Mix task to generate intent training data from templates.

  ## Usage

      mix generate_intent_data <domain>

  ## Domains

    - calendar
    - reminder
    - todo
    - communication
    - entertainment
    - education
    - utilities
    - display
    - search
    - analysis
    - status

  """

  use Mix.Task
  require Logger
  import Bitwise

  @shortdoc "Generate intent training data from templates"

  @output_dir "data/training/intents"
  @negative_dir "data/training/intents/negative_examples"

  def run(args) do
    case args do
      [] ->
        Mix.shell().info("Usage: mix generate_intent_data <domain>")
        Mix.shell().info("Domains: calendar, reminder, todo, communication, entertainment, education, utilities, display, search, analysis, status")

      [domain | _] ->
        File.mkdir_p!(@output_dir)
        File.mkdir_p!(@negative_dir)
        generate_domain(domain)
    end
  end

  defp generate_domain("calendar"), do: generate_calendar_intents()
  defp generate_domain("reminder"), do: generate_reminder_intents()
  defp generate_domain("todo"), do: generate_todo_intents()
  defp generate_domain("communication"), do: generate_communication_intents()
  defp generate_domain("entertainment"), do: generate_entertainment_intents()
  defp generate_domain("education"), do: generate_education_intents()
  defp generate_domain("utilities"), do: generate_utilities_intents()
  defp generate_domain("display"), do: generate_display_intents()
  defp generate_domain("search"), do: generate_search_intents()
  defp generate_domain("analysis"), do: generate_analysis_intents()
  defp generate_domain("status"), do: generate_status_intents()
  defp generate_domain(domain), do: Mix.shell().error("Unknown domain: #{domain}")

  # ============================================================================
  # Calendar Domain
  # ============================================================================

  defp generate_calendar_intents do
    Mix.shell().info("Generating calendar intents...")

    intents = [
      {"calendar.event.create", calendar_event_create_templates()},
      {"calendar.event.delete", calendar_event_delete_templates()},
      {"calendar.event.update", calendar_event_update_templates()},
      {"calendar.event.reschedule", calendar_event_reschedule_templates()},
      {"calendar.event.check", calendar_event_check_templates()},
      {"calendar.query.today", calendar_query_today_templates()},
      {"calendar.query.tomorrow", calendar_query_tomorrow_templates()},
      {"calendar.query.week", calendar_query_week_templates()},
      {"calendar.query.date", calendar_query_date_templates()},
      {"calendar.query.next", calendar_query_next_templates()},
      {"calendar.query.between", calendar_query_between_templates()},
      {"calendar.query.free_time", calendar_query_free_time_templates()},
      {"calendar.event.add_attendee", calendar_event_add_attendee_templates()},
      {"calendar.event.remove_attendee", calendar_event_remove_attendee_templates()},
      {"calendar.event.set_location", calendar_event_set_location_templates()},
      {"calendar.event.set_reminder", calendar_event_set_reminder_templates()},
      {"calendar.event.set_recurring", calendar_event_set_recurring_templates()},
      {"calendar.event.add_notes", calendar_event_add_notes_templates()},
      {"calendar.meeting.schedule", calendar_meeting_schedule_templates()},
      {"calendar.meeting.find_time", calendar_meeting_find_time_templates()},
      {"calendar.meeting.cancel", calendar_meeting_cancel_templates()},
      {"calendar.meeting.join", calendar_meeting_join_templates()},
      {"calendar.meeting.check_conflicts", calendar_meeting_check_conflicts_templates()}
    ]

    Enum.each(intents, fn {intent_name, templates} ->
      generate_intent_file(intent_name, templates)
    end)

    # Generate negative examples for calendar domain
    generate_calendar_negative_examples()

    Mix.shell().info("Calendar intents generated!")
  end

  defp calendar_event_create_templates do
    [
      "create an event called {event_title}",
      "add {event_title} to my calendar",
      "schedule {event_title} for {date_time}",
      "create a calendar event for {event_title}",
      "add a new event {event_title} on {date_time}",
      "put {event_title} on my calendar",
      "I need to add {event_title} to my schedule",
      "can you create an event for {event_title}",
      "schedule a {event_title}",
      "add {event_title} at {date_time}",
      "create {event_title} on my calendar for {date_time}",
      "I want to schedule {event_title}",
      "book {event_title} for {date_time}",
      "set up an event called {event_title}",
      "make an appointment for {event_title}",
      "new calendar event {event_title}",
      "add an appointment {event_title} at {date_time}",
      "create a new appointment for {event_title}",
      "schedule an appointment called {event_title}",
      "put a {event_title} on my calendar for {date_time}",
      "I'd like to create an event for {event_title}",
      "please add {event_title} to my calendar",
      "can you schedule {event_title} for me",
      "block time for {event_title}",
      "reserve time for {event_title} on {date_time}"
    ]
  end

  defp calendar_event_delete_templates do
    [
      "delete the {event_title} event",
      "remove {event_title} from my calendar",
      "cancel the {event_title}",
      "delete my {event_title} on {date_time}",
      "remove the {event_title} appointment",
      "cancel my {event_title}",
      "get rid of the {event_title} event",
      "I need to delete {event_title}",
      "please remove {event_title} from my schedule",
      "cancel the event called {event_title}",
      "delete the appointment for {event_title}",
      "take {event_title} off my calendar",
      "remove my {date_time} {event_title}",
      "I want to cancel {event_title}",
      "delete {event_title} from my schedule",
      "can you remove the {event_title}",
      "unschedule {event_title}",
      "clear {event_title} from my calendar",
      "erase the {event_title} event",
      "I don't need the {event_title} anymore"
    ]
  end

  defp calendar_event_update_templates do
    [
      "update the {event_title} event",
      "change the {event_title} details",
      "modify my {event_title}",
      "edit the {event_title} event",
      "update {event_title} on my calendar",
      "I need to change {event_title}",
      "make changes to {event_title}",
      "alter the {event_title} event",
      "revise my {event_title}",
      "update the details for {event_title}",
      "change {event_title} information",
      "edit my {event_title} appointment",
      "modify the {event_title} details",
      "I want to update {event_title}",
      "can you edit {event_title}",
      "adjust the {event_title}",
      "change my {event_title} event",
      "update my calendar event {event_title}",
      "make an update to {event_title}",
      "I need to modify {event_title}"
    ]
  end

  defp calendar_event_reschedule_templates do
    [
      "reschedule {event_title} to {date_time}",
      "move {event_title} to {date_time}",
      "change {event_title} to {date_time}",
      "reschedule my {event_title}",
      "move the {event_title} event to {date_time}",
      "I need to reschedule {event_title}",
      "can you move {event_title} to {date_time}",
      "push {event_title} to {date_time}",
      "postpone {event_title} to {date_time}",
      "delay {event_title} until {date_time}",
      "shift {event_title} to {date_time}",
      "change the time of {event_title} to {date_time}",
      "move my {event_title} appointment to {date_time}",
      "reschedule the {event_title} for {date_time}",
      "I want to move {event_title} to {date_time}",
      "bump {event_title} to {date_time}",
      "relocate {event_title} to {date_time}",
      "switch {event_title} to {date_time}",
      "change when {event_title} is to {date_time}",
      "move the time of {event_title}"
    ]
  end

  defp calendar_event_check_templates do
    [
      "check if I have anything on {date_time}",
      "do I have any events on {date_time}",
      "what's scheduled for {date_time}",
      "is there anything on my calendar for {date_time}",
      "check my calendar for {date_time}",
      "am I free on {date_time}",
      "do I have plans on {date_time}",
      "what do I have on {date_time}",
      "anything scheduled for {date_time}",
      "check {date_time} on my calendar",
      "is {date_time} free",
      "what's happening on {date_time}",
      "do I have anything planned for {date_time}",
      "look up {date_time} on my calendar",
      "any events on {date_time}",
      "what's on {date_time}",
      "check for events on {date_time}",
      "is my {date_time} busy",
      "what events do I have on {date_time}",
      "show me {date_time} on my calendar"
    ]
  end

  defp calendar_query_today_templates do
    [
      "what's on my calendar today",
      "what do I have today",
      "show me today's schedule",
      "what are my events today",
      "today's calendar",
      "what's scheduled for today",
      "do I have anything today",
      "show today's events",
      "what's happening today",
      "my schedule for today",
      "today's appointments",
      "what meetings do I have today",
      "list today's events",
      "what's my day look like",
      "anything on the calendar today",
      "what do I have going on today",
      "today's agenda",
      "give me today's schedule",
      "what's today looking like",
      "show me what's on today"
    ]
  end

  defp calendar_query_tomorrow_templates do
    [
      "what's on my calendar tomorrow",
      "what do I have tomorrow",
      "show me tomorrow's schedule",
      "what are my events tomorrow",
      "tomorrow's calendar",
      "what's scheduled for tomorrow",
      "do I have anything tomorrow",
      "show tomorrow's events",
      "what's happening tomorrow",
      "my schedule for tomorrow",
      "tomorrow's appointments",
      "what meetings do I have tomorrow",
      "list tomorrow's events",
      "what's my day look like tomorrow",
      "anything on the calendar tomorrow",
      "what do I have going on tomorrow",
      "tomorrow's agenda",
      "give me tomorrow's schedule",
      "what's tomorrow looking like",
      "show me what's on tomorrow"
    ]
  end

  defp calendar_query_week_templates do
    [
      "what's on my calendar this week",
      "show me this week's schedule",
      "what do I have this week",
      "my schedule for the week",
      "this week's events",
      "what's happening this week",
      "weekly calendar",
      "show my week",
      "what meetings do I have this week",
      "list this week's events",
      "what's scheduled this week",
      "give me my weekly schedule",
      "this week's agenda",
      "what's the week looking like",
      "anything on my calendar this week",
      "my events for the week",
      "show the week ahead",
      "what do I have coming up this week",
      "weekly overview",
      "what's on for the week"
    ]
  end

  defp calendar_query_date_templates do
    [
      "what's on my calendar on {date_time}",
      "show me {date_time}",
      "what do I have on {date_time}",
      "events on {date_time}",
      "what's happening on {date_time}",
      "my schedule for {date_time}",
      "show my calendar for {date_time}",
      "what meetings do I have on {date_time}",
      "list events for {date_time}",
      "what's scheduled on {date_time}",
      "give me {date_time} schedule",
      "anything on {date_time}",
      "what do I have going on {date_time}",
      "{date_time} agenda",
      "my events on {date_time}",
      "check {date_time}",
      "what's on {date_time}",
      "show me the schedule for {date_time}",
      "appointments on {date_time}",
      "calendar for {date_time}"
    ]
  end

  defp calendar_query_next_templates do
    [
      "what's my next event",
      "when is my next meeting",
      "what's coming up next",
      "my next appointment",
      "what's next on my calendar",
      "next event",
      "when's my next event",
      "show me my next meeting",
      "what do I have next",
      "next on my schedule",
      "what's up next",
      "my upcoming event",
      "next scheduled event",
      "what's the next thing on my calendar",
      "when is my next appointment",
      "show next event",
      "next meeting",
      "what's my next scheduled item",
      "upcoming appointment",
      "what's immediately next on my calendar"
    ]
  end

  defp calendar_query_between_templates do
    [
      "what events do I have between {date_time} and {date_time}",
      "show my calendar from {date_time} to {date_time}",
      "events between {date_time} and {date_time}",
      "what's scheduled from {date_time} to {date_time}",
      "my schedule between {date_time} and {date_time}",
      "list events from {date_time} through {date_time}",
      "what do I have from {date_time} to {date_time}",
      "show events between {date_time} and {date_time}",
      "calendar from {date_time} to {date_time}",
      "what's on from {date_time} to {date_time}",
      "appointments between {date_time} and {date_time}",
      "meetings from {date_time} to {date_time}",
      "show schedule {date_time} to {date_time}",
      "events during {date_time} to {date_time}",
      "what's happening between {date_time} and {date_time}",
      "give me events from {date_time} to {date_time}",
      "check calendar {date_time} through {date_time}",
      "list what's on from {date_time} to {date_time}",
      "schedule between {date_time} and {date_time}",
      "what do I have during {date_time} to {date_time}"
    ]
  end

  defp calendar_query_free_time_templates do
    [
      "when am I free",
      "find free time on {date_time}",
      "when do I have availability",
      "show me my free slots",
      "what time am I available",
      "find available time",
      "when can I schedule something",
      "open slots on my calendar",
      "when am I available on {date_time}",
      "free time on {date_time}",
      "show available times",
      "when is my calendar open",
      "find gaps in my schedule",
      "available time slots",
      "when can I fit something in",
      "what times are free",
      "my availability on {date_time}",
      "open times on {date_time}",
      "when do I have free time",
      "check my availability"
    ]
  end

  defp calendar_event_add_attendee_templates do
    [
      "add {person} to {event_title}",
      "invite {person} to {event_title}",
      "include {person} in {event_title}",
      "add {person} as an attendee to {event_title}",
      "invite {person} to the {event_title} meeting",
      "add {person} to the {event_title} event",
      "include {person} in the {event_title}",
      "I want to add {person} to {event_title}",
      "can you invite {person} to {event_title}",
      "put {person} on the {event_title} invite",
      "add {person} to my {event_title}",
      "invite {person} to my {event_title}",
      "include {person} on {event_title}",
      "add attendee {person} to {event_title}",
      "add {person} to the invite for {event_title}",
      "get {person} on the {event_title}",
      "bring {person} into {event_title}",
      "add {person} to the calendar invite for {event_title}",
      "invite {person} to join {event_title}",
      "add another person {person} to {event_title}"
    ]
  end

  defp calendar_event_remove_attendee_templates do
    [
      "remove {person} from {event_title}",
      "uninvite {person} from {event_title}",
      "take {person} off {event_title}",
      "remove {person} as an attendee from {event_title}",
      "remove {person} from the {event_title} meeting",
      "take {person} off the {event_title} event",
      "remove {person} from the {event_title}",
      "I want to remove {person} from {event_title}",
      "can you uninvite {person} from {event_title}",
      "take {person} off the {event_title} invite",
      "remove {person} from my {event_title}",
      "uninvite {person} from my {event_title}",
      "remove {person} from the invite for {event_title}",
      "delete {person} from {event_title}",
      "remove attendee {person} from {event_title}",
      "drop {person} from {event_title}",
      "kick {person} from {event_title}",
      "exclude {person} from {event_title}",
      "cancel {person}'s invite to {event_title}",
      "remove {person} from the guest list for {event_title}"
    ]
  end

  defp calendar_event_set_location_templates do
    [
      "set location of {event_title} to {location}",
      "change the location of {event_title} to {location}",
      "add {location} as the location for {event_title}",
      "set {event_title} location to {location}",
      "move {event_title} to {location}",
      "update the location for {event_title} to {location}",
      "make {location} the location for {event_title}",
      "add location {location} to {event_title}",
      "set the venue for {event_title} to {location}",
      "change where {event_title} is to {location}",
      "set meeting location for {event_title} to {location}",
      "put {location} as the place for {event_title}",
      "update {event_title} location to {location}",
      "add {location} to the {event_title} event",
      "set the place for {event_title} to {location}",
      "change {event_title} location to {location}",
      "add venue {location} to {event_title}",
      "set where {event_title} happens to {location}",
      "update the venue for {event_title} to {location}",
      "{event_title} should be at {location}"
    ]
  end

  defp calendar_event_set_reminder_templates do
    [
      "add a reminder for {event_title}",
      "set a reminder for {event_title} {duration} before",
      "remind me about {event_title} {duration} before",
      "add a {duration} reminder to {event_title}",
      "set reminder for {event_title}",
      "I want a reminder for {event_title}",
      "notify me {duration} before {event_title}",
      "add notification for {event_title}",
      "alert me about {event_title} {duration} ahead",
      "set up a reminder for {event_title}",
      "remind me {duration} before {event_title}",
      "add a heads up for {event_title}",
      "set an alert for {event_title}",
      "give me a reminder for {event_title}",
      "remind me about the {event_title}",
      "add {duration} advance notice for {event_title}",
      "set a notification {duration} before {event_title}",
      "create a reminder for {event_title}",
      "remind me when {event_title} is coming up",
      "add advance warning for {event_title}"
    ]
  end

  defp calendar_event_set_recurring_templates do
    [
      "make {event_title} recurring",
      "set {event_title} to repeat {recurrence}",
      "make {event_title} repeat {recurrence}",
      "set {event_title} as a recurring event",
      "have {event_title} repeat {recurrence}",
      "make {event_title} a {recurrence} event",
      "set up {event_title} to recur {recurrence}",
      "repeat {event_title} {recurrence}",
      "make {event_title} happen {recurrence}",
      "schedule {event_title} {recurrence}",
      "set {event_title} to occur {recurrence}",
      "have {event_title} happen {recurrence}",
      "create a recurring {event_title}",
      "make {event_title} repeat every week",
      "set {event_title} to every {recurrence}",
      "{event_title} should repeat {recurrence}",
      "make {event_title} a regular thing {recurrence}",
      "set up recurring {event_title}",
      "have {event_title} occur {recurrence}",
      "{event_title} repeats {recurrence}"
    ]
  end

  defp calendar_event_add_notes_templates do
    [
      "add notes to {event_title}",
      "add a description to {event_title}",
      "put notes on {event_title}",
      "add details to {event_title}",
      "write notes for {event_title}",
      "add a note to {event_title}",
      "include notes in {event_title}",
      "add information to {event_title}",
      "put a description on {event_title}",
      "add agenda to {event_title}",
      "include details in {event_title}",
      "add meeting notes to {event_title}",
      "write a description for {event_title}",
      "attach notes to {event_title}",
      "add more details to {event_title}",
      "update notes for {event_title}",
      "put some notes on {event_title}",
      "add context to {event_title}",
      "include agenda in {event_title}",
      "add event notes to {event_title}"
    ]
  end

  defp calendar_meeting_schedule_templates do
    [
      "schedule a meeting with {person}",
      "set up a meeting with {person}",
      "book a meeting with {person} for {date_time}",
      "arrange a meeting with {person}",
      "create a meeting with {person}",
      "schedule a call with {person}",
      "set up a meeting for {date_time} with {person}",
      "I need to schedule a meeting with {person}",
      "can you schedule a meeting with {person}",
      "book time with {person}",
      "schedule a session with {person}",
      "set a meeting with {person} at {date_time}",
      "organize a meeting with {person}",
      "plan a meeting with {person}",
      "put a meeting with {person} on my calendar",
      "book a call with {person} for {date_time}",
      "schedule a {duration} meeting with {person}",
      "set up a {duration} meeting with {person}",
      "I want to meet with {person} on {date_time}",
      "arrange a {duration} call with {person}"
    ]
  end

  defp calendar_meeting_find_time_templates do
    [
      "find a time to meet with {person}",
      "when can I meet with {person}",
      "find availability with {person}",
      "check when {person} is free",
      "find a slot to meet with {person}",
      "when are {person} and I both free",
      "find mutual availability with {person}",
      "check {person}'s availability",
      "find a time that works for {person} and me",
      "when can {person} and I meet",
      "look for available time with {person}",
      "find a meeting time with {person}",
      "check for overlapping free time with {person}",
      "when is {person} available",
      "find a common time with {person}",
      "look up {person}'s calendar",
      "check mutual availability with {person}",
      "find open time with {person}",
      "when can we both meet",
      "search for meeting time with {person}"
    ]
  end

  defp calendar_meeting_cancel_templates do
    [
      "cancel my meeting with {person}",
      "cancel the meeting on {date_time}",
      "cancel the {event_title} meeting",
      "I need to cancel my meeting with {person}",
      "cancel my {date_time} meeting",
      "please cancel the meeting with {person}",
      "remove the meeting with {person}",
      "delete my meeting with {person}",
      "cancel the scheduled meeting with {person}",
      "I can't make the meeting with {person}",
      "cancel my call with {person}",
      "drop the meeting with {person}",
      "cancel the {event_title} on {date_time}",
      "abort the meeting with {person}",
      "cancel tomorrow's meeting with {person}",
      "I need to drop my meeting with {person}",
      "please remove the meeting with {person}",
      "get rid of my meeting with {person}",
      "cancel {person}'s meeting",
      "unschedule my meeting with {person}"
    ]
  end

  defp calendar_meeting_join_templates do
    [
      "join my meeting",
      "join the {event_title} meeting",
      "connect to my meeting",
      "join the call",
      "get me into my meeting",
      "join the video call",
      "connect me to the meeting",
      "enter the {event_title} meeting",
      "join my {date_time} meeting",
      "open my meeting",
      "launch the meeting",
      "start my meeting",
      "join the zoom call",
      "connect to the {event_title}",
      "get on the call",
      "join the conference call",
      "open the meeting link",
      "join my next meeting",
      "connect to the video meeting",
      "take me to my meeting"
    ]
  end

  defp calendar_meeting_check_conflicts_templates do
    [
      "check for conflicts on {date_time}",
      "do I have any conflicts on {date_time}",
      "are there any scheduling conflicts",
      "check for overlapping events on {date_time}",
      "do I have double booked anything",
      "check my calendar for conflicts",
      "any conflicts on {date_time}",
      "is there a conflict with {event_title}",
      "check if {date_time} has conflicts",
      "do I have overlapping meetings",
      "are there any double bookings",
      "check for scheduling issues",
      "any overlapping events on {date_time}",
      "do I have any calendar conflicts",
      "check if I'm double booked",
      "look for conflicts on {date_time}",
      "are there conflicts with {event_title}",
      "check my {date_time} for conflicts",
      "any scheduling conflicts on {date_time}",
      "verify no conflicts on {date_time}"
    ]
  end

  defp generate_calendar_negative_examples do
    negative_examples = [
      %{text: "remind me to buy groceries", correct_intent: "reminder.set"},
      %{text: "set a timer for 10 minutes", correct_intent: "timer.set"},
      %{text: "add milk to my shopping list", correct_intent: "shopping.add"},
      %{text: "what's the weather today", correct_intent: "weather"},
      %{text: "call John", correct_intent: "call.make"},
      %{text: "play some music", correct_intent: "music.play"},
      %{text: "turn on the lights", correct_intent: "smarthome.lights.on"},
      %{text: "send a message to Sarah", correct_intent: "message.send"},
      %{text: "set an alarm for 7am", correct_intent: "alarm.set"},
      %{text: "add task to my todo list", correct_intent: "todo.add"}
    ]

    write_negative_examples("calendar", negative_examples)
  end

  # ============================================================================
  # Helper Functions
  # ============================================================================

  defp generate_intent_file(intent_name, templates) do
    # Generate examples from templates
    examples = expand_templates(templates, intent_name)

    # Ensure minimum 20 examples
    examples = if length(examples) < 20 do
      # Duplicate and vary examples if needed
      examples ++ Enum.take(examples, 20 - length(examples))
    else
      examples
    end

    # Write to file
    file_name = String.replace(intent_name, ".", "_") <> ".json"
    path = Path.join(@output_dir, file_name)

    case Jason.encode(examples, pretty: true) do
      {:ok, json} ->
        File.write!(path, json)
        Mix.shell().info("  Generated #{path} (#{length(examples)} examples)")

      {:error, reason} ->
        Mix.shell().error("  Failed to generate #{path}: #{inspect(reason)}")
    end
  end

  defp expand_templates(templates, intent_name) do
    # Sample entity values for expansion
    event_titles = ["meeting", "dentist appointment", "lunch", "conference call", "interview", "workout", "standup", "presentation"]
    date_times = ["tomorrow", "next Monday", "Friday at 3pm", "tomorrow at 2pm", "next week", "in 2 hours"]
    persons = ["John", "Sarah", "the team", "my manager", "Mike", "Lisa"]
    locations = ["the office", "conference room A", "downtown", "Zoom", "the coffee shop"]
    durations = ["30 minutes", "1 hour", "15 minutes", "2 hours"]
    recurrences = ["weekly", "daily", "every Monday", "monthly", "every weekday"]

    templates
    |> Enum.flat_map(fn template ->
      # Generate 3-5 variations of each template
      1..3
      |> Enum.map(fn _ ->
        text = template
        |> String.replace("{event_title}", Enum.random(event_titles))
        |> String.replace("{date_time}", Enum.random(date_times))
        |> String.replace("{person}", Enum.random(persons))
        |> String.replace("{location}", Enum.random(locations))
        |> String.replace("{duration}", Enum.random(durations))
        |> String.replace("{recurrence}", Enum.random(recurrences))

        build_training_example(text, intent_name)
      end)
    end)
    |> Enum.uniq_by(fn ex -> ex["text"] end)
  end

  defp build_training_example(text, intent_name) do
    tokens = tokenize(text)
    pos_tags = tag_pos(tokens)
    entities = extract_entities(text, tokens)

    %{
      "text" => text,
      "tokens" => tokens,
      "pos_tags" => pos_tags,
      "entities" => entities,
      "id" => generate_uuid(),
      "intent" => intent_name
    }
  end

  defp tokenize(text) do
    text
    |> String.downcase()
    |> String.replace(~r/[^\w\s'-]/, " ")
    |> String.split(~r/\s+/, trim: true)
  end

  defp tag_pos(tokens) do
    # Simple rule-based POS tagging
    Enum.map(tokens, fn token ->
      cond do
        token in ["i", "you", "he", "she", "it", "we", "they", "me", "my", "your"] -> "PRON"
        token in ["the", "a", "an", "this", "that", "my", "your"] -> "DET"
        token in ["is", "are", "was", "were", "be", "been", "am", "have", "has", "had", "do", "does", "did", "can", "could", "will", "would", "should", "may", "might", "must"] -> "AUX"
        token in ["and", "or", "but", "if", "because", "when", "while", "although"] -> "CONJ"
        token in ["in", "on", "at", "to", "for", "with", "from", "by", "about", "of", "between", "through", "during", "before", "after"] -> "ADP"
        token in ["not", "never", "always", "also", "just", "only", "very", "really", "too"] -> "ADV"
        String.match?(token, ~r/^\d+$/) -> "NUM"
        String.match?(token, ~r/^[A-Z]/) -> "PROPN"
        token in ["schedule", "create", "add", "delete", "remove", "cancel", "update", "change", "set", "show", "check", "find", "join", "move", "make", "put", "get", "take", "give", "send", "call", "invite", "book", "arrange", "reschedule", "remind", "notify", "list"] -> "VERB"
        true -> "NOUN"
      end
    end)
  end

  defp extract_entities(text, tokens) do
    entities = []

    # Simple entity detection patterns
    event_titles = ["meeting", "dentist appointment", "lunch", "conference call", "interview", "workout", "standup", "presentation", "dentist", "conference", "appointment"]
    date_times = ["tomorrow", "next monday", "friday at 3pm", "tomorrow at 2pm", "next week", "in 2 hours", "today", "next friday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    persons = ["john", "sarah", "the team", "my manager", "mike", "lisa"]
    locations = ["the office", "conference room a", "downtown", "zoom", "the coffee shop", "office"]
    durations = ["30 minutes", "1 hour", "15 minutes", "2 hours", "an hour", "half an hour"]

    _text_lower = String.downcase(text)

    entities = entities ++ find_entity_spans(tokens, event_titles, "event_title")
    entities = entities ++ find_entity_spans(tokens, date_times, "date_time")
    entities = entities ++ find_entity_spans(tokens, persons, "person")
    entities = entities ++ find_entity_spans(tokens, locations, "location")
    entities = entities ++ find_entity_spans(tokens, durations, "duration")

    entities
  end

  defp find_entity_spans(tokens, values, entity_type) do
    tokens_lower = Enum.map(tokens, &String.downcase/1)

    values
    |> Enum.flat_map(fn value ->
      value_tokens = String.split(String.downcase(value), ~r/\s+/, trim: true)
      find_subsequence_indices(tokens_lower, value_tokens)
      |> Enum.map(fn start_idx ->
        end_idx = start_idx + length(value_tokens) - 1
        matched_text = tokens |> Enum.slice(start_idx..end_idx) |> Enum.join(" ")
        %{
          "text" => matched_text,
          "type" => entity_type,
          "start" => start_idx,
          "end" => end_idx
        }
      end)
    end)
  end

  defp find_subsequence_indices(tokens, subsequence) do
    if length(subsequence) > length(tokens) do
      []
    else
      0..(length(tokens) - length(subsequence))
      |> Enum.filter(fn i ->
        Enum.slice(tokens, i, length(subsequence)) == subsequence
      end)
    end
  end

  defp generate_uuid do
    # Generate a simple UUID v4
    <<a::32, b::16, c::16, d::16, e::48>> = :crypto.strong_rand_bytes(16)
    :io_lib.format("~8.16.0b-~4.16.0b-~4.16.0b-~4.16.0b-~12.16.0b", [a, b, band(c, 0x0FFF) ||| 0x4000, band(d, 0x3FFF) ||| 0x8000, e])
    |> IO.iodata_to_binary()
  end

  defp write_negative_examples(domain, examples) do
    file_name = "#{domain}_negative.json"
    path = Path.join(@negative_dir, file_name)

    case Jason.encode(examples, pretty: true) do
      {:ok, json} ->
        File.write!(path, json)
        Mix.shell().info("  Generated #{path} (#{length(examples)} examples)")

      {:error, reason} ->
        Mix.shell().error("  Failed to generate #{path}: #{inspect(reason)}")
    end
  end

  # ============================================================================
  # Reminder Domain
  # ============================================================================

  defp generate_reminder_intents do
    Mix.shell().info("Generating reminder intents...")

    intents = [
      {"reminder.set", reminder_set_templates()},
      {"reminder.set.location_based", reminder_location_templates()},
      {"reminder.set.recurring", reminder_recurring_templates()},
      {"reminder.cancel", reminder_cancel_templates()},
      {"reminder.list", reminder_list_templates()},
      {"reminder.check", reminder_check_templates()},
      {"reminder.snooze", reminder_snooze_templates()},
      {"reminder.mark_done", reminder_mark_done_templates()},
      {"reminder.update", reminder_update_templates()},
      {"alarm.set", alarm_set_templates()},
      {"alarm.set.recurring", alarm_recurring_templates()},
      {"alarm.cancel", alarm_cancel_templates()},
      {"alarm.cancel.all", alarm_cancel_all_templates()},
      {"alarm.snooze", alarm_snooze_templates()},
      {"alarm.stop", alarm_stop_templates()},
      {"alarm.list", alarm_list_templates()},
      {"alarm.check", alarm_check_templates()},
      {"timer.set", timer_set_templates()},
      {"timer.set.named", timer_set_named_templates()},
      {"timer.cancel", timer_cancel_templates()},
      {"timer.pause", timer_pause_templates()},
      {"timer.resume", timer_resume_templates()},
      {"timer.check", timer_check_templates()},
      {"timer.add_time", timer_add_time_templates()},
      {"timer.list", timer_list_templates()}
    ]

    Enum.each(intents, fn {intent_name, templates} ->
      generate_intent_file(intent_name, templates)
    end)

    generate_reminder_negative_examples()
    Mix.shell().info("Reminder intents generated!")
  end

  defp reminder_set_templates do
    [
      "remind me to {reminder_content}",
      "set a reminder to {reminder_content}",
      "remind me about {reminder_content}",
      "create a reminder for {reminder_content}",
      "remind me to {reminder_content} at {date_time}",
      "set a reminder for {reminder_content} at {date_time}",
      "remind me at {date_time} to {reminder_content}",
      "I need a reminder to {reminder_content}",
      "can you remind me to {reminder_content}",
      "don't let me forget to {reminder_content}",
      "please remind me to {reminder_content}",
      "add a reminder to {reminder_content}",
      "remind me {reminder_content}",
      "set reminder {reminder_content}",
      "create reminder to {reminder_content}",
      "I want to be reminded to {reminder_content}",
      "give me a reminder to {reminder_content}",
      "remind me in {duration} to {reminder_content}",
      "set a reminder in {duration} for {reminder_content}",
      "remind me {duration} from now to {reminder_content}"
    ]
  end

  defp reminder_location_templates do
    [
      "remind me when I get to {location} to {reminder_content}",
      "remind me at {location} to {reminder_content}",
      "when I arrive at {location} remind me to {reminder_content}",
      "remind me to {reminder_content} when I reach {location}",
      "set a location reminder for {location}",
      "remind me when I leave {location}",
      "when I get to {location} tell me to {reminder_content}",
      "location based reminder for {location}",
      "remind me at {location}",
      "when I'm at {location} remind me to {reminder_content}",
      "set reminder for when I arrive at {location}",
      "remind me to {reminder_content} at {location}",
      "when I get home remind me to {reminder_content}",
      "when I get to work remind me to {reminder_content}",
      "remind me to {reminder_content} when I get home",
      "at {location} remind me to {reminder_content}",
      "location reminder {location} {reminder_content}",
      "remind when arriving at {location}",
      "set geofence reminder for {location}",
      "remind me upon arrival at {location}"
    ]
  end

  defp reminder_recurring_templates do
    [
      "remind me every {recurrence} to {reminder_content}",
      "set a recurring reminder to {reminder_content}",
      "remind me {recurrence} to {reminder_content}",
      "create a {recurrence} reminder for {reminder_content}",
      "remind me to {reminder_content} every day",
      "remind me to {reminder_content} every week",
      "set a daily reminder to {reminder_content}",
      "set a weekly reminder for {reminder_content}",
      "remind me every morning to {reminder_content}",
      "remind me every night to {reminder_content}",
      "recurring reminder for {reminder_content}",
      "remind me {recurrence} about {reminder_content}",
      "set repeating reminder to {reminder_content}",
      "remind me regularly to {reminder_content}",
      "schedule recurring reminder {reminder_content}",
      "I need a {recurrence} reminder to {reminder_content}",
      "remind me every {recurrence}",
      "set up a repeating reminder for {reminder_content}",
      "daily reminder to {reminder_content}",
      "weekly reminder to {reminder_content}"
    ]
  end

  defp reminder_cancel_templates do
    [
      "cancel my reminder",
      "delete the reminder to {reminder_content}",
      "remove my reminder",
      "cancel the reminder for {reminder_content}",
      "delete my reminder about {reminder_content}",
      "remove the reminder to {reminder_content}",
      "cancel reminder",
      "delete reminder",
      "I don't need the reminder anymore",
      "cancel the {reminder_content} reminder",
      "remove reminder for {reminder_content}",
      "delete the {reminder_content} reminder",
      "clear my reminder",
      "get rid of the reminder",
      "cancel my {date_time} reminder",
      "remove the {date_time} reminder",
      "delete reminder to {reminder_content}",
      "cancel the reminder I set",
      "turn off the reminder",
      "dismiss the reminder"
    ]
  end

  defp reminder_list_templates do
    [
      "show my reminders",
      "list my reminders",
      "what reminders do I have",
      "show all reminders",
      "list all my reminders",
      "display my reminders",
      "what are my reminders",
      "view my reminders",
      "show me my reminders",
      "all reminders",
      "get my reminders",
      "show upcoming reminders",
      "list upcoming reminders",
      "my reminders",
      "reminders list",
      "show reminders",
      "what reminders have I set",
      "display all reminders",
      "see my reminders",
      "check my reminders"
    ]
  end

  defp reminder_check_templates do
    [
      "do I have any reminders",
      "are there any reminders",
      "check for reminders",
      "any reminders set",
      "do I have reminders for {date_time}",
      "check my reminders for {date_time}",
      "any reminders today",
      "do I have any reminders today",
      "are there reminders for tomorrow",
      "check if I have reminders",
      "any upcoming reminders",
      "reminders for {date_time}",
      "do I have a reminder to {reminder_content}",
      "is there a reminder for {reminder_content}",
      "check for {reminder_content} reminder",
      "any reminders coming up",
      "have I set any reminders",
      "do I have pending reminders",
      "check reminders",
      "any active reminders"
    ]
  end

  defp reminder_snooze_templates do
    [
      "snooze the reminder",
      "snooze for {duration}",
      "snooze reminder",
      "remind me later",
      "snooze this reminder",
      "snooze it",
      "remind me again in {duration}",
      "push the reminder back",
      "delay the reminder",
      "snooze for a bit",
      "snooze reminder for {duration}",
      "postpone the reminder",
      "remind me in {duration}",
      "snooze {duration}",
      "put off the reminder",
      "delay for {duration}",
      "snooze this",
      "remind me later about this",
      "push back the reminder {duration}",
      "postpone {duration}"
    ]
  end

  defp reminder_mark_done_templates do
    [
      "mark reminder as done",
      "complete the reminder",
      "I did {reminder_content}",
      "reminder done",
      "mark it as complete",
      "I finished {reminder_content}",
      "done with the reminder",
      "mark reminder complete",
      "finished the reminder",
      "reminder completed",
      "I've done {reminder_content}",
      "mark as done",
      "complete reminder",
      "I completed {reminder_content}",
      "tick off the reminder",
      "check off reminder",
      "did it",
      "that's done",
      "completed {reminder_content}",
      "mark {reminder_content} as done"
    ]
  end

  defp reminder_update_templates do
    [
      "update my reminder",
      "change the reminder to {date_time}",
      "update reminder time",
      "modify my reminder",
      "change my reminder",
      "edit the reminder",
      "update the reminder for {reminder_content}",
      "change reminder to {reminder_content}",
      "reschedule my reminder",
      "move my reminder to {date_time}",
      "update reminder to {date_time}",
      "modify the reminder time",
      "change when the reminder is",
      "edit my reminder",
      "update the {reminder_content} reminder",
      "alter my reminder",
      "change the {reminder_content} reminder",
      "push the reminder to {date_time}",
      "move reminder to {date_time}",
      "reschedule reminder to {date_time}"
    ]
  end

  defp alarm_set_templates do
    [
      "set an alarm for {date_time}",
      "wake me up at {date_time}",
      "set alarm {date_time}",
      "alarm at {date_time}",
      "create an alarm for {date_time}",
      "set an alarm at {date_time}",
      "I need an alarm for {date_time}",
      "set a {date_time} alarm",
      "alarm for {date_time}",
      "wake me at {date_time}",
      "set my alarm for {date_time}",
      "add an alarm at {date_time}",
      "set alarm for {date_time}",
      "create alarm {date_time}",
      "new alarm at {date_time}",
      "can you set an alarm for {date_time}",
      "I want an alarm at {date_time}",
      "please set an alarm for {date_time}",
      "set up an alarm for {date_time}",
      "make an alarm for {date_time}"
    ]
  end

  defp alarm_recurring_templates do
    [
      "set a recurring alarm for {date_time}",
      "set alarm for {date_time} every day",
      "daily alarm at {date_time}",
      "set alarm for every {recurrence} at {date_time}",
      "wake me up every day at {date_time}",
      "repeating alarm at {date_time}",
      "set a weekday alarm for {date_time}",
      "alarm every {recurrence} at {date_time}",
      "set recurring alarm at {date_time}",
      "create daily alarm for {date_time}",
      "set a {recurrence} alarm for {date_time}",
      "wake me every day at {date_time}",
      "set up recurring alarm at {date_time}",
      "alarm for {date_time} on weekdays",
      "repeating alarm every {recurrence}",
      "set alarm to repeat {recurrence}",
      "make the {date_time} alarm recurring",
      "daily wake up alarm at {date_time}",
      "alarm every morning at {date_time}",
      "set weekday alarm {date_time}"
    ]
  end

  defp alarm_cancel_templates do
    [
      "cancel my alarm",
      "delete the alarm",
      "remove the {date_time} alarm",
      "cancel the {date_time} alarm",
      "delete my alarm for {date_time}",
      "turn off my alarm",
      "remove my alarm",
      "cancel alarm",
      "delete alarm",
      "no alarm for {date_time}",
      "remove the alarm",
      "cancel my {date_time} alarm",
      "I don't need the alarm",
      "delete the {date_time} alarm",
      "disable my alarm",
      "turn off the {date_time} alarm",
      "cancel tomorrow's alarm",
      "delete my morning alarm",
      "remove my {date_time} alarm",
      "get rid of the alarm"
    ]
  end

  defp alarm_cancel_all_templates do
    [
      "cancel all alarms",
      "delete all my alarms",
      "remove all alarms",
      "turn off all alarms",
      "clear all alarms",
      "cancel all my alarms",
      "delete all alarms",
      "disable all alarms",
      "remove all my alarms",
      "get rid of all alarms",
      "clear my alarms",
      "turn off every alarm",
      "cancel every alarm",
      "delete every alarm",
      "no more alarms",
      "wipe all alarms",
      "remove every alarm",
      "shut off all alarms",
      "cancel each alarm",
      "delete the alarms"
    ]
  end

  defp alarm_snooze_templates do
    [
      "snooze the alarm",
      "snooze",
      "snooze alarm",
      "snooze for {duration}",
      "snooze it",
      "hit snooze",
      "5 more minutes",
      "snooze alarm for {duration}",
      "let me sleep a bit more",
      "snooze the alarm for {duration}",
      "just a few more minutes",
      "snooze please",
      "give me {duration} more",
      "snooze this alarm",
      "push snooze",
      "delay the alarm",
      "a few more minutes",
      "snooze for a bit",
      "let me snooze",
      "hit the snooze button"
    ]
  end

  defp alarm_stop_templates do
    [
      "stop the alarm",
      "turn off the alarm",
      "stop alarm",
      "dismiss alarm",
      "turn it off",
      "stop",
      "alarm off",
      "shut off the alarm",
      "dismiss the alarm",
      "silence the alarm",
      "quiet the alarm",
      "make it stop",
      "enough",
      "okay I'm up",
      "stop ringing",
      "turn off",
      "kill the alarm",
      "end alarm",
      "disable alarm",
      "shut up"
    ]
  end

  defp alarm_list_templates do
    [
      "show my alarms",
      "list my alarms",
      "what alarms do I have",
      "show all alarms",
      "list all my alarms",
      "display my alarms",
      "view my alarms",
      "my alarms",
      "alarms list",
      "show alarms",
      "what are my alarms",
      "get my alarms",
      "all alarms",
      "show me my alarms",
      "list alarms",
      "display alarms",
      "see my alarms",
      "check my alarms",
      "alarm schedule",
      "when are my alarms set for"
    ]
  end

  defp alarm_check_templates do
    [
      "do I have any alarms set",
      "is there an alarm set",
      "check my alarms",
      "do I have an alarm for {date_time}",
      "is my alarm set",
      "any alarms set",
      "check if I have alarms",
      "do I have alarms",
      "is there an alarm for tomorrow",
      "check for alarms",
      "am I set to wake up",
      "is my {date_time} alarm on",
      "are there any alarms",
      "do I have a morning alarm",
      "is an alarm set for {date_time}",
      "check alarm status",
      "any active alarms",
      "is my alarm on",
      "are alarms active",
      "alarm status"
    ]
  end

  defp timer_set_templates do
    [
      "set a timer for {duration}",
      "start a {duration} timer",
      "timer for {duration}",
      "set timer {duration}",
      "{duration} timer",
      "create a timer for {duration}",
      "start timer for {duration}",
      "I need a {duration} timer",
      "set a {duration} timer",
      "can you set a timer for {duration}",
      "start a timer for {duration}",
      "give me a {duration} timer",
      "time {duration}",
      "countdown {duration}",
      "set {duration} timer",
      "start {duration} timer",
      "timer {duration}",
      "please set a {duration} timer",
      "put on a timer for {duration}",
      "set up a {duration} timer"
    ]
  end

  defp timer_set_named_templates do
    [
      "set a {event_title} timer for {duration}",
      "start a timer called {event_title} for {duration}",
      "{duration} {event_title} timer",
      "set {event_title} timer for {duration}",
      "create a {event_title} timer for {duration}",
      "timer for {event_title} {duration}",
      "set a {duration} timer for {event_title}",
      "{event_title} timer {duration}",
      "start {event_title} timer {duration}",
      "set timer {event_title} {duration}",
      "new {event_title} timer for {duration}",
      "create timer called {event_title}",
      "add {event_title} timer for {duration}",
      "name the timer {event_title}",
      "{duration} timer called {event_title}",
      "set a timer named {event_title}",
      "start a {event_title} timer",
      "timer named {event_title} for {duration}",
      "create a {duration} {event_title} timer",
      "label timer as {event_title}"
    ]
  end

  defp timer_cancel_templates do
    [
      "cancel the timer",
      "stop the timer",
      "delete timer",
      "cancel timer",
      "end the timer",
      "remove the timer",
      "cancel my timer",
      "stop timer",
      "turn off the timer",
      "delete the timer",
      "cancel the {event_title} timer",
      "stop the {event_title} timer",
      "get rid of the timer",
      "kill the timer",
      "abort timer",
      "remove timer",
      "cancel {duration} timer",
      "end timer",
      "dismiss the timer",
      "terminate timer"
    ]
  end

  defp timer_pause_templates do
    [
      "pause the timer",
      "pause timer",
      "hold the timer",
      "pause my timer",
      "stop the timer temporarily",
      "freeze the timer",
      "pause the {event_title} timer",
      "hold timer",
      "put timer on hold",
      "suspend the timer",
      "pause counting",
      "stop counting temporarily",
      "hold on the timer",
      "freeze timer",
      "pause the countdown",
      "temporarily stop timer",
      "halt the timer",
      "pause time",
      "stop the clock",
      "timer pause"
    ]
  end

  defp timer_resume_templates do
    [
      "resume the timer",
      "resume timer",
      "continue the timer",
      "start the timer again",
      "unpause timer",
      "resume my timer",
      "continue timer",
      "restart the timer",
      "resume the {event_title} timer",
      "unpause the timer",
      "keep the timer going",
      "resume counting",
      "continue counting",
      "start timer again",
      "go on with the timer",
      "timer resume",
      "continue the countdown",
      "pick up the timer",
      "bring back the timer",
      "timer continue"
    ]
  end

  defp timer_check_templates do
    [
      "how much time is left",
      "check the timer",
      "how long on the timer",
      "timer status",
      "how much time left on the timer",
      "what's the timer at",
      "check timer",
      "how long left",
      "time remaining",
      "how much longer",
      "timer check",
      "time left on timer",
      "how much time remains",
      "what's left on the timer",
      "check the {event_title} timer",
      "how's the timer doing",
      "timer remaining",
      "how many minutes left",
      "time on the timer",
      "when does the timer end"
    ]
  end

  defp timer_add_time_templates do
    [
      "add {duration} to the timer",
      "add more time to the timer",
      "extend the timer by {duration}",
      "give me {duration} more",
      "add {duration} more",
      "increase timer by {duration}",
      "add time to timer",
      "put {duration} more on the timer",
      "extend timer {duration}",
      "add {duration} to timer",
      "{duration} more on the timer",
      "increase the timer",
      "extend the timer",
      "add another {duration}",
      "give me more time on the timer",
      "bump the timer up {duration}",
      "timer add {duration}",
      "more time on timer",
      "extend by {duration}",
      "add {duration} extra"
    ]
  end

  defp timer_list_templates do
    [
      "show my timers",
      "list timers",
      "what timers are running",
      "show all timers",
      "list my timers",
      "active timers",
      "display timers",
      "my timers",
      "running timers",
      "show timers",
      "what timers do I have",
      "all timers",
      "view my timers",
      "check all timers",
      "list all timers",
      "see my timers",
      "timer list",
      "current timers",
      "show running timers",
      "display all timers"
    ]
  end

  defp generate_reminder_negative_examples do
    negative_examples = [
      %{text: "schedule a meeting tomorrow", correct_intent: "calendar.meeting.schedule"},
      %{text: "what's on my calendar", correct_intent: "calendar.query.today"},
      %{text: "add milk to my shopping list", correct_intent: "shopping.add"},
      %{text: "call mom", correct_intent: "call.make"},
      %{text: "what's the weather", correct_intent: "weather"},
      %{text: "play music", correct_intent: "music.play"},
      %{text: "turn off the lights", correct_intent: "smarthome.lights.off"},
      %{text: "send a text to John", correct_intent: "message.send"},
      %{text: "add task to my todo", correct_intent: "todo.add"},
      %{text: "check my email", correct_intent: "email.check"}
    ]

    write_negative_examples("reminder", negative_examples)
  end

  # ============================================================================
  # Todo Domain
  # ============================================================================

  defp generate_todo_intents do
    Mix.shell().info("Generating todo intents...")

    intents = [
      {"todo.add", todo_add_templates()},
      {"todo.add.with_due_date", todo_add_due_date_templates()},
      {"todo.add.with_priority", todo_add_priority_templates()},
      {"todo.delete", todo_delete_templates()},
      {"todo.complete", todo_complete_templates()},
      {"todo.uncomplete", todo_uncomplete_templates()},
      {"todo.update", todo_update_templates()},
      {"todo.list", todo_list_templates()},
      {"todo.list.today", todo_list_today_templates()},
      {"todo.list.overdue", todo_list_overdue_templates()},
      {"todo.list.high_priority", todo_list_priority_templates()},
      {"todo.list.by_project", todo_list_project_templates()},
      {"todo.count", todo_count_templates()},
      {"todo.set_priority", todo_set_priority_templates()},
      {"todo.set_due_date", todo_set_due_date_templates()},
      {"todo.set_project", todo_set_project_templates()},
      {"shopping.add", shopping_add_templates()},
      {"shopping.remove", shopping_remove_templates()},
      {"shopping.list", shopping_list_templates()},
      {"shopping.clear", shopping_clear_templates()}
    ]

    Enum.each(intents, fn {intent_name, templates} ->
      generate_intent_file(intent_name, templates)
    end)

    generate_todo_negative_examples()
    Mix.shell().info("Todo intents generated!")
  end

  defp todo_add_templates do
    [
      "add {task_content} to my todo list",
      "add task {task_content}",
      "create a task {task_content}",
      "add {task_content} to my list",
      "new task {task_content}",
      "add {task_content}",
      "put {task_content} on my todo list",
      "I need to {task_content}",
      "add to do {task_content}",
      "task {task_content}",
      "create task {task_content}",
      "add {task_content} to todos",
      "remember to {task_content}",
      "add {task_content} as a task",
      "new todo {task_content}",
      "put {task_content} on my list",
      "add item {task_content}",
      "create a todo {task_content}",
      "add {task_content} to my tasks",
      "todo {task_content}"
    ]
  end

  defp todo_add_due_date_templates do
    [
      "add {task_content} due {date_time}",
      "add task {task_content} for {date_time}",
      "create task {task_content} due {date_time}",
      "add {task_content} by {date_time}",
      "{task_content} due {date_time}",
      "add {task_content} to finish by {date_time}",
      "new task {task_content} due {date_time}",
      "add {task_content} with deadline {date_time}",
      "task {task_content} for {date_time}",
      "add {task_content} needs to be done by {date_time}",
      "create {task_content} due on {date_time}",
      "add {task_content} deadline {date_time}",
      "{task_content} by {date_time}",
      "add {task_content} to complete by {date_time}",
      "new todo {task_content} due {date_time}",
      "add {task_content} for completion by {date_time}",
      "task due {date_time} {task_content}",
      "create task for {date_time} {task_content}",
      "add item {task_content} due {date_time}",
      "{task_content} should be done by {date_time}"
    ]
  end

  defp todo_add_priority_templates do
    [
      "add {priority} priority task {task_content}",
      "add {task_content} as {priority} priority",
      "create {priority} priority task {task_content}",
      "{priority} priority {task_content}",
      "add {task_content} with {priority} priority",
      "new {priority} task {task_content}",
      "add urgent task {task_content}",
      "add important task {task_content}",
      "create {priority} task {task_content}",
      "add {task_content} marked as {priority}",
      "{priority} {task_content}",
      "add {priority} item {task_content}",
      "new {priority} priority {task_content}",
      "task {task_content} priority {priority}",
      "add {task_content} {priority} priority",
      "create urgent {task_content}",
      "add {task_content} it's {priority}",
      "{task_content} is {priority}",
      "mark {task_content} as {priority}",
      "add {priority} todo {task_content}"
    ]
  end

  defp todo_delete_templates do
    [
      "delete task {task_content}",
      "remove {task_content} from my list",
      "delete {task_content}",
      "remove task {task_content}",
      "take {task_content} off my list",
      "delete {task_content} from my todos",
      "remove {task_content}",
      "get rid of task {task_content}",
      "delete the {task_content} task",
      "remove {task_content} from todos",
      "erase task {task_content}",
      "delete todo {task_content}",
      "remove the {task_content} item",
      "take off {task_content}",
      "delete item {task_content}",
      "remove {task_content} task",
      "clear task {task_content}",
      "drop {task_content} from my list",
      "delete my {task_content} task",
      "remove my {task_content}"
    ]
  end

  defp todo_complete_templates do
    [
      "mark {task_content} as done",
      "complete {task_content}",
      "check off {task_content}",
      "I finished {task_content}",
      "done with {task_content}",
      "{task_content} is done",
      "mark {task_content} complete",
      "I completed {task_content}",
      "finish {task_content}",
      "tick off {task_content}",
      "complete task {task_content}",
      "{task_content} completed",
      "mark {task_content} as complete",
      "I did {task_content}",
      "check {task_content} off",
      "done {task_content}",
      "{task_content} is finished",
      "mark done {task_content}",
      "completed {task_content}",
      "finish task {task_content}"
    ]
  end

  defp todo_uncomplete_templates do
    [
      "mark {task_content} as not done",
      "uncomplete {task_content}",
      "uncheck {task_content}",
      "{task_content} isn't done",
      "mark {task_content} incomplete",
      "undo complete on {task_content}",
      "reopen {task_content}",
      "unmark {task_content}",
      "{task_content} needs to be redone",
      "mark {task_content} as pending",
      "move {task_content} back to todo",
      "unfinish {task_content}",
      "{task_content} is not done",
      "revert {task_content} to pending",
      "uncheck task {task_content}",
      "mark {task_content} undone",
      "I didn't finish {task_content}",
      "put {task_content} back on the list",
      "undo {task_content} completion",
      "{task_content} still needs doing"
    ]
  end

  defp todo_update_templates do
    [
      "update task {task_content}",
      "edit {task_content}",
      "change {task_content}",
      "modify task {task_content}",
      "update {task_content}",
      "edit task {task_content}",
      "change task {task_content}",
      "modify {task_content}",
      "revise {task_content}",
      "update the {task_content} task",
      "edit my {task_content}",
      "change my {task_content} task",
      "alter {task_content}",
      "update {task_content} details",
      "edit the {task_content}",
      "modify the {task_content} task",
      "revise task {task_content}",
      "change the {task_content}",
      "update my {task_content}",
      "make changes to {task_content}"
    ]
  end

  defp todo_list_templates do
    [
      "show my tasks",
      "list my todos",
      "what's on my todo list",
      "show my todo list",
      "list all tasks",
      "my tasks",
      "show todos",
      "what tasks do I have",
      "display my tasks",
      "view my todo list",
      "all my tasks",
      "list todos",
      "show all tasks",
      "my todo list",
      "get my tasks",
      "see my todos",
      "what are my tasks",
      "display todos",
      "show me my tasks",
      "list my tasks"
    ]
  end

  defp todo_list_today_templates do
    [
      "tasks due today",
      "what's due today",
      "today's tasks",
      "show tasks for today",
      "what do I need to do today",
      "today's todos",
      "tasks for today",
      "what's on my list for today",
      "show today's tasks",
      "due today",
      "things to do today",
      "what needs to be done today",
      "today's to do list",
      "list today's tasks",
      "todos for today",
      "show what's due today",
      "my tasks for today",
      "what tasks are due today",
      "today's list",
      "tasks I need to do today"
    ]
  end

  defp todo_list_overdue_templates do
    [
      "show overdue tasks",
      "what's overdue",
      "overdue items",
      "list overdue tasks",
      "overdue todos",
      "what tasks are overdue",
      "show me overdue items",
      "overdue list",
      "tasks past due",
      "late tasks",
      "what am I behind on",
      "overdue things",
      "show late tasks",
      "past due tasks",
      "what's past the deadline",
      "list late items",
      "overdue assignments",
      "missed deadline tasks",
      "show tasks past due",
      "what haven't I finished"
    ]
  end

  defp todo_list_priority_templates do
    [
      "show high priority tasks",
      "important tasks",
      "urgent todos",
      "list priority tasks",
      "what's urgent",
      "high priority items",
      "show urgent tasks",
      "critical tasks",
      "important items",
      "priority tasks",
      "list high priority",
      "what's important",
      "show important tasks",
      "urgent items",
      "top priority tasks",
      "most important tasks",
      "show critical tasks",
      "high priority todos",
      "what needs immediate attention",
      "priority items"
    ]
  end

  defp todo_list_project_templates do
    [
      "show tasks in {project}",
      "tasks for {project}",
      "{project} tasks",
      "list {project} todos",
      "what's in {project}",
      "show {project} items",
      "{project} list",
      "tasks in {project} project",
      "show me {project} tasks",
      "list items in {project}",
      "{project} todos",
      "get {project} tasks",
      "display {project} tasks",
      "what tasks are in {project}",
      "show {project} list",
      "items in {project}",
      "{project} items",
      "list {project} tasks",
      "tasks from {project}",
      "view {project} tasks"
    ]
  end

  defp todo_count_templates do
    [
      "how many tasks do I have",
      "count my tasks",
      "how many todos",
      "number of tasks",
      "task count",
      "how many items on my list",
      "count todos",
      "how many things to do",
      "total tasks",
      "how many tasks left",
      "count my todos",
      "how many pending tasks",
      "number of todos",
      "tasks remaining",
      "how many items",
      "count all tasks",
      "todo count",
      "how many tasks are there",
      "total number of tasks",
      "how much is on my list"
    ]
  end

  defp todo_set_priority_templates do
    [
      "set {task_content} to {priority} priority",
      "make {task_content} {priority}",
      "change {task_content} priority to {priority}",
      "mark {task_content} as {priority}",
      "set priority of {task_content} to {priority}",
      "{task_content} is {priority} priority",
      "update {task_content} to {priority}",
      "prioritize {task_content} as {priority}",
      "change priority of {task_content}",
      "make {task_content} {priority} priority",
      "set {task_content} as {priority}",
      "{priority} priority for {task_content}",
      "change {task_content} to {priority}",
      "update priority of {task_content}",
      "assign {priority} priority to {task_content}",
      "move {task_content} to {priority}",
      "flag {task_content} as {priority}",
      "{task_content} priority {priority}",
      "set {task_content} priority {priority}",
      "mark {task_content} {priority}"
    ]
  end

  defp todo_set_due_date_templates do
    [
      "set {task_content} due date to {date_time}",
      "change due date of {task_content} to {date_time}",
      "make {task_content} due {date_time}",
      "set deadline for {task_content} to {date_time}",
      "{task_content} should be due {date_time}",
      "update {task_content} due date to {date_time}",
      "move {task_content} deadline to {date_time}",
      "change {task_content} to be due {date_time}",
      "set {task_content} for {date_time}",
      "due date for {task_content} is {date_time}",
      "reschedule {task_content} to {date_time}",
      "make {task_content} due on {date_time}",
      "{task_content} due by {date_time}",
      "change deadline of {task_content}",
      "move {task_content} to {date_time}",
      "update due date of {task_content}",
      "set {task_content} deadline {date_time}",
      "push {task_content} to {date_time}",
      "delay {task_content} to {date_time}",
      "{task_content} by {date_time}"
    ]
  end

  defp todo_set_project_templates do
    [
      "move {task_content} to {project}",
      "put {task_content} in {project}",
      "add {task_content} to {project} project",
      "assign {task_content} to {project}",
      "set {task_content} project to {project}",
      "change {task_content} project to {project}",
      "move {task_content} to {project} list",
      "categorize {task_content} under {project}",
      "file {task_content} under {project}",
      "{task_content} belongs to {project}",
      "put {task_content} under {project}",
      "organize {task_content} in {project}",
      "place {task_content} in {project}",
      "set project for {task_content} to {project}",
      "assign {project} project to {task_content}",
      "move task {task_content} to {project}",
      "{task_content} to {project}",
      "change {task_content} to {project}",
      "tag {task_content} with {project}",
      "link {task_content} to {project}"
    ]
  end

  defp shopping_add_templates do
    [
      "add {task_content} to my shopping list",
      "put {task_content} on my shopping list",
      "add {task_content} to shopping",
      "I need to buy {task_content}",
      "add {task_content} to groceries",
      "shopping list add {task_content}",
      "put {task_content} on the list",
      "buy {task_content}",
      "need {task_content}",
      "add {task_content} to buy",
      "remember to buy {task_content}",
      "get {task_content}",
      "pick up {task_content}",
      "add {task_content} to the shopping list",
      "I need {task_content}",
      "shopping add {task_content}",
      "put {task_content} on shopping",
      "add to shopping {task_content}",
      "groceries add {task_content}",
      "{task_content} on shopping list"
    ]
  end

  defp shopping_remove_templates do
    [
      "remove {task_content} from my shopping list",
      "take {task_content} off shopping list",
      "delete {task_content} from shopping",
      "I don't need {task_content}",
      "remove {task_content} from groceries",
      "take off {task_content}",
      "remove {task_content}",
      "delete {task_content} from the list",
      "don't need {task_content} anymore",
      "cross off {task_content}",
      "scratch {task_content}",
      "remove {task_content} from list",
      "take {task_content} off",
      "shopping remove {task_content}",
      "got {task_content} already",
      "already have {task_content}",
      "don't buy {task_content}",
      "skip {task_content}",
      "cancel {task_content}",
      "delete shopping item {task_content}"
    ]
  end

  defp shopping_list_templates do
    [
      "show my shopping list",
      "what's on my shopping list",
      "shopping list",
      "show shopping",
      "what do I need to buy",
      "list shopping items",
      "my shopping list",
      "display shopping list",
      "view shopping list",
      "groceries list",
      "show groceries",
      "what's on the list",
      "shopping items",
      "get shopping list",
      "see shopping list",
      "what to buy",
      "show me the shopping list",
      "list what I need to buy",
      "display groceries",
      "view shopping"
    ]
  end

  defp shopping_clear_templates do
    [
      "clear my shopping list",
      "empty the shopping list",
      "delete all shopping items",
      "remove everything from shopping list",
      "clear shopping",
      "wipe the shopping list",
      "start fresh shopping list",
      "delete shopping list",
      "erase shopping list",
      "remove all from shopping",
      "clear all shopping items",
      "empty shopping",
      "reset shopping list",
      "clear groceries",
      "delete all from shopping",
      "remove all items from shopping",
      "start over on shopping list",
      "blank shopping list",
      "clear the list",
      "wipe shopping"
    ]
  end

  defp generate_todo_negative_examples do
    negative_examples = [
      %{text: "remind me to call mom", correct_intent: "reminder.set"},
      %{text: "set an alarm for 7am", correct_intent: "alarm.set"},
      %{text: "schedule a meeting tomorrow", correct_intent: "calendar.meeting.schedule"},
      %{text: "what's the weather", correct_intent: "weather"},
      %{text: "play some music", correct_intent: "music.play"},
      %{text: "send a message to John", correct_intent: "message.send"},
      %{text: "turn on the lights", correct_intent: "smarthome.lights.on"},
      %{text: "call mom", correct_intent: "call.make"},
      %{text: "check my calendar", correct_intent: "calendar.query.today"},
      %{text: "set a timer", correct_intent: "timer.set"}
    ]

    write_negative_examples("todo", negative_examples)
  end

  # ============================================================================
  # Communication Domain
  # ============================================================================

  defp generate_communication_intents do
    Mix.shell().info("Generating communication intents...")

    intents = [
      {"call.make", call_make_templates()},
      {"call.end", call_end_templates()},
      {"call.answer", call_answer_templates()},
      {"call.decline", call_decline_templates()},
      {"call.mute", call_mute_templates()},
      {"call.unmute", call_unmute_templates()},
      {"call.speaker", call_speaker_templates()},
      {"call.check_missed", call_check_missed_templates()},
      {"message.send", message_send_templates()},
      {"message.read", message_read_templates()},
      {"message.reply", message_reply_templates()},
      {"message.check", message_check_templates()},
      {"email.compose", email_compose_templates()},
      {"email.send", email_send_templates()},
      {"email.read", email_read_templates()},
      {"email.reply", email_reply_templates()},
      {"email.check", email_check_templates()},
      {"contact.find", contact_find_templates()},
      {"contact.add", contact_add_templates()}
    ]

    Enum.each(intents, fn {intent_name, templates} ->
      generate_intent_file(intent_name, templates)
    end)

    generate_communication_negative_examples()
    Mix.shell().info("Communication intents generated!")
  end

  defp call_make_templates do
    [
      "call {person}",
      "phone {person}",
      "dial {person}",
      "call {person} on the phone",
      "ring {person}",
      "give {person} a call",
      "make a call to {person}",
      "call {person} please",
      "I want to call {person}",
      "can you call {person}",
      "place a call to {person}",
      "connect me to {person}",
      "get {person} on the phone",
      "phone call to {person}",
      "call {person}'s number",
      "dial {person}'s phone",
      "I need to call {person}",
      "ring up {person}",
      "make a phone call to {person}",
      "call {person} now"
    ]
  end

  defp call_end_templates do
    [
      "hang up",
      "end the call",
      "end call",
      "hang up the phone",
      "disconnect",
      "finish the call",
      "terminate call",
      "I'm done with the call",
      "end this call",
      "stop the call",
      "disconnect the call",
      "hang up now",
      "done with the call",
      "close the call",
      "bye hang up",
      "end phone call",
      "get off the phone",
      "stop calling",
      "finish call",
      "cut the call"
    ]
  end

  defp call_answer_templates do
    [
      "answer the call",
      "pick up",
      "answer",
      "accept the call",
      "pick up the phone",
      "take the call",
      "answer the phone",
      "accept call",
      "get the call",
      "answer it",
      "pick it up",
      "take it",
      "accept incoming call",
      "I'll take it",
      "put them through",
      "connect the call",
      "answer incoming",
      "get the phone",
      "receive the call",
      "accept"
    ]
  end

  defp call_decline_templates do
    [
      "decline the call",
      "reject the call",
      "don't answer",
      "ignore the call",
      "decline",
      "reject",
      "send to voicemail",
      "I can't take it",
      "don't pick up",
      "decline call",
      "reject call",
      "no thanks",
      "not now",
      "let it go to voicemail",
      "dismiss the call",
      "skip the call",
      "refuse the call",
      "ignore it",
      "don't answer it",
      "pass on this call"
    ]
  end

  defp call_mute_templates do
    [
      "mute",
      "mute the call",
      "mute myself",
      "turn off my mic",
      "mute my microphone",
      "go on mute",
      "silence my mic",
      "mute me",
      "put myself on mute",
      "turn mic off",
      "mute audio",
      "disable my microphone",
      "mute my audio",
      "mic off",
      "turn off microphone",
      "silence me",
      "I want to mute",
      "mute please",
      "can you mute me",
      "switch to mute"
    ]
  end

  defp call_unmute_templates do
    [
      "unmute",
      "unmute myself",
      "turn on my mic",
      "unmute my microphone",
      "take off mute",
      "unmute me",
      "turn mic on",
      "unmute audio",
      "enable my microphone",
      "mic on",
      "turn on microphone",
      "get off mute",
      "I want to unmute",
      "unmute please",
      "can you unmute me",
      "switch off mute",
      "activate mic",
      "turn the mic back on",
      "remove mute",
      "unmute now"
    ]
  end

  defp call_speaker_templates do
    [
      "put it on speaker",
      "speaker phone",
      "turn on speaker",
      "switch to speaker",
      "speakerphone",
      "put on speaker",
      "enable speaker",
      "speaker mode",
      "hands free",
      "turn on speakerphone",
      "activate speaker",
      "go to speaker",
      "speaker please",
      "use the speaker",
      "speaker on",
      "I want speaker",
      "put this on speaker",
      "loud speaker",
      "switch to speakerphone",
      "turn speaker on"
    ]
  end

  defp call_check_missed_templates do
    [
      "check missed calls",
      "any missed calls",
      "show missed calls",
      "did I miss any calls",
      "who called me",
      "missed calls",
      "list missed calls",
      "do I have missed calls",
      "check for missed calls",
      "show me missed calls",
      "any calls I missed",
      "who tried to call",
      "check if I missed calls",
      "display missed calls",
      "view missed calls",
      "recent missed calls",
      "did anyone call",
      "calls I missed",
      "unanswered calls",
      "check call history"
    ]
  end

  defp message_send_templates do
    [
      "send a message to {person}",
      "text {person}",
      "message {person}",
      "send {person} a text",
      "text message to {person}",
      "send text to {person}",
      "message to {person}",
      "I want to text {person}",
      "send a text to {person}",
      "SMS {person}",
      "shoot a text to {person}",
      "write a message to {person}",
      "send {person} a message",
      "compose a text to {person}",
      "text {person} saying",
      "message {person} that",
      "send SMS to {person}",
      "drop a text to {person}",
      "ping {person}",
      "send {person} text"
    ]
  end

  defp message_read_templates do
    [
      "read my messages",
      "show my texts",
      "check messages",
      "read texts",
      "show messages",
      "what messages do I have",
      "display my messages",
      "read my texts",
      "view messages",
      "open messages",
      "show text messages",
      "list my messages",
      "get my messages",
      "read text messages",
      "what texts do I have",
      "check my texts",
      "see my messages",
      "display texts",
      "messages",
      "show my messages"
    ]
  end

  defp message_reply_templates do
    [
      "reply to {person}",
      "respond to {person}",
      "text {person} back",
      "reply to the message",
      "send a reply",
      "reply back",
      "respond to the text",
      "answer {person}'s message",
      "write back to {person}",
      "reply to {person}'s text",
      "text back",
      "send response to {person}",
      "get back to {person}",
      "answer the message",
      "reply to last message",
      "respond to last text",
      "reply {person}",
      "message {person} back",
      "write {person} back",
      "reply to that"
    ]
  end

  defp message_check_templates do
    [
      "do I have any messages",
      "any new messages",
      "check for messages",
      "any texts",
      "do I have new texts",
      "new messages",
      "any unread messages",
      "check for new texts",
      "did anyone text me",
      "any messages for me",
      "unread texts",
      "check inbox",
      "any incoming messages",
      "did I get any messages",
      "new texts",
      "have I got messages",
      "message notifications",
      "any new SMS",
      "check text messages",
      "do I have texts"
    ]
  end

  defp email_compose_templates do
    [
      "compose an email to {person}",
      "write an email to {person}",
      "new email to {person}",
      "start an email to {person}",
      "email {person}",
      "draft an email to {person}",
      "create an email for {person}",
      "compose email to {person}",
      "write email to {person}",
      "I want to email {person}",
      "begin email to {person}",
      "type an email to {person}",
      "compose new email",
      "write a new email",
      "start email to {person}",
      "draft email to {person}",
      "open new email to {person}",
      "new message to {person}",
      "create email to {person}",
      "begin composing email"
    ]
  end

  defp email_send_templates do
    [
      "send the email",
      "send email",
      "send this email",
      "send it",
      "deliver the email",
      "send email now",
      "dispatch the email",
      "send my email",
      "mail it",
      "send the message",
      "send email to {person}",
      "deliver email",
      "post the email",
      "transmit the email",
      "email send",
      "send out the email",
      "fire off the email",
      "submit the email",
      "complete sending",
      "go ahead and send"
    ]
  end

  defp email_read_templates do
    [
      "read my emails",
      "show my emails",
      "check email",
      "read emails",
      "show emails",
      "what emails do I have",
      "display my emails",
      "view emails",
      "open emails",
      "list my emails",
      "get my emails",
      "read my inbox",
      "check my inbox",
      "show inbox",
      "what's in my inbox",
      "see my emails",
      "display inbox",
      "emails",
      "show me my emails",
      "open inbox"
    ]
  end

  defp email_reply_templates do
    [
      "reply to this email",
      "respond to email",
      "reply to {person}'s email",
      "email {person} back",
      "reply to the email",
      "send a reply",
      "respond to {person}",
      "answer the email",
      "write back",
      "reply email",
      "respond to last email",
      "get back to {person}",
      "reply back",
      "answer {person}'s email",
      "email back",
      "send response",
      "reply to sender",
      "respond to this",
      "write a reply",
      "reply now"
    ]
  end

  defp email_check_templates do
    [
      "do I have any emails",
      "any new emails",
      "check for emails",
      "new emails",
      "any unread emails",
      "check for new emails",
      "did I get any emails",
      "email notifications",
      "any incoming emails",
      "unread emails",
      "check email inbox",
      "any messages in email",
      "new mail",
      "have I got email",
      "any new mail",
      "email check",
      "do I have mail",
      "any emails waiting",
      "check for mail",
      "email updates"
    ]
  end

  defp contact_find_templates do
    [
      "find {person}",
      "look up {person}",
      "search for {person}",
      "find contact {person}",
      "look up {person}'s number",
      "get {person}'s info",
      "find {person}'s contact",
      "search contacts for {person}",
      "look for {person}",
      "find {person} in contacts",
      "get {person}'s contact info",
      "locate {person}",
      "where is {person}'s info",
      "pull up {person}",
      "show me {person}'s contact",
      "find number for {person}",
      "search {person}",
      "lookup {person}",
      "get contact for {person}",
      "find info for {person}"
    ]
  end

  defp contact_add_templates do
    [
      "add new contact",
      "create a contact for {person}",
      "add {person} to contacts",
      "new contact {person}",
      "save {person} as a contact",
      "add contact {person}",
      "create contact {person}",
      "save {person} to contacts",
      "add {person} to my contacts",
      "new contact for {person}",
      "save contact {person}",
      "add {person}",
      "store {person}'s number",
      "create new contact {person}",
      "add {person} number",
      "save this number as {person}",
      "add to contacts {person}",
      "save as contact {person}",
      "new contact named {person}",
      "add {person}'s contact"
    ]
  end

  defp generate_communication_negative_examples do
    negative_examples = [
      %{text: "remind me to call mom", correct_intent: "reminder.set"},
      %{text: "schedule a meeting", correct_intent: "calendar.meeting.schedule"},
      %{text: "add milk to my list", correct_intent: "shopping.add"},
      %{text: "what's the weather", correct_intent: "weather"},
      %{text: "play some music", correct_intent: "music.play"},
      %{text: "turn on the lights", correct_intent: "smarthome.lights.on"},
      %{text: "set an alarm", correct_intent: "alarm.set"},
      %{text: "what's on my calendar", correct_intent: "calendar.query.today"},
      %{text: "add task buy groceries", correct_intent: "todo.add"},
      %{text: "set a timer", correct_intent: "timer.set"}
    ]

    write_negative_examples("communication", negative_examples)
  end

  # ============================================================================
  # Entertainment Domain
  # ============================================================================

  defp generate_entertainment_intents do
    Mix.shell().info("Generating entertainment intents...")

    intents = [
      {"movie.search", movie_search_templates()},
      {"movie.recommend", movie_recommend_templates()},
      {"movie.info", movie_info_templates()},
      {"movie.play", movie_play_templates()},
      {"tv.search", tv_search_templates()},
      {"tv.recommend", tv_recommend_templates()},
      {"tv.info", tv_info_templates()},
      {"sports.score", sports_score_templates()},
      {"sports.schedule", sports_schedule_templates()},
      {"sports.standings", sports_standings_templates()},
      {"game.play", game_play_templates()},
      {"game.trivia", game_trivia_templates()},
      {"game.joke", game_joke_templates()},
      {"book.search", book_search_templates()},
      {"book.recommend", book_recommend_templates()}
    ]

    Enum.each(intents, fn {intent_name, templates} ->
      generate_intent_file(intent_name, templates)
    end)

    generate_entertainment_negative_examples()
    Mix.shell().info("Entertainment intents generated!")
  end

  defp movie_search_templates do
    ["find movies about {topic}", "search for {movie_title}", "movies with {actor}", "find action movies", "search for comedy films", "movies directed by {director}", "find sci-fi movies", "search horror movies", "find movies from 2024", "look for thriller movies", "find romantic comedies", "search drama films", "movies like {movie_title}", "find animated movies", "search for new movies", "find {genre} movies", "look up movies with {actor}", "find movies starring {actor}", "search for films by {director}", "movies about aliens"]
  end

  defp movie_recommend_templates do
    ["recommend a movie", "suggest a good movie", "what movie should I watch", "give me a movie recommendation", "recommend a {genre} movie", "suggest something to watch", "what's a good movie", "recommend me a film", "movie suggestions", "what should I watch tonight", "suggest a movie for date night", "recommend a family movie", "good movies to watch", "what's a good {genre} film", "suggest a classic movie", "recommend something new", "movie recommendation please", "what movie do you suggest", "give me a good movie", "recommend a movie like {movie_title}"]
  end

  defp movie_info_templates do
    ["tell me about {movie_title}", "info about {movie_title}", "what is {movie_title} about", "who's in {movie_title}", "who directed {movie_title}", "when did {movie_title} come out", "how long is {movie_title}", "rating of {movie_title}", "cast of {movie_title}", "plot of {movie_title}", "what's {movie_title} about", "details about {movie_title}", "{movie_title} information", "give me info on {movie_title}", "what year was {movie_title}", "who stars in {movie_title}", "synopsis of {movie_title}", "summary of {movie_title}", "reviews of {movie_title}", "is {movie_title} good"]
  end

  defp movie_play_templates do
    ["play {movie_title}", "start {movie_title}", "watch {movie_title}", "play the movie {movie_title}", "put on {movie_title}", "I want to watch {movie_title}", "can you play {movie_title}", "show {movie_title}", "stream {movie_title}", "play movie {movie_title}", "start playing {movie_title}", "let's watch {movie_title}", "begin {movie_title}", "turn on {movie_title}", "queue up {movie_title}", "load {movie_title}", "open {movie_title}", "run {movie_title}", "play the film {movie_title}", "start the movie {movie_title}"]
  end

  defp tv_search_templates do
    ["find TV shows about {topic}", "search for {tv_show}", "TV shows with {actor}", "find comedy shows", "search for drama series", "shows like {tv_show}", "find new TV shows", "look for reality shows", "search sitcoms", "find documentary series", "TV shows from 2024", "look up {tv_show}", "find shows starring {actor}", "search TV series", "find streaming shows", "look for {genre} shows", "find binge-worthy shows", "search for miniseries", "TV shows on Netflix", "find popular shows"]
  end

  defp tv_recommend_templates do
    ["recommend a TV show", "suggest a good show", "what show should I watch", "give me a show recommendation", "recommend a {genre} show", "suggest something to binge", "what's a good series", "recommend me a show", "show suggestions", "what should I binge", "suggest a show for tonight", "recommend a funny show", "good shows to watch", "what series do you suggest", "suggest a new show", "recommend something to stream", "TV recommendation please", "what show do you recommend", "give me a good show", "recommend a show like {tv_show}"]
  end

  defp tv_info_templates do
    ["tell me about {tv_show}", "info about {tv_show}", "what is {tv_show} about", "who's in {tv_show}", "how many seasons of {tv_show}", "when did {tv_show} start", "cast of {tv_show}", "plot of {tv_show}", "what's {tv_show} about", "details about {tv_show}", "{tv_show} information", "is {tv_show} still on", "who stars in {tv_show}", "synopsis of {tv_show}", "summary of {tv_show}", "reviews of {tv_show}", "is {tv_show} good", "where can I watch {tv_show}", "episodes of {tv_show}", "when does {tv_show} air"]
  end

  defp sports_score_templates do
    ["what's the score", "score of the {sports_team} game", "how's the {sports_team} doing", "did the {sports_team} win", "final score of {sports_team}", "{sports_team} score", "who won the game", "current score", "what's the {sports_team} score", "game score", "score update", "how did {sports_team} do", "result of {sports_team} game", "who's winning", "latest score", "sports scores", "{sports_league} scores", "today's scores", "score of today's game", "what was the final"]
  end

  defp sports_schedule_templates do
    ["when do the {sports_team} play", "next {sports_team} game", "{sports_team} schedule", "when's the next game", "upcoming {sports_team} games", "what time is the game", "game schedule", "when does {sports_team} play next", "{sports_league} schedule", "next game time", "upcoming games", "today's games", "games this weekend", "when is the match", "sports schedule", "what games are on today", "game times", "{sports_team} next match", "schedule for {sports_team}", "when do they play"]
  end

  defp sports_standings_templates do
    ["show {sports_league} standings", "team standings", "who's in first place", "league standings", "{sports_team} standings", "current standings", "{sports_league} rankings", "who's leading the league", "playoff standings", "division standings", "conference standings", "where is {sports_team} in standings", "top teams in {sports_league}", "standings table", "who's on top", "league leaders", "current rankings", "team rankings", "who's winning the division", "playoff picture"]
  end

  defp game_play_templates do
    ["let's play a game", "play a game with me", "I want to play a game", "can we play a game", "start a game", "play something", "let's play", "game time", "play with me", "want to play a game", "begin a game", "games to play", "play a fun game", "let's have fun", "interactive game", "start playing", "game on", "play a word game", "play twenty questions", "play a guessing game"]
  end

  defp game_trivia_templates do
    ["play trivia", "trivia game", "ask me trivia", "let's play trivia", "trivia question", "quiz me", "start trivia", "trivia time", "give me a trivia question", "play quiz", "ask me a question", "test my knowledge", "trivia challenge", "random trivia", "fun facts quiz", "general knowledge quiz", "play a quiz game", "trivia please", "quiz game", "knowledge test"]
  end

  defp game_joke_templates do
    ["tell me a joke", "make me laugh", "say something funny", "joke please", "give me a joke", "I need a laugh", "tell a joke", "funny joke", "make a joke", "got any jokes", "hear a joke", "random joke", "tell me something funny", "cheer me up with a joke", "comedy time", "dad joke", "knock knock joke", "pun please", "witty joke", "humor me"]
  end

  defp book_search_templates do
    ["find books about {topic}", "search for {book_title}", "books by {person}", "find mystery books", "search for romance novels", "books like {book_title}", "find new books", "look for thrillers", "search fiction books", "find sci-fi novels", "books from this year", "look up {book_title}", "find bestsellers", "search non-fiction", "find audiobooks", "look for {genre} books", "find highly rated books", "search for classics", "books to read", "find popular books"]
  end

  defp book_recommend_templates do
    ["recommend a book", "suggest a good book", "what book should I read", "give me a book recommendation", "recommend a {genre} book", "suggest something to read", "what's a good book", "recommend me a novel", "book suggestions", "what should I read next", "suggest a book for vacation", "recommend a page-turner", "good books to read", "what book do you suggest", "suggest a classic", "recommend something new", "book recommendation please", "what book do you recommend", "give me a good book", "recommend a book like {book_title}"]
  end

  defp generate_entertainment_negative_examples do
    negative_examples = [
      %{text: "remind me to watch a movie", correct_intent: "reminder.set"},
      %{text: "schedule movie night", correct_intent: "calendar.event.create"},
      %{text: "what's the weather", correct_intent: "weather"},
      %{text: "play music", correct_intent: "music.play"},
      %{text: "turn on the TV", correct_intent: "smarthome.tv.on"},
      %{text: "call John", correct_intent: "call.make"},
      %{text: "add to my reading list", correct_intent: "todo.add"},
      %{text: "set a game timer", correct_intent: "timer.set"},
      %{text: "what time is it", correct_intent: "time.current"},
      %{text: "search the web", correct_intent: "websearch.query"}
    ]

    write_negative_examples("entertainment", negative_examples)
  end

  # ============================================================================
  # Education Domain
  # ============================================================================

  defp generate_education_intents do
    Mix.shell().info("Generating education intents...")

    intents = [
      {"knowledge.define", knowledge_define_templates()},
      {"knowledge.synonym", knowledge_synonym_templates()},
      {"knowledge.spell", knowledge_spell_templates()},
      {"knowledge.fact", knowledge_fact_templates()},
      {"knowledge.who", knowledge_who_templates()},
      {"knowledge.what", knowledge_what_templates()},
      {"knowledge.when", knowledge_when_templates()},
      {"knowledge.where", knowledge_where_templates()},
      {"knowledge.how", knowledge_how_templates()},
      {"translate", translate_templates()},
      {"translate.to_language", translate_to_templates()},
      {"math.calculate", math_calculate_templates()},
      {"math.percentage", math_percentage_templates()},
      {"math.convert", math_convert_templates()},
      {"reference.capital", reference_capital_templates()},
      {"reference.population", reference_population_templates()},
      {"reference.distance", reference_distance_templates()}
    ]

    Enum.each(intents, fn {intent_name, templates} ->
      generate_intent_file(intent_name, templates)
    end)

    generate_education_negative_examples()
    Mix.shell().info("Education intents generated!")
  end

  defp knowledge_define_templates do
    ["define {word}", "what does {word} mean", "definition of {word}", "what is the definition of {word}", "meaning of {word}", "what's {word} mean", "define the word {word}", "tell me what {word} means", "{word} definition", "what is {word}", "explain {word}", "what's the meaning of {word}", "dictionary {word}", "look up {word}", "what does the word {word} mean", "define {word} for me", "meaning of the word {word}", "what is meant by {word}", "what do you mean by {word}", "give me the definition of {word}"]
  end

  defp knowledge_synonym_templates do
    ["synonym for {word}", "what's another word for {word}", "synonyms of {word}", "different word for {word}", "give me a synonym for {word}", "what can I say instead of {word}", "alternative to {word}", "similar word to {word}", "words like {word}", "another way to say {word}", "what else means {word}", "synonym of {word}", "related words to {word}", "like {word} but different", "thesaurus {word}", "other words for {word}", "what's similar to {word}", "equivalent of {word}", "substitute for {word}", "synonyms for the word {word}"]
  end

  defp knowledge_spell_templates do
    ["how do you spell {word}", "spell {word}", "spelling of {word}", "how is {word} spelled", "what's the spelling of {word}", "can you spell {word}", "spell out {word}", "how to spell {word}", "{word} spelling", "correct spelling of {word}", "is {word} spelled correctly", "how do I spell {word}", "spelling for {word}", "what's the correct spelling of {word}", "spell the word {word}", "how should I spell {word}", "letter by letter {word}", "s-p-e-l-l {word}", "help me spell {word}", "check spelling of {word}"]
  end

  defp knowledge_fact_templates do
    ["tell me a fact", "random fact", "give me a fun fact", "interesting fact", "did you know", "fact of the day", "tell me something interesting", "share a fact", "trivia fact", "cool fact", "fun fact please", "tell me something I don't know", "surprising fact", "science fact", "history fact", "nature fact", "world fact", "amazing fact", "educational fact", "quick fact"]
  end

  defp knowledge_who_templates do
    ["who is {person}", "who was {person}", "tell me about {person}", "who's {person}", "information about {person}", "biography of {person}", "facts about {person}", "who is {person} known for", "what did {person} do", "who exactly is {person}", "{person} bio", "details about {person}", "what is {person} famous for", "why is {person} famous", "learn about {person}", "who was {person} really", "history of {person}", "{person} information", "what do you know about {person}", "tell me who {person} is"]
  end

  defp knowledge_what_templates do
    ["what is {topic}", "what are {topic}", "tell me about {topic}", "explain {topic}", "what exactly is {topic}", "information about {topic}", "describe {topic}", "what's {topic}", "facts about {topic}", "what do you know about {topic}", "details about {topic}", "learn about {topic}", "educate me on {topic}", "what can you tell me about {topic}", "explain what {topic} is", "give me info on {topic}", "what is a {topic}", "define {topic}", "overview of {topic}", "{topic} explanation"]
  end

  defp knowledge_when_templates do
    ["when did {topic} happen", "when was {topic}", "what year was {topic}", "when did {topic} occur", "date of {topic}", "when is {topic}", "what date is {topic}", "when was {topic} invented", "when was {topic} discovered", "when did {topic} start", "when did {topic} end", "year of {topic}", "when was {topic} founded", "when did {topic} begin", "when was {topic} built", "what time was {topic}", "when did {topic} take place", "historical date of {topic}", "when exactly was {topic}", "timeline of {topic}"]
  end

  defp knowledge_where_templates do
    ["where is {location}", "where was {topic}", "location of {location}", "where can I find {topic}", "where is {location} located", "where did {topic} happen", "where was {topic} invented", "where was {topic} discovered", "where is {location} on the map", "where exactly is {location}", "location of {topic}", "where was {topic} founded", "where did {topic} originate", "where can I see {topic}", "where is the {location}", "in which country is {location}", "where was {topic} built", "where does {topic} come from", "geographical location of {location}", "where did {topic} take place"]
  end

  defp knowledge_how_templates do
    ["how does {topic} work", "how do {topic} work", "explain how {topic} works", "how is {topic} made", "how to {topic}", "how can I {topic}", "how did {topic} happen", "process of {topic}", "how is {topic} done", "steps for {topic}", "how do you {topic}", "how does {topic} function", "mechanism of {topic}", "how was {topic} created", "how is {topic} produced", "how do I {topic}", "explain the process of {topic}", "how does {topic} operate", "how can {topic} be done", "way to {topic}"]
  end

  defp translate_templates do
    ["translate {word}", "how do you say {word}", "translate this", "what is {word} in other languages", "translate {word} please", "can you translate {word}", "translation of {word}", "help me translate", "translate the word {word}", "what does {word} mean in other languages", "translate for me", "how to say {word}", "give me translation", "translate {word} for me", "I need a translation", "help with translation", "translation please", "what's the translation of {word}", "can you translate this", "translate the phrase"]
  end

  defp translate_to_templates do
    ["translate {word} to {language}", "how do you say {word} in {language}", "{word} in {language}", "what is {word} in {language}", "translate to {language}", "say {word} in {language}", "{language} translation of {word}", "convert {word} to {language}", "how is {word} said in {language}", "{word} translated to {language}", "translate {word} into {language}", "what's {word} in {language}", "tell me {word} in {language}", "{language} word for {word}", "how would you say {word} in {language}", "express {word} in {language}", "{word} to {language}", "translation of {word} to {language}", "in {language} what is {word}", "{language} for {word}"]
  end

  defp math_calculate_templates do
    ["calculate {math_expression}", "what is {math_expression}", "compute {math_expression}", "{math_expression} equals", "solve {math_expression}", "what's {math_expression}", "work out {math_expression}", "figure out {math_expression}", "{math_expression} result", "answer to {math_expression}", "do the math {math_expression}", "calculate {math_expression} for me", "what does {math_expression} equal", "evaluate {math_expression}", "process {math_expression}", "find {math_expression}", "determine {math_expression}", "compute the result of {math_expression}", "mathematical result of {math_expression}", "{math_expression} solution"]
  end

  defp math_percentage_templates do
    ["what's {number} percent of {number}", "{number}% of {number}", "calculate {number} percent", "percentage of {number}", "what is {number} percent of {number}", "find {number}% of {number}", "compute {number} percent of {number}", "{number} percent of {number} is", "work out {number}%", "figure {number} percent", "how much is {number}% of {number}", "{number} percentage of {number}", "what's the percentage", "calculate percentage", "percent calculation", "{number}% calculation", "what percentage is {number} of {number}", "find the percentage", "percent of {number}", "percentage calculator"]
  end

  defp math_convert_templates do
    ["convert {number} {unit_from} to {unit_to}", "{number} {unit_from} in {unit_to}", "how many {unit_to} is {number} {unit_from}", "{number} {unit_from} to {unit_to}", "what is {number} {unit_from} in {unit_to}", "change {number} {unit_from} to {unit_to}", "{unit_from} to {unit_to} conversion", "convert {unit_from} to {unit_to}", "{number} {unit_from} equals how many {unit_to}", "calculate {number} {unit_from} in {unit_to}", "how much is {number} {unit_from} in {unit_to}", "transform {number} {unit_from} to {unit_to}", "unit conversion {unit_from} {unit_to}", "{unit_from} {unit_to} calculator", "convert units {unit_from} {unit_to}", "measurement conversion", "{number} {unit_from} converted to {unit_to}", "what's {number} {unit_from} in {unit_to}", "{unit_from} into {unit_to}", "change units"]
  end

  defp reference_capital_templates do
    ["what is the capital of {location}", "capital of {location}", "{location} capital", "what's the capital of {location}", "capital city of {location}", "name the capital of {location}", "tell me the capital of {location}", "which city is the capital of {location}", "capital of the country {location}", "what's {location}'s capital", "{location}'s capital city", "where is the capital of {location}", "what is {location} capital", "find capital of {location}", "capital for {location}", "main city of {location}", "what city is the capital of {location}", "identify capital of {location}", "{location} capital name", "government seat of {location}"]
  end

  defp reference_population_templates do
    ["what is the population of {location}", "population of {location}", "how many people live in {location}", "{location} population", "what's the population of {location}", "how big is {location}", "number of people in {location}", "tell me the population of {location}", "{location}'s population", "population count for {location}", "how many inhabitants in {location}", "citizens of {location}", "population size of {location}", "what is {location} population", "find population of {location}", "how populous is {location}", "demographic of {location}", "people in {location}", "population figure for {location}", "residents of {location}"]
  end

  defp reference_distance_templates do
    ["how far is {location} from {location}", "distance between {location} and {location}", "how far is it to {location}", "distance to {location}", "{location} to {location} distance", "how many miles to {location}", "how far from here to {location}", "travel distance to {location}", "what's the distance to {location}", "km from {location} to {location}", "miles between {location} and {location}", "how far away is {location}", "distance from {location}", "length from {location} to {location}", "how long is the drive to {location}", "measure distance to {location}", "calculate distance to {location}", "route distance {location} {location}", "how far to travel to {location}", "distance calculator {location}"]
  end

  defp generate_education_negative_examples do
    negative_examples = [
      %{text: "remind me to study", correct_intent: "reminder.set"},
      %{text: "schedule a class", correct_intent: "calendar.event.create"},
      %{text: "play educational music", correct_intent: "music.play"},
      %{text: "search for schools", correct_intent: "websearch.query"},
      %{text: "add homework to my list", correct_intent: "todo.add"},
      %{text: "call my teacher", correct_intent: "call.make"},
      %{text: "what's the weather", correct_intent: "weather"},
      %{text: "turn on the lights", correct_intent: "smarthome.lights.on"},
      %{text: "set a study timer", correct_intent: "timer.set"},
      %{text: "send an email to professor", correct_intent: "email.compose"}
    ]

    write_negative_examples("education", negative_examples)
  end

  # ============================================================================
  # Utilities Domain
  # ============================================================================

  defp generate_utilities_intents do
    Mix.shell().info("Generating utilities intents...")

    intents = [
      {"time.current", time_current_templates()},
      {"time.in_location", time_location_templates()},
      {"date.current", date_current_templates()},
      {"date.day_of_week", date_day_templates()},
      {"random.number", random_number_templates()},
      {"random.coin", random_coin_templates()},
      {"random.dice", random_dice_templates()},
      {"random.choice", random_choice_templates()},
      {"device.battery", device_battery_templates()},
      {"device.storage", device_storage_templates()},
      {"device.wifi", device_wifi_templates()},
      {"device.volume.set", device_volume_templates()},
      {"device.brightness.set", device_brightness_templates()},
      {"device.screenshot", device_screenshot_templates()},
      {"device.settings", device_settings_templates()},
      {"device.location", device_location_templates()}
    ]

    Enum.each(intents, fn {intent_name, templates} ->
      generate_intent_file(intent_name, templates)
    end)

    generate_utilities_negative_examples()
    Mix.shell().info("Utilities intents generated!")
  end

  defp time_current_templates do
    ["what time is it", "current time", "what's the time", "time please", "tell me the time", "what time is it now", "the time", "time now", "what's the current time", "do you have the time", "can you tell me the time", "what is the time", "show me the time", "time check", "give me the time", "present time", "time right now", "actual time", "what time do we have", "clock check"]
  end

  defp time_location_templates do
    ["what time is it in {location}", "time in {location}", "current time in {location}", "{location} time", "what's the time in {location}", "tell me the time in {location}", "time zone for {location}", "what time in {location}", "{location} current time", "local time in {location}", "time right now in {location}", "what's {location} time", "give me time in {location}", "check time in {location}", "what is the time in {location}", "show time in {location}", "clock in {location}", "{location} timezone", "how late is it in {location}", "what hour is it in {location}"]
  end

  defp date_current_templates do
    ["what's today's date", "what is the date", "today's date", "what day is it", "the date today", "current date", "what's the date today", "tell me the date", "date please", "what date is it", "show me the date", "give me today's date", "date today", "what is today's date", "present date", "what day is today", "the date", "check the date", "what's today", "today is"]
  end

  defp date_day_templates do
    ["what day of the week is {date_time}", "what day is {date_time}", "is {date_time} a weekday", "what day of week is {date_time}", "which day is {date_time}", "day of {date_time}", "what day will {date_time} be", "tell me what day {date_time} is", "is {date_time} a weekend", "what day does {date_time} fall on", "{date_time} day of week", "find day for {date_time}", "what day was {date_time}", "check day of {date_time}", "which weekday is {date_time}", "day for {date_time}", "what day of the week is it on {date_time}", "day on {date_time}", "{date_time} is what day", "when is {date_time}"]
  end

  defp random_number_templates do
    ["pick a random number", "random number", "give me a random number", "generate a random number", "pick a number", "random number between {number} and {number}", "choose a random number", "number between {number} and {number}", "random from {number} to {number}", "pick any number", "generate number", "random number please", "give me a number", "select a random number", "number generator", "pick a number for me", "random integer", "choose a number", "roll a random number", "any random number"]
  end

  defp random_coin_templates do
    ["flip a coin", "coin flip", "heads or tails", "toss a coin", "flip coin", "coin toss", "random coin flip", "flip", "toss coin", "heads tails", "call the coin", "throw a coin", "coin flip please", "flip it", "do a coin flip", "50 50", "flip a coin for me", "let the coin decide", "coin", "random flip"]
  end

  defp random_dice_templates do
    ["roll a dice", "roll dice", "throw dice", "dice roll", "roll the dice", "roll a die", "toss dice", "roll", "dice throw", "give me a dice roll", "roll for me", "random dice", "dice please", "throw the dice", "roll a d6", "roll two dice", "dice game", "roll dice for me", "cast dice", "dice toss"]
  end

  defp random_choice_templates do
    ["pick between {topic} and {topic}", "choose between {topic} or {topic}", "random choice", "help me decide", "pick one", "which should I choose", "make a choice", "decide for me", "random pick", "choose for me", "pick something", "help me pick", "random selection", "which one", "either {topic} or {topic}", "select one", "choose one for me", "make the decision", "I can't decide", "you pick"]
  end

  defp device_battery_templates do
    ["what's my battery level", "battery status", "how much battery", "check battery", "battery percentage", "battery level", "how's my battery", "show battery", "battery remaining", "current battery", "battery check", "power level", "how much charge left", "battery life", "percentage of battery", "my battery", "device battery", "battery info", "check power level", "remaining battery"]
  end

  defp device_storage_templates do
    ["how much storage do I have", "storage space", "check storage", "available storage", "storage left", "how much space left", "disk space", "free space", "storage status", "my storage", "device storage", "remaining storage", "storage check", "memory space", "available space", "how full is my storage", "storage info", "check disk space", "space remaining", "storage available"]
  end

  defp device_wifi_templates do
    ["wifi status", "am I connected to wifi", "check wifi", "wifi connection", "connected to wifi", "show wifi", "wifi network", "which wifi", "what wifi am I on", "wifi info", "check internet connection", "is wifi connected", "my wifi", "current wifi", "network status", "internet connection", "wifi check", "connected network", "wifi name", "show my wifi"]
  end

  defp device_volume_templates do
    ["set volume to {number}", "volume {number}", "turn volume to {number}", "set the volume to {number} percent", "change volume to {number}", "volume at {number}", "make volume {number}", "adjust volume to {number}", "volume level {number}", "set audio to {number}", "turn up volume to {number}", "turn down volume to {number}", "volume {number} percent", "system volume {number}", "media volume {number}", "set sound to {number}", "audio level {number}", "put volume at {number}", "speaker volume {number}", "change sound to {number}"]
  end

  defp device_brightness_templates do
    ["set brightness to {number}", "brightness {number}", "screen brightness {number}", "turn brightness to {number}", "change brightness to {number}", "brightness level {number}", "adjust brightness to {number}", "display brightness {number}", "make it brighter", "make it dimmer", "increase brightness", "decrease brightness", "lower brightness", "raise brightness", "brightness up", "brightness down", "dim the screen", "brighten the screen", "set screen to {number}", "screen at {number} percent"]
  end

  defp device_screenshot_templates do
    ["take a screenshot", "screenshot", "capture screen", "screen capture", "take screenshot", "grab screenshot", "capture the screen", "screenshot please", "take a screen shot", "screen shot", "capture this screen", "save screenshot", "snap the screen", "get a screenshot", "screenshot now", "take screen capture", "capture display", "screenshot this", "screen grab", "take a picture of the screen"]
  end

  defp device_settings_templates do
    ["open settings", "go to settings", "settings", "show settings", "device settings", "open device settings", "system settings", "access settings", "settings menu", "configuration", "preferences", "open preferences", "phone settings", "app settings", "general settings", "show me settings", "take me to settings", "launch settings", "settings please", "open system settings"]
  end

  defp device_location_templates do
    ["where am I", "my location", "current location", "show my location", "what's my location", "find my location", "my current location", "location", "GPS location", "where is my location", "show where I am", "get my location", "my position", "current position", "locate me", "pin my location", "my coordinates", "where am I right now", "location check", "tell me where I am"]
  end

  defp generate_utilities_negative_examples do
    negative_examples = [
      %{text: "set a reminder", correct_intent: "reminder.set"},
      %{text: "schedule a meeting", correct_intent: "calendar.meeting.schedule"},
      %{text: "what's the weather", correct_intent: "weather"},
      %{text: "play music", correct_intent: "music.play"},
      %{text: "call someone", correct_intent: "call.make"},
      %{text: "send a text", correct_intent: "message.send"},
      %{text: "add to my list", correct_intent: "todo.add"},
      %{text: "set an alarm", correct_intent: "alarm.set"},
      %{text: "turn on the lights", correct_intent: "smarthome.lights.on"},
      %{text: "search the web", correct_intent: "websearch.query"}
    ]

    write_negative_examples("utilities", negative_examples)
  end

  # ============================================================================
  # Display Domain (Star Trek-inspired)
  # ============================================================================

  defp generate_display_intents do
    Mix.shell().info("Generating display intents...")

    intents = [
      {"display.show", display_show_templates()},
      {"display.show_image", display_show_image_templates()},
      {"display.show_chart", display_show_chart_templates()},
      {"display.freeze", display_freeze_templates()},
      {"display.enhance", display_enhance_templates()},
      {"display.magnify", display_magnify_templates()},
      {"display.zoom_in", display_zoom_in_templates()},
      {"display.zoom_out", display_zoom_out_templates()},
      {"display.compare", display_compare_templates()},
      {"display.overlay", display_overlay_templates()},
      {"display.play", display_play_templates()},
      {"display.pause", display_pause_templates()},
      {"display.stop", display_stop_templates()}
    ]

    Enum.each(intents, fn {intent_name, templates} ->
      generate_intent_file(intent_name, templates)
    end)

    generate_display_negative_examples()
    Mix.shell().info("Display intents generated!")
  end

  defp display_show_templates do
    ["show me the {file_name}", "display the {file_name}", "put {file_name} on screen", "show {file_name}", "display {file_name}", "let me see {file_name}", "bring up {file_name}", "show the {file_name}", "display the {file_name} file", "pull up {file_name}", "open and display {file_name}", "show {file_name} on screen", "view {file_name}", "present {file_name}", "exhibit {file_name}", "on screen {file_name}", "display data from {file_name}", "show me {file_name} data", "visualize {file_name}", "render {file_name}"]
  end

  defp display_show_image_templates do
    ["show the image", "display the picture", "show me that image", "put the image on screen", "display that picture", "show the photo", "bring up the image", "show image", "display image", "view the image", "present the image", "show the picture on screen", "display the photo", "let me see the image", "pull up the image", "image on screen", "show the visual", "display the visual", "open the image", "show photograph"]
  end

  defp display_show_chart_templates do
    ["show me a chart of {query_term}", "display a graph of {query_term}", "chart {query_term}", "graph the {query_term}", "show {query_term} chart", "display {query_term} graph", "visualize {query_term}", "plot {query_term}", "show graph of {query_term}", "chart of {query_term}", "display chart for {query_term}", "graph of {query_term}", "show data chart", "display bar chart", "show line graph", "pie chart of {query_term}", "histogram of {query_term}", "trend chart {query_term}", "show visualization", "data graph {query_term}"]
  end

  defp display_freeze_templates do
    ["freeze", "freeze frame", "freeze image", "hold that image", "freeze the display", "pause the image", "stop there", "freeze that", "hold the frame", "freeze on this", "capture this frame", "lock the image", "freeze playback", "hold it there", "freeze screen", "stop the frame", "hold this", "freeze the picture", "static hold", "stop on this frame"]
  end

  defp display_enhance_templates do
    ["enhance", "enhance that section", "enhance the image", "improve the image", "enhance {section_id}", "sharpen the image", "clarify the image", "enhance resolution", "make it clearer", "enhance that area", "improve clarity", "enhance detail", "sharpen that", "make it sharper", "enhance quality", "better resolution", "clean up the image", "enhance and clarify", "improve the picture", "detail enhance"]
  end

  defp display_magnify_templates do
    ["magnify {section_id}", "magnify that section", "enlarge {section_id}", "magnify the image", "zoom into {section_id}", "magnify by {magnification_level}", "expand {section_id}", "magnify the selected area", "increase magnification", "magnify that area", "blow up {section_id}", "magnify to {magnification_level}", "magnification {magnification_level}", "enlarge that section", "magnify the display", "expand the view", "magnify region", "magnify on {section_id}", "larger view of {section_id}", "amplify {section_id}"]
  end

  defp display_zoom_in_templates do
    ["zoom in", "zoom in on that", "get closer", "zoom in on {section_id}", "closer look", "zoom in more", "increase zoom", "zoom closer", "move in", "tighter shot", "zoom in on the {section_id}", "go closer", "magnify view", "zoom in please", "closer view", "zoom in to {magnification_level}", "enhance and zoom", "tighten the view", "zoom in further", "get a closer look"]
  end

  defp display_zoom_out_templates do
    ["zoom out", "pull back", "wider view", "zoom out more", "decrease zoom", "step back", "full view", "zoom out please", "wider shot", "back out", "reduce zoom", "zoom out to see more", "pull back the view", "expand view", "show more", "zoom all the way out", "full screen view", "overall view", "complete picture", "zoom out further"]
  end

  defp display_compare_templates do
    ["compare these two", "show side by side", "compare {file_name} with {file_name}", "comparison view", "put them side by side", "compare the images", "show comparison", "side by side comparison", "compare {query_term}", "show the difference", "compare these files", "visual comparison", "compare data", "show both", "differential view", "contrast these", "compare versions", "show differences between", "dual view comparison", "juxtapose these"]
  end

  defp display_overlay_templates do
    ["overlay the two", "superimpose", "overlay {file_name} on {file_name}", "combine the images", "put one over the other", "overlay these", "superimpose the images", "merge the displays", "layer them", "overlay data", "composite view", "blend the images", "stack the views", "overlay on top", "superimpose data", "combine views", "layer the displays", "merge these", "create overlay", "stack these images"]
  end

  defp display_play_templates do
    ["play the recording", "start playback", "play", "begin playback", "play the video", "start the recording", "play it", "begin playing", "run the recording", "play the footage", "start playing", "resume playback", "play the file", "initiate playback", "run video", "play from the beginning", "start video", "commence playback", "play this", "activate playback"]
  end

  defp display_pause_templates do
    ["pause", "pause playback", "pause the video", "hold playback", "stop playing temporarily", "pause it", "freeze playback", "suspend playback", "pause the recording", "halt playback", "pause here", "stop here", "temporary stop", "pause display", "hold it", "pause the footage", "stop playback", "break playback", "pause playing", "hold the video"]
  end

  defp display_stop_templates do
    ["stop", "stop playback", "end playback", "stop the video", "halt", "stop playing", "terminate playback", "end the recording", "stop the footage", "cease playback", "stop it", "kill playback", "abort playback", "stop the display", "end video", "finish playback", "stop this", "discontinue", "shut off playback", "close playback"]
  end

  defp generate_display_negative_examples do
    negative_examples = [
      %{text: "play some music", correct_intent: "music.play"},
      %{text: "show me the weather", correct_intent: "weather"},
      %{text: "turn on the TV", correct_intent: "smarthome.tv.on"},
      %{text: "take a screenshot", correct_intent: "device.screenshot"},
      %{text: "find a file", correct_intent: "search.locate_file"},
      %{text: "analyze the data", correct_intent: "analyze.data"},
      %{text: "what time is it", correct_intent: "time.current"},
      %{text: "set the brightness", correct_intent: "device.brightness.set"},
      %{text: "search for images", correct_intent: "search.find_by_type"},
      %{text: "show my calendar", correct_intent: "calendar.query.today"}
    ]

    write_negative_examples("display", negative_examples)
  end

  # ============================================================================
  # Search Domain (Star Trek-inspired)
  # ============================================================================

  defp generate_search_intents do
    Mix.shell().info("Generating search intents...")

    intents = [
      {"search.locate_person", search_locate_person_templates()},
      {"search.locate_file", search_locate_file_templates()},
      {"search.identify", search_identify_templates()},
      {"search.find_in_database", search_database_templates()},
      {"search.find_by_date", search_by_date_templates()},
      {"search.find_by_type", search_by_type_templates()},
      {"search.cross_reference", search_cross_reference_templates()},
      {"search.find_related", search_find_related_templates()},
      {"search.find_duplicates", search_find_duplicates_templates()},
      {"search.how_many", search_how_many_templates()},
      {"search.count", search_count_templates()},
      {"search.access_file", search_access_file_templates()},
      {"search.access_log", search_access_log_templates()},
      {"search.confirm", search_confirm_templates()},
      {"search.verify", search_verify_templates()}
    ]

    Enum.each(intents, fn {intent_name, templates} ->
      generate_intent_file(intent_name, templates)
    end)

    generate_search_negative_examples()
    Mix.shell().info("Search intents generated!")
  end

  defp search_locate_person_templates do
    ["locate {person}", "where is {person}", "find {person}", "location of {person}", "where can I find {person}", "track {person}", "position of {person}", "find the location of {person}", "where's {person}", "locate {person} please", "find where {person} is", "search for {person}", "get location of {person}", "look for {person}", "trace {person}", "pinpoint {person}", "find {person}'s location", "where is {person} now", "locate {person} currently", "find {person} for me"]
  end

  defp search_locate_file_templates do
    ["find the {file_name}", "locate {file_name}", "where is {file_name}", "search for {file_name}", "find {file_name}", "look for {file_name}", "locate the {file_name} file", "find file {file_name}", "where's the {file_name}", "search {file_name}", "find the file called {file_name}", "locate document {file_name}", "where is the {file_name} document", "find {file_name} please", "look up {file_name}", "search for file {file_name}", "get {file_name}", "pull up {file_name}", "find document {file_name}", "locate {file_name} for me"]
  end

  defp search_identify_templates do
    ["identify this", "what is this", "identify the object", "recognize this", "what am I looking at", "identify", "can you identify this", "tell me what this is", "identify this item", "what is that", "identify the subject", "recognition", "who or what is this", "analyze and identify", "determine what this is", "identify for me", "what's this", "identify the contents", "scan and identify", "ID this"]
  end

  defp search_database_templates do
    ["search the database for {query_term}", "find {query_term} in the database", "database search {query_term}", "query database for {query_term}", "look up {query_term} in database", "search {data_source} for {query_term}", "find in database {query_term}", "database query {query_term}", "search records for {query_term}", "find {query_term} in records", "query {data_source} for {query_term}", "search {data_source}", "look in database for {query_term}", "database lookup {query_term}", "search for {query_term} in {data_source}", "pull records for {query_term}", "find data on {query_term}", "search all records for {query_term}", "query for {query_term}", "database find {query_term}"]
  end

  defp search_by_date_templates do
    ["find entries from {date_range}", "search records from {date_range}", "show entries from {date_range}", "find items from {date_range}", "search by date {date_range}", "get records from {date_range}", "entries dated {date_range}", "find {date_range} records", "search {date_range}", "items from {date_range}", "data from {date_range}", "records dated {date_range}", "find from {date_range}", "search {date_range} data", "show {date_range} entries", "pull {date_range} records", "filter by {date_range}", "date range {date_range}", "get {date_range} data", "find during {date_range}"]
  end

  defp search_by_type_templates do
    ["find all {file_name} files", "search for {file_name} type", "list all {file_name}", "show me {file_name} files", "find files of type {file_name}", "filter by type {file_name}", "get all {file_name}", "find {file_name} documents", "search for {file_name}", "list {file_name} files", "show {file_name} items", "all {file_name} files", "find by type {file_name}", "filter {file_name}", "type search {file_name}", "get files matching {file_name}", "locate {file_name} files", "find {file_name} format", "search type {file_name}", "show all {file_name}"]
  end

  defp search_cross_reference_templates do
    ["cross reference these", "cross reference {query_term} with {query_term}", "compare and cross reference", "find correlations", "cross check the data", "reference against each other", "cross reference the information", "check for cross references", "correlate these entries", "cross reference records", "find connections between", "match and compare", "cross check these", "reference check", "find links between", "correlate the data", "cross match", "check against each other", "find overlaps", "cross reference analysis"]
  end

  defp search_find_related_templates do
    ["find related entries", "show related items", "find similar", "related records", "associated entries", "find connections", "show related data", "linked items", "find related to {query_term}", "similar entries", "related to {query_term}", "show associated", "find matching entries", "related information", "find linked", "show connected items", "find associations", "related data", "show similar", "connected records"]
  end

  defp search_find_duplicates_templates do
    ["find duplicates", "check for duplicates", "duplicate entries", "find duplicate records", "show duplicates", "identify duplicates", "duplicate check", "find repeated entries", "search for duplicates", "duplicate detection", "find copies", "locate duplicates", "show duplicate items", "find matching duplicates", "detect duplicate data", "duplicates in the system", "find redundant entries", "check for copies", "identify repeated data", "scan for duplicates"]
  end

  defp search_how_many_templates do
    ["how many entries", "how many {query_term}", "count of {query_term}", "number of {query_term}", "how many records match", "total {query_term}", "how many items", "quantity of {query_term}", "how many are there", "count {query_term}", "how many results", "total number of {query_term}", "how many entries for {query_term}", "amount of {query_term}", "how many matching", "how many found", "query count", "result count", "how many exist", "total count of {query_term}"]
  end

  defp search_count_templates do
    ["count the entries", "count {query_term}", "get the count", "tally the results", "count all", "number of records", "count items", "total count", "enumerate", "count matching entries", "count the {query_term}", "total up", "count records", "get count of {query_term}", "sum up the entries", "count all {query_term}", "quantity count", "total number", "count the matches", "record count"]
  end

  defp search_access_file_templates do
    ["access the {file_name}", "open {file_name}", "retrieve {file_name}", "get access to {file_name}", "access {file_name} file", "open the {file_name} file", "load {file_name}", "access the {file_name} document", "retrieve the {file_name}", "get {file_name}", "open file {file_name}", "access file {file_name}", "pull {file_name}", "access record {file_name}", "retrieve file {file_name}", "load the {file_name}", "access document {file_name}", "open record {file_name}", "get the {file_name} file", "access the {file_name} record"]
  end

  defp search_access_log_templates do
    ["access the log", "show the logs", "open activity log", "view the log", "access system logs", "show log entries", "open the log file", "access {system_name} logs", "view activity log", "get the logs", "show system logs", "access log file", "open logs", "retrieve the logs", "access {process_name} log", "show the log file", "view logs", "access event log", "get log entries", "show activity logs"]
  end

  defp search_confirm_templates do
    ["confirm that {query_term}", "verify {query_term} is true", "confirm {query_term}", "is {query_term} correct", "confirm the information", "verify that {query_term}", "please confirm {query_term}", "can you confirm {query_term}", "confirm this is {query_term}", "verify the {query_term}", "is {query_term} accurate", "confirm accuracy of {query_term}", "double check {query_term}", "confirm the data", "is this {query_term} right", "verify correctness", "confirm {query_term} please", "make sure {query_term}", "check if {query_term} is true", "confirm authenticity"]
  end

  defp search_verify_templates do
    ["verify this information", "verify {query_term}", "check the validity", "verify the data", "validate {query_term}", "verify accuracy", "confirm and verify", "verify this is correct", "check authenticity", "verify the record", "validate this information", "verify {query_term} is accurate", "authentication check", "verify the source", "validate accuracy", "verify this data", "check if valid", "verify the entry", "validation check", "verify correctness of {query_term}"]
  end

  defp generate_search_negative_examples do
    negative_examples = [
      %{text: "show me the image", correct_intent: "display.show_image"},
      %{text: "analyze the data", correct_intent: "analyze.data"},
      %{text: "what time is it", correct_intent: "time.current"},
      %{text: "calculate something", correct_intent: "math.calculate"},
      %{text: "status report", correct_intent: "status.report"},
      %{text: "zoom in on that", correct_intent: "display.zoom_in"},
      %{text: "compare the charts", correct_intent: "analyze.compare"},
      %{text: "estimate the time", correct_intent: "compute.estimate"},
      %{text: "check system status", correct_intent: "status.check_system"},
      %{text: "show the chart", correct_intent: "display.show_chart"}
    ]

    write_negative_examples("search", negative_examples)
  end

  # ============================================================================
  # Analysis Domain (Star Trek-inspired)
  # ============================================================================

  defp generate_analysis_intents do
    Mix.shell().info("Generating analysis intents...")

    intents = [
      {"analyze.data", analyze_data_templates()},
      {"analyze.pattern", analyze_pattern_templates()},
      {"analyze.trend", analyze_trend_templates()},
      {"analyze.anomaly", analyze_anomaly_templates()},
      {"analyze.compare", analyze_compare_templates()},
      {"analyze.difference", analyze_difference_templates()},
      {"compute.calculate", compute_calculate_templates()},
      {"compute.sum", compute_sum_templates()},
      {"compute.average", compute_average_templates()},
      {"compute.percentage", compute_percentage_templates()},
      {"compute.estimate", compute_estimate_templates()},
      {"compute.project", compute_project_templates()},
      {"compute.elapsed_time", compute_elapsed_templates()},
      {"compute.eta", compute_eta_templates()},
      {"analyze.diagnose", analyze_diagnose_templates()}
    ]

    Enum.each(intents, fn {intent_name, templates} ->
      generate_intent_file(intent_name, templates)
    end)

    generate_analysis_negative_examples()
    Mix.shell().info("Analysis intents generated!")
  end

  defp analyze_data_templates do
    ["analyze the data", "analyze {data_source}", "run analysis on the data", "analyze this dataset", "perform data analysis", "analyze the {data_source}", "data analysis", "analyze the information", "run data analysis", "analyze these numbers", "examine the data", "analyze the dataset", "process and analyze", "run analysis", "analyze the records", "detailed analysis", "analyze {query_term} data", "analyze the results", "perform analysis on {data_source}", "full data analysis"]
  end

  defp analyze_pattern_templates do
    ["find patterns in the data", "pattern analysis", "detect patterns", "analyze for patterns", "identify patterns", "look for patterns", "pattern recognition", "find recurring patterns", "analyze patterns in {data_source}", "pattern detection", "search for patterns", "what patterns exist", "identify data patterns", "find the pattern", "patterns in {query_term}", "analyze pattern trends", "detect recurring patterns", "spot patterns", "pattern identification", "analyze for pattern recognition"]
  end

  defp analyze_trend_templates do
    ["show me the trend", "analyze trends", "trend analysis", "what's the trend", "identify trends", "find trends", "trending analysis", "analyze {query_term} trends", "show trend for {query_term}", "trend detection", "what are the trends", "analyze the trend", "trend over time", "historical trend", "show trends in {data_source}", "upward or downward trend", "analyze trend data", "find the trend", "trend identification", "analyze for trends"]
  end

  defp analyze_anomaly_templates do
    ["detect anomalies", "find anomalies", "anomaly detection", "look for anomalies", "identify anomalies", "analyze for anomalies", "find outliers", "anomaly analysis", "detect outliers", "unusual data points", "find irregularities", "spot anomalies", "anomaly scan", "check for anomalies", "identify outliers", "find deviations", "abnormal data detection", "locate anomalies", "analyze anomalies in {data_source}", "outlier detection"]
  end

  defp analyze_compare_templates do
    ["compare the datasets", "compare {query_term} with {query_term}", "comparison analysis", "analyze and compare", "compare these two", "run comparison", "compare the data", "side by side comparison", "comparative analysis", "compare {data_source}", "compare the values", "analyze differences", "compare metrics", "comparison of {query_term}", "compare the results", "analyze both", "compare {comparison_target}", "run a comparison", "compare these datasets", "difference comparison"]
  end

  defp analyze_difference_templates do
    ["what's the difference", "show the difference between", "difference analysis", "find the differences", "what changed", "analyze the difference", "difference between {query_term} and {query_term}", "calculate the difference", "compare differences", "identify differences", "what's different", "differential analysis", "show me the changes", "difference from {comparison_target}", "analyze changes", "what differs", "find what changed", "delta analysis", "differences in the data", "show differences"]
  end

  defp compute_calculate_templates do
    ["calculate the result", "compute {math_expression}", "run the calculation", "calculate {query_term}", "perform calculation", "compute the value", "calculate it", "mathematical calculation", "compute this", "calculate the total", "run computation", "figure this out", "compute the result", "calculate for me", "do the calculation", "compute {query_term}", "calculation needed", "work out the math", "calculate these values", "compute the figures"]
  end

  defp compute_sum_templates do
    ["sum these values", "add up the total", "total sum", "sum of {query_term}", "add these numbers", "calculate the sum", "total of all", "sum it up", "add up", "get the total", "sum all values", "total calculation", "sum the numbers", "add together", "cumulative sum", "aggregate total", "sum of the data", "add all", "calculate total sum", "running total"]
  end

  defp compute_average_templates do
    ["what's the average", "calculate the average", "average of {query_term}", "find the mean", "average value", "compute average", "mean value", "average calculation", "get the average", "calculate mean", "find average", "average of these", "compute the mean", "average across", "mean of {query_term}", "overall average", "average calculation for {query_term}", "what's the mean", "calculate the mean", "determine average"]
  end

  defp compute_percentage_templates do
    ["what percentage", "calculate percentage", "percentage of {query_term}", "find the percentage", "percentage calculation", "what percent is {query_term}", "compute percentage", "percentage change", "calculate the percent", "percent of total", "percentage of the whole", "what's the percentage", "percentage analysis", "find the percent", "percentage breakdown", "percent calculation", "what percentage of {query_term}", "calculate percent change", "percentage value", "determine percentage"]
  end

  defp compute_estimate_templates do
    ["estimate the time needed", "provide an estimate", "estimate {query_term}", "give me an estimate", "estimate how long", "estimation", "rough estimate", "estimate the cost", "approximate", "estimate the effort", "time estimate", "estimate duration", "estimate the value", "provide estimation", "calculate estimate", "estimate required time", "estimate for {query_term}", "approximate value", "give estimation", "estimate this"]
  end

  defp compute_project_templates do
    ["project future values", "projection for {query_term}", "project the trend", "forecast", "project forward", "future projection", "project the data", "extrapolate", "make a projection", "project into the future", "trend projection", "project next quarter", "forecast {query_term}", "future forecast", "project growth", "projection analysis", "project the outcome", "predict future", "project values", "forecast ahead"]
  end

  defp compute_elapsed_templates do
    ["how long has it been", "elapsed time", "time elapsed", "how much time has passed", "time since {query_term}", "duration so far", "how long since", "elapsed duration", "time running", "how long ago", "time passed", "elapsed time since {query_term}", "running time", "how long elapsed", "time since start", "duration elapsed", "how much time elapsed", "elapsed since {query_term}", "total elapsed time", "time duration so far"]
  end

  defp compute_eta_templates do
    ["estimated time of arrival", "ETA", "when will it finish", "how long until complete", "estimated completion", "time remaining", "when will it be done", "ETA for {query_term}", "estimated finish time", "how much longer", "completion estimate", "time to completion", "when will {query_term} finish", "projected finish", "estimated time left", "how long until done", "time until complete", "arrival time", "completion time estimate", "when does it end"]
  end

  defp analyze_diagnose_templates do
    ["diagnose the problem", "run diagnostics", "diagnostic analysis", "diagnose {system_name}", "find the issue", "troubleshoot", "diagnose the issue", "problem diagnosis", "identify the problem", "run diagnostic check", "diagnose the error", "diagnostic scan", "find what's wrong", "analyze the problem", "system diagnosis", "diagnostic report", "diagnose {process_name}", "troubleshoot the issue", "diagnose and fix", "identify the cause"]
  end

  defp generate_analysis_negative_examples do
    negative_examples = [
      %{text: "show me the data", correct_intent: "display.show"},
      %{text: "find the file", correct_intent: "search.locate_file"},
      %{text: "status report", correct_intent: "status.report"},
      %{text: "what time is it", correct_intent: "time.current"},
      %{text: "search for records", correct_intent: "search.find_in_database"},
      %{text: "verify the information", correct_intent: "search.verify"},
      %{text: "play the recording", correct_intent: "display.play"},
      %{text: "check system health", correct_intent: "status.health_check"},
      %{text: "zoom in", correct_intent: "display.zoom_in"},
      %{text: "count the entries", correct_intent: "search.count"}
    ]

    write_negative_examples("analysis", negative_examples)
  end

  # ============================================================================
  # Status Domain (Star Trek-inspired)
  # ============================================================================

  defp generate_status_intents do
    Mix.shell().info("Generating status intents...")

    intents = [
      {"status.report", status_report_templates()},
      {"status.summary", status_summary_templates()},
      {"status.time", status_time_templates()},
      {"status.check_system", status_check_system_templates()},
      {"status.check_process", status_check_process_templates()},
      {"status.check_complete", status_check_complete_templates()},
      {"status.health_check", status_health_check_templates()},
      {"status.run_diagnostic", status_run_diagnostic_templates()},
      {"status.what_running", status_what_running_templates()},
      {"status.active_tasks", status_active_tasks_templates()},
      {"status.progress", status_progress_templates()},
      {"status.identify_self", status_identify_self_templates()},
      {"status.version", status_version_templates()},
      {"status.capabilities", status_capabilities_templates()}
    ]

    Enum.each(intents, fn {intent_name, templates} ->
      generate_intent_file(intent_name, templates)
    end)

    generate_status_negative_examples()
    Mix.shell().info("Status intents generated!")
  end

  defp status_report_templates do
    ["status report", "give me a status report", "report status", "current status", "status please", "what's the status", "status update", "system status", "full status report", "provide status", "status check", "report on status", "give status update", "overall status", "status summary", "brief status", "status overview", "get status", "current status report", "status now"]
  end

  defp status_summary_templates do
    ["give me a summary", "summary please", "summarize", "brief summary", "overview", "quick summary", "summary of the situation", "summarize the status", "provide a summary", "high level summary", "executive summary", "status summary", "sum it up", "in summary", "give overview", "summarize for me", "brief overview", "short summary", "quick overview", "summary report"]
  end

  defp status_time_templates do
    ["what time is it", "current time", "time please", "what's the time", "tell me the time", "time now", "the time", "check the time", "give me the time", "display time", "show time", "time check", "current time please", "what time do we have", "present time", "actual time", "what's the current time", "time status", "clock check", "system time"]
  end

  defp status_check_system_templates do
    ["check system status", "system status", "how is the system", "is the system running", "check {system_name} status", "system check", "status of {system_name}", "is {system_name} working", "system health", "check the system", "system operational status", "how is {system_name}", "is the system ok", "check all systems", "system readiness", "system status check", "are systems running", "verify system status", "check system health", "system operational"]
  end

  defp status_check_process_templates do
    ["is the process running", "check {process_name} status", "is {process_name} running", "process status", "status of {process_name}", "check process", "is the {process_name} active", "process running check", "verify {process_name} is running", "check if {process_name} running", "process operational", "is {process_name} still running", "status of the process", "is {process_name} working", "check on {process_name}", "process health", "is the job running", "check running processes", "{process_name} status check", "task status"]
  end

  defp status_check_complete_templates do
    ["is it finished", "is it done", "is the task complete", "check if complete", "has it finished", "is {process_name} done", "completion status", "is it completed", "check completion", "done yet", "is it over", "has it completed", "is the process done", "finished yet", "is it ready", "completion check", "is {query_term} complete", "check if done", "is it finished yet", "task completion status"]
  end

  defp status_health_check_templates do
    ["health check", "run a health check", "system health", "check system health", "health status", "is everything healthy", "health report", "system health check", "service health", "check health", "application health", "health monitoring", "overall health", "health of the system", "run health diagnostics", "infrastructure health", "health assessment", "check service health", "wellness check", "system wellness"]
  end

  defp status_run_diagnostic_templates do
    ["run diagnostics", "diagnostic check", "run a diagnostic", "perform diagnostics", "system diagnostics", "diagnostics please", "execute diagnostics", "run system diagnostics", "diagnostic scan", "self diagnostic", "run tests", "diagnostic routine", "check diagnostics", "full diagnostic", "initiate diagnostics", "diagnostic mode", "run diagnostic tests", "diagnostics report", "perform diagnostic check", "diagnostic analysis"]
  end

  defp status_what_running_templates do
    ["what's currently running", "what's running", "show running processes", "active processes", "what is running now", "list running tasks", "currently running", "what processes are active", "show active processes", "running now", "what's active", "current processes", "what's happening now", "show what's running", "active right now", "running tasks", "what's in progress", "show running", "what's executing", "active operations"]
  end

  defp status_active_tasks_templates do
    ["show active tasks", "active tasks", "current tasks", "what tasks are active", "list active tasks", "running tasks", "tasks in progress", "show current tasks", "active jobs", "what's being worked on", "tasks currently running", "ongoing tasks", "show running tasks", "active operations", "current active tasks", "list running jobs", "tasks in execution", "show ongoing work", "active task list", "current operations"]
  end

  defp status_progress_templates do
    ["what's the progress", "progress report", "check progress", "how far along", "progress status", "progress update", "show progress", "current progress", "how is it progressing", "progress so far", "completion progress", "percentage complete", "progress percentage", "how much done", "progress check", "advancement status", "show current progress", "progress indicator", "track progress", "what percentage done"]
  end

  defp status_identify_self_templates do
    ["what are you", "who are you", "identify yourself", "what is your name", "tell me about yourself", "what kind of system are you", "your identity", "introduce yourself", "what should I call you", "what's your name", "describe yourself", "who am I talking to", "what can you tell me about yourself", "your designation", "what are you called", "system identification", "identify", "name yourself", "what system is this", "self identification"]
  end

  defp status_version_templates do
    ["what version", "version number", "current version", "which version", "version info", "what version is this", "software version", "system version", "check version", "version status", "show version", "what release", "version information", "running version", "what version are you", "version check", "app version", "display version", "get version", "version details"]
  end

  defp status_capabilities_templates do
    ["what can you do", "your capabilities", "what are you capable of", "list your functions", "what features do you have", "show capabilities", "what are your abilities", "what can I ask you", "capability list", "available functions", "what do you do", "your features", "list capabilities", "what are you able to do", "system capabilities", "functionality list", "what can you help with", "your skills", "help me understand what you do", "supported features"]
  end

  defp generate_status_negative_examples do
    negative_examples = [
      %{text: "analyze the data", correct_intent: "analyze.data"},
      %{text: "search for files", correct_intent: "search.locate_file"},
      %{text: "show me a chart", correct_intent: "display.show_chart"},
      %{text: "calculate the sum", correct_intent: "compute.sum"},
      %{text: "find patterns", correct_intent: "analyze.pattern"},
      %{text: "zoom in", correct_intent: "display.zoom_in"},
      %{text: "cross reference", correct_intent: "search.cross_reference"},
      %{text: "estimate the time", correct_intent: "compute.estimate"},
      %{text: "play the recording", correct_intent: "display.play"},
      %{text: "verify information", correct_intent: "search.verify"}
    ]

    write_negative_examples("status", negative_examples)
  end
end
