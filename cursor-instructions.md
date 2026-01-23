# Cursor Instructions - Elixir/Phoenix Chat Bot

Authoritative guidance for AI code suggestions in this Elixir/Phoenix chat bot repository. The goals of this runtime:
- Adaptive, continuously learning chatbot persona using Elixir's actor model
- Real-time communication via Phoenix LiveView and WebSocket channels
- Clean separation of concerns: Brain GenServer, LiveView UI, Channel communication
- Fault-tolerant process supervision and graceful error handling
- Deterministic fallbacks and transparent logging

---

## 1. High-Level Architecture Principles

1. **Elixir Actor Model**: 
   - Primary: GenServer-based Brain for state management and conversation handling
   - Secondary: Phoenix LiveView for real-time UI updates
   - Communication: Phoenix Channels for WebSocket communication
2. **Process Supervision**:
   - `Brain` = Core GenServer managing conversations, learning, and global memory
   - `LiveView` = Real-time UI with automatic state synchronization
   - `Channels` = WebSocket communication layer for real-time events
3. **State Management**:
   - All persistent state managed by the Brain GenServer
   - LiveView state is ephemeral and synced via PubSub
   - Use Phoenix.PubSub for broadcasting updates across processes
4. **Fault Tolerance**: Leverage Elixir's "let it crash" philosophy with proper supervision trees
5. **Observability**: Use structured logging with appropriate levels and namespaces

---

## 2. ABSOLUTE RESTRICTIONS

When creating or modifying Elixir/Phoenix code:
- DO NOT use regex for semantic extraction - use pattern matching and NLP libraries
- DO NOT mutate state outside of GenServer callbacks
- DO NOT block the main process with synchronous operations
- DO NOT use `String.to_atom/1` on user input (memory leak risk)
- DO NOT nest multiple modules in the same file
- DO NOT use map access syntax on structs (`struct[:field]`) - use direct field access
- DO NOT use `if/else if` - use `cond` or `case` for multiple conditionals
- DO NOT use `live_redirect` or `live_patch` - use `push_navigate` and `push_patch`
- DO NOT use `<.flash_group>` outside of the Layouts module
- DO NOT write inline `<script>` tags in HEEx templates

---

## 3. Cursor-Specific Guidelines

### Code Generation Preferences
- **File Operations**: Always use absolute paths when possible. Prefer editing existing files over creating new ones unless explicitly required.
- **Tool Usage**: When making changes, use the appropriate tools (search_replace, MultiEdit, write) based on the scope of changes needed.
- **Parallel Operations**: Batch multiple file reads, searches, or edits when possible for efficiency.
- **Error Handling**: Always check for compilation errors after making changes and fix them if clear how to do so.

### Elixir Code Style & Structure
- **GenServer Pattern**: Use GenServer for stateful processes, LiveView for UI state
- **Function Length**: Keep functions ≤ ~60 lines; extract helpers liberally
- **Pure Functions**: Prefer pure functions for data transforms (avoid side effects except at GenServer boundaries)
- **Pattern Matching**: Use pattern matching extensively for data destructuring and control flow
- **Pipe Operator**: Use `|>` for data transformation pipelines
- **Error Messages**: Keep error messages concise and action-oriented; log all caught errors once

### Phoenix/LiveView Guidelines
- **Templates**: Always use HEEx templates with `~H` sigil, never `~E`
- **Forms**: Always use `Phoenix.Component.form/1` and `to_form/2`, never `Phoenix.HTML.form_for`
- **Components**: Use imported components from `core_components.ex` (`.input`, `.icon`, etc.)
- **Layouts**: Always begin LiveView templates with `<Layouts.app flash={@flash} ...>`
- **Streams**: Use LiveView streams for collections to avoid memory issues
- **IDs**: Always add unique DOM IDs to key elements for testing

---

## 4. Preferred Techniques

| Task | Use | Avoid |
|------|-----|-------|
| State Management | GenServer with `handle_call`, `handle_cast`, `handle_info` | Global variables or module attributes |
| Real-time Updates | Phoenix.PubSub.broadcast + LiveView handle_info | Direct state mutation |
| Form Handling | `to_form/2` + `<.form for={@form}>` | Direct changeset access in templates |
| List Operations | `Enum` functions, pattern matching | Index-based access with `[]` |
| Error Handling | `{:ok, result}` / `{:error, reason}` tuples | Exceptions for control flow |
| HTTP Requests | `Req` library | `:httpoison`, `:tesla`, `:httpc` |
| Icons | `<.icon name="hero-x-mark" />` | Heroicons modules directly |
| CSS Classes | Tailwind with conditional lists `class={[...]}` | DaisyUI components |
| File Operations | Use appropriate Cursor tools (search_replace, MultiEdit) | Manual file manipulation |

---

## 5. GenServer Guidelines

**State Management**:
- Keep state minimal and normalized
- Use `handle_call` for synchronous operations that need a response
- Use `handle_cast` for fire-and-forget operations
- Use `handle_info` for handling timeouts and PubSub messages

**Error Handling**:
- Always return `{:ok, result}` or `{:error, reason}` tuples
- Log errors at appropriate levels (debug, info, warn, error)
- Use supervision trees for automatic restart on crashes

**PubSub Integration**:
- Broadcast updates using `Phoenix.PubSub.broadcast/3`
- Subscribe to topics in LiveView `mount/3` when `connected?/1` is true
- Handle broadcasts in `handle_info/2` with proper pattern matching

---

## 6. LiveView Guidelines

**Template Structure**:
- Always wrap content with `<Layouts.app flash={@flash} ...>`
- Use HEEx syntax with proper interpolation `{...}` and `<%= ... %>`
- Add unique DOM IDs to forms and key elements
- Use conditional class lists: `class={["base-class", condition && "conditional-class"]}`

**State Management**:
- Keep LiveView state minimal - delegate to GenServer when possible
- Use `assign/3` for state updates
- Use streams for collections: `stream(socket, :items, items)`
- Handle form submissions with `phx-submit` and `handle_event/3`

**Real-time Updates**:
- Subscribe to PubSub topics in `mount/3`
- Handle broadcasts in `handle_info/2`
- Use `push_navigate/2` and `push_patch/2` for navigation

---

## 7. Channel Guidelines

**WebSocket Communication**:
- Use Phoenix Channels for real-time communication
- Implement `join/3` for connection handling
- Use `handle_in/3` for incoming messages
- Use `push/3` for sending responses
- Broadcast to rooms using `broadcast/3`

**Message Patterns**:
- Use consistent message formats with proper error handling
- Include request IDs for matching responses
- Handle connection drops gracefully

---

## 8. Testing Guidelines

**ExUnit Best Practices**:
- Use `async: true` for independent tests
- Test GenServer behavior with `GenServer.call/2` and `GenServer.cast/2`
- Test LiveView with `Phoenix.LiveViewTest` functions
- Use `element/2` and `has_element?/2` for DOM assertions

**Test Structure**:
- Test GenServer state changes and message handling
- Test LiveView mounting, events, and real-time updates
- Test Channel join, message handling, and broadcasting
- Use `start_supervised/1` for test processes

**Assertions**:
- Test structural properties, not content snapshots
- Use pattern matching for result validation
- Test error conditions and fallback behavior

---

## 9. Logging Standards

**Namespace Examples**:
- `brain`, `liveview`, `channel`, `pubsub`, `supervisor`

**Level Usage**:
- `debug`: Flow transitions, state changes, message routing
- `info`: Successful operations, connection events, learning updates
- `warn`: Fallback activations, connection issues, partial failures
- `error`: Process crashes, communication failures, validation errors

**Structured Logging**:
- Use maps for structured log data: `Logger.info("Event", %{key: value})`
- Truncate user input in logs to prevent sensitive data exposure
- Include relevant context (conversation_id, user_id, etc.)

---

## 10. Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| LOG_LEVEL | Log verbosity | info |
| PORT | Phoenix server port | 4000 |
| SECRET_KEY_BASE | Phoenix secret key | (generated) |
| DATABASE_URL | Database connection | (if using Ecto) |

---

## 11. Mix Guidelines

**Development Commands**:
- `mix compile` - Compile the project
- `mix test` - Run all tests
- `mix test test/specific_test.exs` - Run specific test file
- `mix test --failed` - Run only previously failed tests
- `mix phx.server` - Start the Phoenix server
- `mix deps.get` - Install dependencies

**Code Quality**:
- Use `mix format` for code formatting
- Use `mix credo` for static analysis (if configured)
- Use `mix dialyzer` for type checking (if configured)

---

## 12. Adding New Features

**Checklist**:
1. Define the feature in the appropriate layer (GenServer, LiveView, or Channel)
2. Add state management if needed (GenServer)
3. Add real-time updates via PubSub if needed
4. Update LiveView templates and event handlers
5. Add comprehensive tests
6. Add appropriate logging
7. Update documentation
8. Use appropriate Cursor tools for file modifications
9. Check for compilation errors and fix if clear how to do so

---

## 13. Performance & Safety

**Process Management**:
- Use supervision trees for automatic restart on crashes
- Avoid blocking operations in GenServer callbacks
- Use `Task.async_stream/3` for concurrent operations
- Implement proper timeouts for external operations

**Memory Management**:
- Use LiveView streams for large collections
- Implement proper cleanup in `terminate/2` callbacks
- Avoid unbounded data growth in GenServer state
- Use pattern matching for efficient data access

---

## 14. Anti-Patterns (Reject Suggestions That Do This)

- Using regex for semantic parsing
- Mutating state outside GenServer callbacks
- Blocking operations in LiveView event handlers
- Using `String.to_atom/1` on user input
- Nesting multiple modules in the same file
- Using map access syntax on structs
- Using `if/else if` instead of `cond` or `case`
- Writing inline scripts in HEEx templates
- Using deprecated Phoenix functions
- Creating new files when existing files can be modified
- Sequential operations that could be batched

---

## 15. Example: GOOD vs BAD (Message Handling)

**BAD**:
```elixir
# Direct state mutation and blocking operation
def handle_event("send_message", %{"message" => message}, socket) do
  # BAD: Direct state mutation
  socket.assigns.messages = socket.assigns.messages ++ [message]
  
  # BAD: Blocking HTTP call
  response = HTTPoison.post!("http://api.example.com", message)
  
  {:noreply, socket}
end
```

**GOOD**:
```elixir
# GenServer delegation and async processing
def handle_event("send_message", %{"message" => message}, socket) do
  # GOOD: Delegate to GenServer
  case ChatBot.Brain.evaluate(socket.assigns.conversation_id, message) do
    {:ok, response} ->
      # GOOD: Update via assign
      socket = assign(socket, :messages, socket.assigns.messages ++ [response])
      {:noreply, socket}
    
    {:error, reason} ->
      # GOOD: Handle error gracefully
      socket = put_flash(socket, :error, "Failed to send message: #{reason}")
      {:noreply, socket}
  end
end
```

---

## 16. Cursor-Specific Workflow Tips

- **Multi-file Changes**: Use MultiEdit when making multiple changes to the same file
- **Search & Replace**: Use search_replace for single, precise changes
- **File Creation**: Only create new files when explicitly required; prefer modifying existing files
- **Parallel Operations**: Batch file reads, searches, and edits when possible
- **Error Checking**: Always run compilation checks after making changes
- **Path Handling**: Use absolute paths when possible for better reliability

---

By following this document, Cursor AI (and contributors) should generate Elixir/Phoenix code that leverages the actor model effectively, maintains fault tolerance, and provides excellent real-time user experiences while following Phoenix and Elixir best practices.
