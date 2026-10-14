// If you want to use Phoenix channels, run `mix help phx.gen.channel`
// to get started and then uncomment the line below.
// import "./user_socket.js"

// You can include dependencies in two ways.
//
// The simplest option is to put them in assets/vendor and
// import them using relative paths:
//
//     import "../vendor/some-package.js"
//
// Alternatively, you can `npm install some-package --prefix assets` and import
// them using a path starting with the package name:
//
//     import "some-package"
//
// If you have dependencies that try to import CSS, esbuild will generate a separate `app.css` file.
// To load it, simply add a second `<link>` to your `root.html.heex` file.

// Include phoenix_html to handle method=PUT/DELETE in forms and buttons.
import "phoenix_html"
// Establish Phoenix Socket and LiveView configuration.
import {Socket} from "phoenix"
import {LiveSocket} from "phoenix_live_view"

// The browser's time zone, as Brain.Temporal.Zone.from_browser!/1 reads it:
// its name, its offset now, and every change of offset from the start of
// the year `years` before the current one to the end of the year `years`
// after it. The server holds no time zone database, so these offsets are
// how it reads a day or a clock time in this zone, on either side of a
// daylight saving change. Offsets are seconds east of UTC; instants are
// milliseconds since the epoch. The zone is the operating system's, read
// through Date, so no library is needed.
//
// The offset is sampled once a day and each change is narrowed to the
// millisecond it happens at; two changes within one day would be missed.
function browserZone(years) {
  const offsetAt = ms => Math.round(-new Date(ms).getTimezoneOffset() * 60)
  const at = Date.now()
  const year = new Date(at).getUTCFullYear()
  const from = Date.UTC(year - years, 0, 1)
  const until = Date.UTC(year + years + 1, 0, 1)
  const day = 24 * 60 * 60 * 1000
  const transitions = []
  let before = offsetAt(from)

  for (let probe = from; probe < until - 1; ) {
    const next = Math.min(probe + day, until - 1)
    const offset = offsetAt(next)

    if (offset !== before) {
      // The change lies in (probe, next]: narrow it to its first millisecond.
      let low = probe
      let high = next
      while (high - low > 1) {
        const middle = Math.floor((low + high) / 2)
        if (offsetAt(middle) === before) low = middle
        else high = middle
      }
      transitions.push([high, offset])
      before = offset
    }

    probe = next
  }

  return {
    name: Intl.DateTimeFormat().resolvedOptions().timeZone,
    at: at,
    offset: offsetAt(at),
    from: from,
    until: until,
    offset_at_from: offsetAt(from),
    transitions: transitions,
  }
}

const localTimeFormats = {
  time: {hour: "2-digit", minute: "2-digit", hourCycle: "h23"},
  datetime: {weekday: "short", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hourCycle: "h23"},
}

const Hooks = {
  AutoScrollBottom: {
    mounted() {
      this.el.scrollTop = this.el.scrollHeight
    },
    updated() {
      this.el.scrollTop = this.el.scrollHeight
    },
  },
  // On a form whose turns the server reads dates and times in: fills its
  // `time_zone` input with the browser's zone (browserZone) as the form is
  // submitted, before LiveView reads the form. `data-zone-years` names how
  // many years either side of this one the zone's offsets are given for.
  BrowserTimeZone: {
    mounted() {
      const years = Number(this.el.dataset.zoneYears)
      if (!Number.isInteger(years) || years < 1) {
        throw new Error(`BrowserTimeZone: data-zone-years must be a whole number of years, got ${this.el.dataset.zoneYears}`)
      }
      const field = this.el.querySelector("input[name='time_zone']")
      if (!field) throw new Error(`BrowserTimeZone: #${this.el.id} has no time_zone input`)

      // LiveView reads a submitted form in its own submit listener on window,
      // which runs after this one on the form itself.
      this.el.addEventListener("submit", () => {
        field.value = JSON.stringify(browserZone(years))
      })
    },
  },
  // A <time> element whose `datetime` (an ISO 8601 instant) is shown in the
  // browser's own time zone and locale, as its `data-format` says: "time"
  // (hours and minutes) or "datetime" (with the day).
  LocalTime: {
    mounted() {
      const format = localTimeFormats[this.el.dataset.format]
      if (!format) throw new Error(`LocalTime: unknown data-format ${this.el.dataset.format}`)
      const at = new Date(this.el.getAttribute("datetime"))
      if (Number.isNaN(at.getTime())) throw new Error(`LocalTime: unreadable datetime ${this.el.getAttribute("datetime")}`)
      this.el.textContent = new Intl.DateTimeFormat(undefined, format).format(at)
    },
  },
}
import topbar from "../vendor/topbar"

const csrfToken = document.querySelector("meta[name='csrf-token']").getAttribute("content")
const liveSocket = new LiveSocket("/live", Socket, {
  longPollFallbackMs: 2500,
  params: {_csrf_token: csrfToken},
  hooks: {...Hooks},
})

// Show progress bar on live navigation and form submits
topbar.config({barColors: {0: "#29d"}, shadowColor: "rgba(0, 0, 0, .3)"})
window.addEventListener("phx:page-loading-start", _info => topbar.show(300))
window.addEventListener("phx:page-loading-stop", _info => topbar.hide())

// connect if there are any LiveViews on the page
liveSocket.connect()

// expose liveSocket on window for web console debug logs and latency simulation:
// >> liveSocket.enableDebug()
// >> liveSocket.enableLatencySim(1000)  // enabled for duration of browser session
// >> liveSocket.disableLatencySim()
window.liveSocket = liveSocket

// The lines below enable quality of life phoenix_live_reload
// development features:
//
//     1. stream server logs to the browser console
//     2. click on elements to jump to their definitions in your code editor
//
if (process.env.NODE_ENV === "development") {
  window.addEventListener("phx:live_reload:attached", ({detail: reloader}) => {
    // Enable server log streaming to client.
    // Disable with reloader.disableServerLogs()
    reloader.enableServerLogs()

    // Open configured PLUG_EDITOR at file:line of the clicked element's HEEx component
    //
    //   * click with "c" key pressed to open at caller location
    //   * click with "d" key pressed to open at function component definition location
    let keyDown
    window.addEventListener("keydown", e => keyDown = e.key)
    window.addEventListener("keyup", e => keyDown = null)
    window.addEventListener("click", e => {
      if(keyDown === "c"){
        e.preventDefault()
        e.stopImmediatePropagation()
        reloader.openEditorAtCaller(e.target)
      } else if(keyDown === "d"){
        e.preventDefault()
        e.stopImmediatePropagation()
        reloader.openEditorAtDef(e.target)
      }
    }, true)

    window.liveReloader = reloader
  })
}

