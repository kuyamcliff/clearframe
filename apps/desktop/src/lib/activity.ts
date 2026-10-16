// Interface activity for the session log (Documents/ClipFX Enhance Log.txt): every click,
// choice and shortcut, every backend command with its duration and result, errors and
// crashes, slow frames and memory. Entries are batched and written by the Rust side; nothing
// leaves the computer. In a plain browser (UI development and tests) this does nothing.

type Level = "debug" | "info" | "warn" | "error";
type Entry = { level: Level; t: number; message: string };

const enabled = (): boolean => typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

let buffer: Entry[] = [];
let timer: number | null = null;
let sending = false;

async function send(entries: Entry[]) {
  const core = await import("@tauri-apps/api/core");
  await core.invoke("log_ui", { entries });
}

function flush() {
  timer = null;
  if (!buffer.length || sending) return;
  const entries = buffer;
  buffer = [];
  sending = true;
  send(entries)
    .catch(() => { /* the log is best effort */ })
    .finally(() => {
      sending = false;
      if (buffer.length) schedule();
    });
}

function schedule(delay = 500) {
  if (timer === null) timer = window.setTimeout(flush, delay);
}

/** Add one line to the session log. */
export function uiLog(level: Level, message: string) {
  if (!enabled()) return;
  buffer.push({ level, t: Math.round(performance.now()), message });
  if (buffer.length > 2000) buffer = buffer.slice(-2000);
  schedule(level === "error" ? 0 : 500);
}

const clip = (s: string | null | undefined, n = 80) => {
  const v = (s ?? "").replace(/\s+/g, " ").trim();
  return v.length > n ? `${v.slice(0, n)}…` : v;
};

/** A short, readable description of what was clicked. */
function describe(el: Element | null): string {
  if (!el) return "?";
  const target = el.closest("button, a, [role=button], [role=menuitem], [role=tab], [role=option], label, input, select, textarea, summary, li, [data-testid]") ?? el;
  const parts: string[] = [target.tagName.toLowerCase()];
  const label = target.getAttribute("aria-label") || target.getAttribute("title");
  const text = target instanceof HTMLInputElement || target instanceof HTMLSelectElement || target instanceof HTMLTextAreaElement ? "" : clip(target.textContent);
  if (label) parts.push(`"${clip(label)}"`);
  if (text && text !== label) parts.push(`text "${text}"`);
  if (target.id) parts.push(`#${target.id}`);
  const cls = typeof target.className === "string" ? target.className.split(" ").filter(Boolean).slice(0, 3).join(".") : "";
  if (cls) parts.push(`.${cls}`);
  if (target instanceof HTMLInputElement) parts.push(`type=${target.type}${target.name ? ` name=${target.name}` : ""}`);
  if ((target as HTMLButtonElement).disabled) parts.push("(disabled)");
  // Where in the window: nearest dialog or panel heading.
  const area = target.closest("[role=dialog], dialog, aside, header, nav, section");
  const heading = area?.querySelector("h1, h2, h3, .dialog-title, .pref-h")?.textContent;
  if (area) parts.push(`in ${area.tagName.toLowerCase()}${heading ? ` "${clip(heading, 40)}"` : ""}`);
  return parts.join(" ");
}

function valueOf(el: Element): string {
  if (el instanceof HTMLInputElement) {
    if (el.type === "checkbox" || el.type === "radio") return String(el.checked);
    if (el.type === "password") return "<hidden>";
    if (el.type === "text" || el.type === "search" || el.type === "email" || el.type === "url") return `(${el.value.length} chars typed)`;
    return el.value;
  }
  if (el instanceof HTMLSelectElement) return `"${clip(el.selectedOptions[0]?.textContent ?? el.value, 60)}"`;
  if (el instanceof HTMLTextAreaElement) return `(${el.value.length} chars)`;
  return "";
}

function errText(e: unknown): string {
  if (e instanceof Error) return `${e.name}: ${e.message}${e.stack ? `\n${e.stack}` : ""}`;
  try { return typeof e === "string" ? e : JSON.stringify(e); } catch { return String(e); }
}

let installed = false;

/** Start recording interface activity. Call once at startup. */
export function installActivityTracking() {
  if (installed || !enabled()) return;
  installed = true;

  uiLog("info", `interface loaded: ${navigator.userAgent} | screen ${screen.width}x${screen.height} @${window.devicePixelRatio}x | window ${window.innerWidth}x${window.innerHeight} | lang ${navigator.language}`);

  document.addEventListener("click", (e) => uiLog("info", `click ${describe(e.target as Element)}`), true);
  document.addEventListener("dblclick", (e) => uiLog("info", `double-click ${describe(e.target as Element)}`), true);
  document.addEventListener("contextmenu", (e) => uiLog("info", `right-click ${describe(e.target as Element)}`), true);
  document.addEventListener("change", (e) => {
    const el = e.target as Element;
    uiLog("info", `change ${describe(el)} = ${valueOf(el)}`);
  }, true);
  document.addEventListener("keydown", (e) => {
    // Shortcuts and navigation keys only; ordinary typing is not recorded.
    const special = e.ctrlKey || e.metaKey || e.altKey || e.key.length > 1;
    if (!special || ["Shift", "Control", "Alt", "Meta", "Backspace", "Tab"].includes(e.key)) return;
    const combo = [e.ctrlKey && "Ctrl", e.metaKey && "Cmd", e.altKey && "Alt", e.shiftKey && "Shift", e.key].filter(Boolean).join("+");
    uiLog("debug", `key ${combo} on ${describe(e.target as Element)}`);
  }, true);
  document.addEventListener("drop", (e) => uiLog("info", `drop ${e.dataTransfer?.files.length ?? 0} file(s) on ${describe(e.target as Element)}`), true);
  document.addEventListener("visibilitychange", () => uiLog("info", `window ${document.visibilityState}`));
  window.addEventListener("focus", () => uiLog("debug", "window focused"));
  window.addEventListener("blur", () => uiLog("debug", "window lost focus"));
  let resizeTimer: number | null = null;
  window.addEventListener("resize", () => {
    if (resizeTimer !== null) window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => uiLog("debug", `window resized to ${window.innerWidth}x${window.innerHeight}`), 500);
  });

  // Errors, crashes and media failures.
  window.addEventListener("error", (e) => {
    const t = e.target as Element | null;
    if (t && t !== (window as unknown as Element) && (t instanceof HTMLMediaElement || t instanceof HTMLImageElement)) {
      const media = t as HTMLMediaElement;
      uiLog("error", `failed to load ${t.tagName.toLowerCase()} ${clip((t as HTMLImageElement).currentSrc || (t as HTMLImageElement).src, 200)}${media.error ? ` (code ${media.error.code}: ${media.error.message})` : ""}`);
      return;
    }
    uiLog("error", `uncaught error: ${e.message} at ${e.filename}:${e.lineno}:${e.colno}${e.error?.stack ? `\n${e.error.stack}` : ""}`);
  }, true);
  window.addEventListener("unhandledrejection", (e) => uiLog("error", `unhandled promise rejection: ${errText(e.reason)}`));
  for (const level of ["error", "warn"] as const) {
    const original = console[level].bind(console);
    console[level] = (...args: unknown[]) => {
      uiLog(level, `console.${level}: ${args.map(errText).join(" ")}`);
      original(...args);
    };
  }

  // Performance: page load, frames that block the interface, memory.
  window.addEventListener("load", () => {
    const nav = performance.getEntriesByType("navigation")[0] as PerformanceNavigationTiming | undefined;
    if (nav) uiLog("info", `page load: DOM ready ${Math.round(nav.domContentLoadedEventEnd)} ms, loaded ${Math.round(nav.loadEventEnd || performance.now())} ms`);
  });
  try {
    new PerformanceObserver((list) => {
      for (const t of list.getEntries()) {
        if (t.duration >= 200) uiLog("warn", `interface blocked for ${Math.round(t.duration)} ms (long task)`);
      }
    }).observe({ type: "longtask", buffered: true });
  } catch { /* not supported by this webview */ }
  const mem = () => {
    const m = (performance as unknown as { memory?: { usedJSHeapSize: number; totalJSHeapSize: number; jsHeapSizeLimit: number } }).memory;
    if (m) uiLog("info", `interface memory: ${Math.round(m.usedJSHeapSize / 1048576)} MB used of ${Math.round(m.totalJSHeapSize / 1048576)} MB (limit ${Math.round(m.jsHeapSizeLimit / 1048576)} MB), ${document.getElementsByTagName("*").length} elements`);
  };
  window.setInterval(mem, 60000);
  window.setTimeout(mem, 5000);

  window.addEventListener("beforeunload", () => { uiLog("info", "interface closing"); flush(); });
}

/** Backend commands polled often: only logged when slow or failing. */
const QUIET = new Set(["system_sample", "jobs_list", "job_events", "media_base", "log_ui", "allow_asset"]);

function argsText(args?: Record<string, unknown>): string {
  if (!args) return "";
  try {
    const s = JSON.stringify(args, (k, v) => (k === "contents" && typeof v === "string" ? `(${v.length} chars)` : v));
    return s.length > 600 ? `${s.slice(0, 600)}…` : s;
  } catch { return "(unserialisable)"; }
}

/** Record one backend command: name, arguments, duration and outcome. */
export function logCommand(cmd: string, args: Record<string, unknown> | undefined, ms: number, error?: string) {
  if (error !== undefined) {
    uiLog("error", `command ${cmd} FAILED after ${ms} ms: ${error} | args ${argsText(args)}`);
  } else if (ms > 2000) {
    uiLog("warn", `command ${cmd} slow: ${ms} ms | args ${argsText(args)}`);
  } else if (!QUIET.has(cmd)) {
    uiLog("info", `command ${cmd} ok in ${ms} ms | args ${argsText(args)}`);
  }
}
