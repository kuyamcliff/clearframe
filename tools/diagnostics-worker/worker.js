// Relay for ClipFX Enhance diagnostics reports: receives a report from the app and emails it to
// the developer through Resend, from the verified domain mail.clipfx.me.
//
// The Resend API key lives only here (a Worker secret), never in the app: an app can be
// unpacked, and a key inside it would let anyone send mail as mail.clipfx.me. The recipient is
// fixed here too, so the relay cannot be used to email anyone else.
//
// Secrets / vars (wrangler secret put …):
//   RESEND_API_KEY  sending-only key restricted to mail.clipfx.me
//   REPORT_TO       where reports go (e.g. admin@clipfx.me)
//   REPORT_FROM     optional, default "ClipFX Diagnostics <logs@mail.clipfx.me>"
//
// Body from the app: gzip of one JSON header line followed by the redacted log lines.
const MAX_BYTES = 8 * 1024 * 1024;
const ID = /^[0-9a-f-]{36}$/;
const VERSION = /^[0-9A-Za-z.+-]{1,32}$/;

function b64(bytes) {
  let s = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) s += String.fromCharCode(...bytes.subarray(i, i + chunk));
  return btoa(s);
}

async function gunzip(buf) {
  const stream = new Response(buf).body.pipeThrough(new DecompressionStream("gzip"));
  return await new Response(stream).text();
}

function esc(s) {
  return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
}

export default {
  async fetch(request, env) {
    if (request.method !== "POST") return new Response("ClipFX diagnostics relay", { status: 200 });
    const install = request.headers.get("x-clipfx-install") || "";
    const version = request.headers.get("x-clipfx-version") || "unknown";
    if (!ID.test(install) || !VERSION.test(version)) return new Response("bad request", { status: 400 });
    const len = Number(request.headers.get("content-length") || 0);
    if (!len || len > MAX_BYTES) return new Response("too large", { status: 413 });
    const body = new Uint8Array(await request.arrayBuffer());
    if (body.byteLength > MAX_BYTES) return new Response("too large", { status: 413 });

    let text;
    try {
      text = await gunzip(body);
    } catch {
      return new Response("not gzip", { status: 400 });
    }
    const nl = text.indexOf("\n");
    let header = {};
    try { header = JSON.parse(nl > 0 ? text.slice(0, nl) : text); } catch { /* keep going */ }
    const lines = nl > 0 ? text.slice(nl + 1).split("\n") : [];
    const errors = lines.filter((l) => / ERROR /.test(l));
    const warnings = lines.filter((l) => / WARN /.test(l));
    const crashed = lines.some((l) => l.includes("PANIC") || l.includes("could not run") || l.includes("could not open"));
    const firstLine = lines.find((l) => l.includes("session start"));

    const kind = crashed ? "CRASH" : errors.length ? `${errors.length} error${errors.length > 1 ? "s" : ""}` : "log";
    const subject = `[ClipFX ${version}] ${kind} · ${header.os || "?"} · ${install.slice(0, 8)}`;
    const summary = [
      `Version: ${version}`,
      `OS: ${header.os || "?"} ${header.os_version || ""} (${header.arch || "?"})`,
      `CPU: ${header.cpu || "?"}, ${header.logical_cores || "?"} threads, ${header.ram_mb || "?"} MB RAM`,
      `Install: ${install}`,
      `Log bytes ${header.log_from ?? "?"}–${header.log_to ?? "?"}${header.skipped_bytes ? ` (${header.skipped_bytes} older bytes skipped)` : ""}`,
      `Lines: ${lines.length}, errors: ${errors.length}, warnings: ${warnings.length}${crashed ? ", CRASH" : ""}`,
      firstLine ? `Session: ${firstLine}` : "",
    ].filter(Boolean);
    const html = `<pre style="font:13px/1.45 ui-monospace,Consolas,monospace">${esc(summary.join("\n"))}\n\n` +
      (errors.length ? `<b>Errors</b>\n${esc(errors.slice(-40).join("\n"))}\n\n` : "") +
      (warnings.length ? `<b>Warnings</b>\n${esc(warnings.slice(-20).join("\n"))}\n\n` : "") +
      `Full log attached.</pre>`;

    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const r = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: { Authorization: `Bearer ${env.RESEND_API_KEY}`, "Content-Type": "application/json", "Idempotency-Key": `${install}-${header.log_from ?? stamp}` },
      body: JSON.stringify({
        from: env.REPORT_FROM || "ClipFX Diagnostics <logs@mail.clipfx.me>",
        to: [env.REPORT_TO],
        subject,
        text: summary.join("\n") + (errors.length ? "\n\nErrors:\n" + errors.slice(-40).join("\n") : "") + "\n\nFull log attached.",
        html,
        attachments: [{ filename: `clipfx-${version}-${install.slice(0, 8)}-${stamp}.log.gz`, content: b64(body) }],
        tags: [{ name: "kind", value: crashed ? "crash" : errors.length ? "error" : "log" }],
      }),
    });
    // Any failure makes the app keep the data and try again later.
    if (!r.ok) return new Response(`mail failed: ${r.status}`, { status: 502 });
    return new Response(null, { status: 204 });
  },
};
