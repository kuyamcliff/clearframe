// Receives ClipFX Enhance diagnostics reports and stores them in R2, one object per report:
//   reports/<yyyy-mm-dd>/<app version>/<install id>/<time>.ndjson.gz
// The body is gzip-compressed: a JSON header line, then redacted log lines. Reports are stored
// as received (still compressed); read them with `gunzip -c` or any gzip tool.
const MAX_BYTES = 8 * 1024 * 1024;
const ID = /^[0-9a-f-]{36}$/;
const VERSION = /^[0-9A-Za-z.+-]{1,32}$/;

export default {
  async fetch(request, env) {
    if (request.method !== "POST") return new Response("ClipFX diagnostics endpoint", { status: 200 });
    const install = request.headers.get("x-clipfx-install") || "";
    const version = request.headers.get("x-clipfx-version") || "unknown";
    if (!ID.test(install) || !VERSION.test(version)) return new Response("bad request", { status: 400 });
    const len = Number(request.headers.get("content-length") || 0);
    if (!len || len > MAX_BYTES) return new Response("too large", { status: 413 });
    const body = await request.arrayBuffer();
    if (body.byteLength > MAX_BYTES) return new Response("too large", { status: 413 });
    const now = new Date();
    const key = `reports/${now.toISOString().slice(0, 10)}/${version}/${install}/${now.toISOString().replace(/[:.]/g, "-")}.ndjson.gz`;
    await env.REPORTS.put(key, body, { httpMetadata: { contentType: "application/x-ndjson", contentEncoding: "gzip" } });
    return new Response(null, { status: 204 });
  },
};
