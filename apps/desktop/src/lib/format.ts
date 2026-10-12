import type { Rational } from "./types";

export function bytes(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined || !isFinite(n)) return "—";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = Math.abs(n);
  while (v >= 1000 && i < u.length - 1) {
    v /= 1000;
    i++;
  }
  return `${n < 0 ? "-" : ""}${i === 0 ? v.toFixed(0) : v.toFixed(digits)} ${u[i]}`;
}

export function mb(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  return n >= 1024 ? `${(n / 1024).toFixed(1)} GB` : `${Math.round(n)} MB`;
}

/** 83.4 -> "1:23.4", 3725 -> "1:02:05" */
export function timecode(sec: number, precise = true): string {
  if (!isFinite(sec) || sec < 0) sec = 0;
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  const ss = precise ? s.toFixed(2).padStart(5, "0") : Math.floor(s).toString().padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}

export function duration(sec: number): string {
  if (!isFinite(sec)) return "—";
  if (sec < 60) return `${Math.max(1, Math.round(sec))} s`;
  if (sec < 3600) return `${Math.floor(sec / 60)} min ${Math.round(sec % 60)} s`;
  return `${Math.floor(sec / 3600)} h ${Math.round((sec % 3600) / 60)} min`;
}

export function eta(range: [number, number] | null | undefined): string {
  if (!range) return "estimating…";
  const [a, b] = range;
  if (b - a < Math.max(10, a * 0.15)) return `about ${duration((a + b) / 2)} left`;
  return `${duration(a)} – ${duration(b)} left`;
}

export function fps(r: Rational | null | undefined): string {
  if (!r || !r.den) return "—";
  const v = r.num / r.den;
  return `${Number.isInteger(v) ? v : v.toFixed(3).replace(/0+$/, "")} fps`;
}

export function pct(f: number, digits = 0): string {
  return `${(Math.max(0, Math.min(1, f)) * 100).toFixed(digits)}%`;
}

export function basename(p: string): string {
  return p.split(/[\\/]/).filter(Boolean).pop() ?? p;
}

export function dirname(p: string): string {
  const i = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
  return i > 0 ? p.slice(0, i) : p;
}

export function stem(p: string): string {
  return basename(p).replace(/\.[^.]+$/, "");
}

export function relTime(iso: string | null | undefined): string {
  if (!iso) return "";
  const t = Date.parse(iso);
  if (!isFinite(t)) return iso;
  const d = (Date.now() - t) / 1000;
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)} min ago`;
  if (d < 86400) return `${Math.floor(d / 3600)} h ago`;
  return new Date(t).toLocaleDateString();
}
