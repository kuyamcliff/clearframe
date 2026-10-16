// Plain-language explanations for errors. Users see what happened and what to do; the original
// text goes to the activity log (and, if they agreed, to diagnostics reports), never on screen.

export interface Friendly { title: string; hint?: string }

const RULES: { test: RegExp; title: string; hint?: string }[] = [
  { test: /reset by the driver|did not finish|successful submission|device lost|device was lost|dxgi_error_device/i,
    title: "The graphics driver stopped responding.", hint: "Retry: work is split into short pieces and falls back to the CPU. If it keeps happening, update your graphics driver or choose the CPU in Preferences → Performance." },
  { test: /out of memory|failed to allocate|insufficient memory|allocation failed|cannot allocate/i,
    title: "Ran out of memory.", hint: "Close other apps, lower the memory limit or tile size in Preferences → Performance, or choose a smaller output size." },
  { test: /missing or damaged|not installed|model .* (missing|not found)/i,
    title: "A model this needs is missing or damaged.", hint: "Open Help → Models to install or repair it." },
  { test: /no space|disk (is )?full|not enough (disk )?space|os error 112|enospc/i,
    title: "Not enough disk space.", hint: "Free some space or choose another export folder." },
  { test: /being used by another process|os error 32|sharing violation|resource busy|ebusy/i,
    title: "The file is open in another program.", hint: "Close it there (a player or editor) and try again." },
  { test: /access is denied|permission denied|os error 5\b|eacces|eperm|read-only/i,
    title: "Windows did not allow access to this file or folder.", hint: "Choose a folder you can write to (for example Videos or Documents). Controlled Folder Access in Windows Security can also block apps." },
  { test: /cannot find the (file|path)|no such file|not found.*(file|path)|os error [23]\b|enoent/i,
    title: "The file could not be found.", hint: "It may have been moved, renamed, or be on a drive that is not connected." },
  { test: /filename or extension is too long|path too long|os error 206|enametoolong/i,
    title: "The file path is too long for Windows.", hint: "Move the video to a shorter folder path or rename it." },
  { test: /invalid data found|could not find codec|unsupported codec|moov atom not found|end of file|corrupt|no video stream|does not contain any stream/i,
    title: "This video could not be read.", hint: "The file may be damaged, still downloading, or use a format FFmpeg cannot decode. Try playing it in another player." },
  { test: /hdr to sdr needs an hdr/i, title: "This video is not HDR.", hint: "HDR → SDR only applies to HDR (PQ or HLG) videos. Use Grade mode instead." },
  { test: /ffmpeg/i, title: "FFmpeg could not process this video.", hint: "Details are in the log (Help → Show log file)." },
  { test: /timed out|timeout|failed to fetch|network|dns|connection (refused|reset)|offline|could not resolve|os error 10060|os error 11001/i,
    title: "Could not connect to the internet.", hint: "Check your connection and try again. Processing works offline; only downloads need the internet." },
  { test: /signature|not signed|verification/i, title: "A download failed its safety check, so it was not used.", hint: "Try again; if it repeats, the file on the server may be damaged." },
];

export function friendlyError(raw: unknown): Friendly {
  const m = raw instanceof Error ? raw.message : typeof raw === "string" ? raw : JSON.stringify(raw);
  for (const r of RULES) if (r.test.test(m)) return { title: r.title, hint: r.hint };
  // Unknown: one clean sentence, no stack traces, codes or paths.
  const first = (m.split("\n")[0] || "Something went wrong.")
    .replace(/[A-Za-z]:\\[^\s"']+|\/(?:home|Users|tmp|var)\/[^\s"']+/g, "the file")
    .replace(/\s*\((os error|code) [^)]*\)/gi, "")
    .replace(/^(Error|Failed|failed):\s*/i, "");
  const short = first.length > 160 ? first.slice(0, 157) + "…" : first;
  return { title: short.charAt(0).toUpperCase() + short.slice(1), hint: "If this keeps happening, the details are in Help → Show log file." };
}
