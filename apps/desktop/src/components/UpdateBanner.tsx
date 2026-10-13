import { useEffect, useState } from "react";
import { Download, RotateCw, X } from "lucide-react";

import { api, onEvent } from "../lib/ipc";
import type { UpdateInfo } from "../lib/types";
import { errorToast, useStore } from "../state/store";

const mb = (b: number) => (b >= 1e6 ? `${(b / 1e6).toFixed(b >= 1e8 ? 0 : 1)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`);

type Phase = "available" | "downloading" | "ready" | "installing";

/** "A new version is available" bar: checks on launch (if allowed), downloads only the
 *  small update package on request, then restarts into the new version. */
export function UpdateBanner() {
  const allowed = useStore((s) => s.settings?.check_updates_on_launch);
  const info = useStore((s) => s.update);
  const setInfo = (u: UpdateInfo | null) => useStore.setState({ update: u });
  const [phase, setPhase] = useState<Phase>("available");
  const [progress, setProgress] = useState<[number, number]>([0, 0]);
  const [hidden, setHidden] = useState(false);

  useEffect(() => {
    if (!allowed) return;
    let live = true;
    api.checkUpdate().then((u) => { if (live && u.newer) setInfo(u); }).catch(() => {}); // offline: stay quiet
    return () => { live = false; };
  }, [allowed]);

  useEffect(() => {
    if (info?.downloaded) setPhase("ready");
    if (info) setHidden(false);
  }, [info]);

  useEffect(() => {
    const off = onEvent((e) => { if (e.type === "update_progress") setProgress([e.done, e.total]); });
    return () => { off.then((f) => f()); };
  }, []);

  if (!info || hidden) return null;
  const download = async () => {
    setPhase("downloading");
    try { await api.updateDownload(); setPhase("ready"); } catch (e) { errorToast(e); setPhase("available"); }
  };
  const install = async () => {
    setPhase("installing");
    try { await api.updateInstall(); } catch (e) { errorToast(e); setPhase("ready"); }
  };
  const [done, total] = progress;
  return (
    <div className="update-bar" role="status" data-testid="update-banner">
      <span className="update-dot" />
      <span><b>ClipFX Enhance {info.latest}</b> is available{info.size ? ` · ${mb(info.size)}` : ""}</span>
      {phase === "downloading" && (
        <span className="update-progress"><i style={{ width: `${total ? (100 * done) / total : 0}%` }} /></span>
      )}
      <span className="grow" />
      {!info.installable && <a className="btn sm" href={info.url} target="_blank" rel="noreferrer">Download page</a>}
      {info.installable && phase === "available" && <button className="btn sm primary" onClick={download}><Download size={13} />Download</button>}
      {phase === "downloading" && <span className="faint">{total ? `${mb(done)} of ${mb(total)}` : "Starting…"}</span>}
      {phase === "ready" && <button className="btn sm primary" onClick={install}><RotateCw size={13} />Restart to update</button>}
      {phase === "installing" && <span className="faint">Installing…</span>}
      {phase !== "downloading" && phase !== "installing" && <button className="btn sm icon ghost" aria-label="Hide" onClick={() => setHidden(true)}><X size={14} /></button>}
    </div>
  );
}
