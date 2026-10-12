import { useEffect, useMemo, useState } from "react";
import { FolderOpen, ListPlus } from "lucide-react";

import { Check, Field, Modal, Notice, NumberInput, Select, Seg, Slider } from "../components/ui";
import { api, pickFile, pickFolder, VIDEO_EXTENSIONS } from "../lib/ipc";
import { bytes, mb, timecode } from "../lib/format";
import type { AudioMode, Container, OutputSettings, RateControl } from "../lib/types";
import { currentJob, errorToast, toast, useStore } from "../state/store";
import { PlanSummary, usePlanCheck } from "./Inspector";

const CONTAINERS: { value: Container; label: string }[] = [
  { value: "mp4", label: "MP4" }, { value: "mkv", label: "MKV" }, { value: "mov", label: "MOV" }, { value: "webm", label: "WebM" },
];

export function ExportDialog() {
  const output = useStore((s) => s.output)!;
  const codecs = useStore((s) => s.codecs);
  const range = useStore((s) => s.range);
  const folder = useStore((s) => s.exportFolder);
  const fileName = useStore((s) => s.exportFileName);
  const settings = useStore((s) => s.settings);
  const [defaultFolder, setDefaultFolder] = useState<string>("");
  const [useRange, setUseRange] = useState(!!range);
  const [busy, setBusy] = useState(false);
  const plan = usePlanCheck();
  const close = () => useStore.setState({ exportOpen: false });

  useEffect(() => {
    api.defaultExportFolder().then(setDefaultFolder).catch(() => {});
  }, []);

  const setOut = (patch: Partial<OutputSettings>) => useStore.setState({ output: { ...output, ...patch } });
  const setVideo = (patch: Partial<OutputSettings["video"]>) => setOut({ video: { ...output.video, ...patch } });

  const compatible = useMemo(() => codecs.filter((c) => c.profile.containers.includes(output.container)), [codecs, output.container]);
  const profile = codecs.find((c) => c.profile.encoder === output.video.encoder)?.profile;
  const encoderOk = compatible.some((c) => c.profile.encoder === output.video.encoder && c.available);

  const pickEncoder = (enc: string) => {
    const p = codecs.find((c) => c.profile.encoder === enc)?.profile;
    if (!p) return;
    const rate: RateControl = p.quality_range ? { mode: "quality", value: p.quality_range[1] } : p.supports_lossless ? { mode: "lossless" } : { mode: "bitrate", kbps: 20000 };
    setVideo({ encoder: enc, preset: p.default_preset, pix_fmt: p.pix_fmts.includes(output.video.pix_fmt) ? output.video.pix_fmt : p.pix_fmts[0], rate, profile: null });
  };
  const pickContainer = (c: Container) => {
    const next = { ...output, container: c };
    useStore.setState({ output: next });
    if (!codecs.some((x) => x.profile.encoder === output.video.encoder && x.profile.containers.includes(c))) {
      const first = codecs.find((x) => x.available && x.profile.containers.includes(c));
      if (first) {
        const p = first.profile;
        useStore.setState({ output: { ...next, video: { encoder: p.encoder, preset: p.default_preset, pix_fmt: p.pix_fmts[0], profile: null, gop: null, rate: p.quality_range ? { mode: "quality", value: p.quality_range[1] } : { mode: "lossless" } } } });
      }
    }
  };

  const job = () => currentJob({ range: useRange ? range : null });

  const enqueue = async () => {
    const j = job();
    if (!j) return;
    setBusy(true);
    try {
      const rec = await api.enqueue(j);
      toast("success", `Added to the queue: ${rec.name}`, { label: "View queue", run: () => useStore.setState({ view: "queue" }) });
      close();
    } catch (e) {
      errorToast(e);
    } finally {
      setBusy(false);
    }
  };

  const batch = async () => {
    const j = job();
    if (!j) return;
    const files = await pickFile("Choose more videos to process with these settings", [{ name: "Video", extensions: VIDEO_EXTENSIONS }], true);
    if (!files.length) return;
    setBusy(true);
    try {
      const r = await api.enqueueBatch([j.source, ...files.filter((f) => f !== j.source)], { ...j, file_name: null, range: null }, false);
      toast(r.failed.length ? "warning" : "success", `Queued ${r.created.length} video(s).${r.failed.length ? ` ${r.failed.length} could not be added: ${r.failed.map((f) => f.error).join("; ")}` : ""}`);
      close();
    } catch (e) {
      errorToast(e);
    } finally {
      setBusy(false);
    }
  };

  const check = plan.check;
  const blocked = !check || check.plan.issues.some((i) => i.severity === "error") || !!check.destination_folder_ok?.Err || !encoderOk;
  const needMb = check?.plan.estimates.output_size_mb[1] ?? 0;
  const lowSpace = check?.free_bytes !== null && check?.free_bytes !== undefined && check.free_bytes < needMb * 1e6;
  const audio = output.audio;
  const rate = output.video.rate;

  return (
    <Modal wide title="Export" onClose={close} footer={
      <>
        <button className="btn" onClick={batch} disabled={busy || blocked}><ListPlus size={15} /> Add more files (batch)…</button>
        <span className="spacer" />
        <button className="btn" onClick={close}>Cancel</button>
        <button className="btn primary" onClick={enqueue} disabled={busy || blocked} data-testid="confirm-export">Add to queue</button>
      </>
    }>
      <div className="grid2">
        <div>
          <h3 style={{ marginBottom: 10 }}>Destination</h3>
          <Field label="Folder" hint={!folder ? (settings?.output_dir ? "Default export folder from Settings." : "Default: next to your Videos folder.") : undefined}>
            <div className="row">
              <span className="path grow" title={folder ?? settings?.output_dir ?? defaultFolder}>{folder ?? settings?.output_dir ?? defaultFolder}</span>
              <button className="btn small" onClick={async () => {
                const f = await pickFolder("Choose export folder", folder ?? defaultFolder);
                if (f) useStore.setState({ exportFolder: f });
              }}><FolderOpen size={14} /> Choose…</button>
            </div>
          </Field>
          <Field label="File name" htmlFor="exp-name" hint={`Leave empty to use the naming template “${settings?.naming_template ?? "{name}_{task}"}”. Existing files are never overwritten: a number is added instead.`}>
            <input id="exp-name" className="input" placeholder="automatic" value={fileName ?? ""} onChange={(e) => useStore.setState({ exportFileName: e.target.value.trim() ? e.target.value : null })} />
          </Field>
          {range && (
            <Check checked={useRange} onChange={setUseRange} label={`Only export the marked range (${timecode(range.start)} – ${timecode(range.end)})`} />
          )}
          {check && (
            <div className="card" style={{ padding: 12 }}>
              <div className="small faint">Will be saved as</div>
              <div className="path" data-testid="destination">{check.destination}</div>
              {check.destination_folder_ok?.Err && <Notice tone="err">{check.destination_folder_ok.Err}</Notice>}
              <div className="small faint mt">Free space: {bytes(check.free_bytes)} · estimated output {mb(check.plan.estimates.output_size_mb[0])} – {mb(check.plan.estimates.output_size_mb[1])}, temporary work files up to {mb(check.plan.estimates.work_disk_mb[1])}</div>
              {lowSpace && <Notice tone="warn">The destination may not have enough free space.</Notice>}
            </div>
          )}
          <div className="mt"><PlanSummary {...plan} /></div>
        </div>
        <div>
          <h3 style={{ marginBottom: 10 }}>Format</h3>
          <Field label="Container">
            <Seg label="Container" value={output.container} onChange={pickContainer} options={CONTAINERS} />
          </Field>
          <Field label="Video codec" htmlFor="exp-enc">
            <Select id="exp-enc" value={output.video.encoder} onChange={pickEncoder} options={[
              ...compatible.map((c) => ({ value: c.profile.encoder, label: `${c.profile.label}${c.available ? "" : " — not available in your FFmpeg"}`, disabled: !c.available })),
              ...(compatible.some((c) => c.profile.encoder === output.video.encoder) ? [] : [{ value: output.video.encoder, label: `${output.video.encoder} — not usable in ${output.container.toUpperCase()}`, disabled: true }]),
            ]} />
            {profile && <div className="hint">{profile.description}</div>}
          </Field>
          {!encoderOk && <Notice tone="err">Choose a codec that your FFmpeg supports for {output.container.toUpperCase()}.</Notice>}
          {profile && (
            <>
              <Field label="Rate control">
                <Seg label="Rate control" value={rate.mode} onChange={(m) => {
                  const r: RateControl = m === "quality" ? { mode: "quality", value: profile.quality_range?.[1] ?? 20 } : m === "bitrate" ? { mode: "bitrate", kbps: 20000 } : { mode: "lossless" };
                  setVideo({ rate: r });
                }} options={[
                  ...(profile.quality_range ? [{ value: "quality" as const, label: "Quality" }] : []),
                  ...(profile.supports_bitrate ? [{ value: "bitrate" as const, label: "Bitrate" }] : []),
                  ...(profile.supports_lossless ? [{ value: "lossless" as const, label: "Lossless" }] : []),
                ]} />
              </Field>
              {rate.mode === "quality" && profile.quality_range && (
                <Field label="Quality" help="Lower numbers mean higher quality and larger files (CRF/CQ).">
                  <Slider label="Quality" value={rate.value} min={profile.quality_range[0]} max={profile.quality_range[2]} step={1} onChange={(v) => setVideo({ rate: { mode: "quality", value: v } })} />
                </Field>
              )}
              {rate.mode === "bitrate" && (
                <Field label="Bitrate">
                  <NumberInput label="Bitrate" value={rate.kbps} min={100} max={1000000} step={500} suffix="kbit/s" width={110} onChange={(v) => setVideo({ rate: { mode: "bitrate", kbps: Math.round(v) } })} />
                </Field>
              )}
              {profile.presets.length > 0 && (
                <Field label="Encoder speed" htmlFor="exp-preset" help="Slower presets compress better at the same quality.">
                  <Select id="exp-preset" value={output.video.preset ?? profile.default_preset ?? profile.presets[0]} onChange={(v) => setVideo({ preset: v })} options={profile.presets.map((p) => ({ value: p, label: p }))} />
                </Field>
              )}
              <Field label="Pixel format" htmlFor="exp-pix" help="10-bit formats (…10le) reduce banding and are required for HDR output.">
                <Select id="exp-pix" value={output.video.pix_fmt} onChange={(v) => setVideo({ pix_fmt: v })} options={profile.pix_fmts.map((p) => ({ value: p, label: p }))} />
              </Field>
            </>
          )}
          <Field label="Audio" htmlFor="exp-audio">
            <Select id="exp-audio" value={audio.mode} onChange={(m) => {
              const a: AudioMode = m === "encode" ? { mode: "encode", codec: output.container === "webm" ? "libopus" : "aac", bitrate_kbps: 192 } : m === "none" ? { mode: "none" } : { mode: "auto" };
              setOut({ audio: a });
            }} options={[{ value: "auto", label: "Keep original (copy when possible)" }, { value: "encode", label: "Re-encode" }, { value: "none", label: "No audio" }]} />
          </Field>
          {audio.mode === "encode" && (
            <div className="row wrap">
              <Field label="Codec" htmlFor="exp-acodec">
                <Select id="exp-acodec" value={audio.codec} onChange={(v) => setOut({ audio: { ...audio, codec: v } })} options={[
                  { value: "aac", label: "AAC" }, { value: "libopus", label: "Opus" }, { value: "flac", label: "FLAC (lossless)" }, { value: "pcm_s16le", label: "PCM 16-bit" }, { value: "ac3", label: "AC-3" },
                ]} />
              </Field>
              <Field label="Bitrate"><NumberInput label="Audio bitrate" value={audio.bitrate_kbps} min={32} max={1536} step={32} suffix="kbit/s" onChange={(v) => setOut({ audio: { ...audio, bitrate_kbps: Math.round(v) } })} /></Field>
            </div>
          )}
          <Check checked={output.subtitles} onChange={(v) => setOut({ subtitles: v })} label="Keep subtitles" desc="Text subtitles are converted when the container needs it; unsupported ones are reported." />
          <Check checked={output.metadata} onChange={(v) => setOut({ metadata: v })} label="Keep metadata (title, dates, rotation)" />
          <Check checked={output.chapters} onChange={(v) => setOut({ chapters: v })} label="Keep chapters" />
        </div>
      </div>
    </Modal>
  );
}
