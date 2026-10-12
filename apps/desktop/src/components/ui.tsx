import { useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode, type RefObject } from "react";
import { AlertTriangle, CheckCircle2, Info, X, XCircle } from "lucide-react";

import { dismissToast, useStore } from "../state/store";

/** Label on the left, control on the right. */
export function F({ label, children, stack, htmlFor, title }: { label: ReactNode; children: ReactNode; stack?: boolean; htmlFor?: string; title?: string }) {
  return (
    <div className={`f${stack ? " stack" : ""}`} title={title}>
      <label className="l" htmlFor={htmlFor}>{label}</label>
      <div style={{ minWidth: 0 }}>{children}</div>
    </div>
  );
}

export function Select<T extends string>({ value, options, onChange, id, disabled, label, className }: {
  value: T; options: { value: T; label: string; disabled?: boolean }[]; onChange: (v: T) => void; id?: string; disabled?: boolean; label?: string; className?: string;
}) {
  return (
    <select id={id} className={`input${className ? " " + className : ""}`} value={value} disabled={disabled} aria-label={label} onChange={(e) => onChange(e.target.value as T)}>
      {options.map((o) => <option key={o.value} value={o.value} disabled={o.disabled}>{o.label}</option>)}
    </select>
  );
}

/** Number field that only commits values inside [min, max]. */
export function NumberInput({ value, onChange, min, max, step = 1, id, label, suffix, disabled, className, style }: {
  value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number; id?: string; label?: string; suffix?: string; disabled?: boolean; className?: string; style?: CSSProperties;
}) {
  const [text, setText] = useState(String(value));
  const [focused, setFocused] = useState(false);
  useEffect(() => { if (!focused) setText(String(value)); }, [value, focused]);
  const ok = (t: string) => { const n = Number(t); return t.trim() !== "" && isFinite(n) && (min === undefined || n >= min) && (max === undefined || n <= max); };
  const valid = ok(text);
  return (
    <span className="row" style={{ gap: 6 }}>
      <input id={id} aria-label={label} className={`${className ?? "input num"}${valid ? "" : " invalid"}`} style={style} inputMode="decimal" value={text} disabled={disabled} aria-invalid={!valid}
        onFocus={() => setFocused(true)}
        onBlur={() => { setFocused(false); if (!valid) setText(String(value)); }}
        onChange={(e) => { setText(e.target.value); if (ok(e.target.value)) onChange(Number(e.target.value)); }}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
          if (e.key === "ArrowUp" || e.key === "ArrowDown") {
            e.preventDefault();
            onChange(Math.min(max ?? Infinity, Math.max(min ?? -Infinity, +(value + (e.key === "ArrowUp" ? step : -step)).toFixed(6))));
          }
        }} />
      {suffix && <span className="faint">{suffix}</span>}
    </span>
  );
}

/** Slider with an editable value on the right. `scale` maps stored values to displayed ones. */
export function Range({ value, onChange, min, max, step = 1, label, scale = 1, disabled }: {
  value: number; onChange: (v: number) => void; min: number; max: number; step?: number; label: string; scale?: number; disabled?: boolean;
}) {
  const p = `${((value - min) / (max - min || 1)) * 100}%`;
  const digits = step * scale >= 1 ? 0 : step * scale >= 0.1 ? 1 : 2;
  return (
    <div className="range">
      <input type="range" min={min} max={max} step={step} value={value} disabled={disabled} aria-label={label} style={{ "--p": p } as CSSProperties}
        onChange={(e) => onChange(Number(e.target.value))} />
      <NumberInput className="v" label={`${label} value`} value={+(value * scale).toFixed(digits)} min={min * scale} max={max * scale} step={step * scale} disabled={disabled}
        onChange={(v) => onChange(v / scale)} />
    </div>
  );
}

export function Switch({ checked, onChange, label, disabled }: { checked: boolean; onChange: (v: boolean) => void; label: string; disabled?: boolean }) {
  return <button type="button" role="switch" className="switch" aria-checked={checked} aria-label={label} disabled={disabled}
    onClick={(e) => { e.stopPropagation(); onChange(!checked); }} />;
}

export function Check({ checked, onChange, label, disabled }: { checked: boolean; onChange: (v: boolean) => void; label: ReactNode; disabled?: boolean }) {
  const id = useId();
  return (
    <label className="chk" htmlFor={id}>
      <input id={id} type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      <span>{label}</span>
    </label>
  );
}

export function Seg<T extends string>({ value, options, onChange, label, inline }: { value: T; options: { value: T; label: ReactNode; title?: string }[]; onChange: (v: T) => void; label: string; inline?: boolean }) {
  return (
    <div className={`seg${inline ? " inline" : ""}`} role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} type="button" aria-pressed={o.value === value} title={o.title} onClick={() => onChange(o.value)}>{o.label}</button>
      ))}
    </div>
  );
}

export function Progress({ value, tone, indeterminate, label }: { value: number; tone?: "ok" | "err"; indeterminate?: boolean; label?: string }) {
  return (
    <div className={`progress${tone ? " " + tone : ""}${indeterminate ? " indet" : ""}`} role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100}
      aria-valuenow={indeterminate ? undefined : Math.round(value * 100)}>
      <div style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
    </div>
  );
}

export function Issue({ tone, children }: { tone: "err" | "warn" | "info"; children: ReactNode }) {
  const I = tone === "err" ? XCircle : tone === "warn" ? AlertTriangle : Info;
  return <div className={`issue ${tone === "info" ? "" : tone}`} role={tone === "err" ? "alert" : undefined}><I size={13} /><span className="selectable">{children}</span></div>;
}

/** Close when clicking outside `ref` or pressing Escape. */
export function useDismiss(ref: RefObject<HTMLElement>, onClose: () => void, enabled = true) {
  useEffect(() => {
    if (!enabled) return;
    const down = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) onClose(); };
    const key = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    const t = window.setTimeout(() => document.addEventListener("mousedown", down), 0);
    document.addEventListener("keydown", key);
    return () => { window.clearTimeout(t); document.removeEventListener("mousedown", down); document.removeEventListener("keydown", key); };
  }, [ref, onClose, enabled]);
}

export interface MenuItem { label: string; onClick: () => void; danger?: boolean; disabled?: boolean; sep?: boolean }

/** Small context menu anchored to a point. */
export function Menu({ at, items, onClose }: { at: { x: number; y: number }; items: MenuItem[]; onClose: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState(at);
  useDismiss(ref, onClose);
  useLayoutEffect(() => {
    const r = ref.current?.getBoundingClientRect();
    if (r) setPos({ x: Math.min(at.x, window.innerWidth - r.width - 8), y: Math.min(at.y, window.innerHeight - r.height - 8) });
  }, [at]);
  return (
    <div className="menu" ref={ref} role="menu" style={{ left: pos.x, top: pos.y }} onClick={(e) => e.stopPropagation()} onMouseDown={(e) => e.stopPropagation()}>
      {items.map((it, i) => (
        <span key={i}>
          {it.sep && <hr />}
          <button role="menuitem" className={it.danger ? "danger" : ""} disabled={it.disabled} onClick={() => { onClose(); it.onClick(); }}>{it.label}</button>
        </span>
      ))}
    </div>
  );
}

export function useMenu() {
  const [menu, setMenu] = useState<{ at: { x: number; y: number }; items: MenuItem[] } | null>(null);
  const open = (e: { clientX: number; clientY: number; stopPropagation?: () => void; currentTarget?: EventTarget }, items: MenuItem[]) => {
    e.stopPropagation?.();
    const el = e.currentTarget as HTMLElement | undefined;
    const r = el?.getBoundingClientRect?.();
    setMenu({ at: r ? { x: r.left, y: r.bottom + 4 } : { x: e.clientX, y: e.clientY }, items });
  };
  const node = menu ? <Menu at={menu.at} items={menu.items} onClose={() => setMenu(null)} /> : null;
  return { open, node };
}

export function Dialog({ title, onClose, children, footer, size, closeDisabled, flush, headerExtra }: {
  title: ReactNode; onClose: () => void; children: ReactNode; footer?: ReactNode; size?: "wide" | "mid"; closeDisabled?: boolean; flush?: boolean; headerExtra?: ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const titleId = useId();
  useEffect(() => {
    const prev = document.activeElement as HTMLElement | null;
    ref.current?.querySelector<HTMLElement>("input:not([type=checkbox]), select, textarea")?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !closeDisabled && !document.querySelector(".menu")) { e.stopPropagation(); onClose(); }
    };
    window.addEventListener("keydown", onKey, true);
    return () => { window.removeEventListener("keydown", onKey, true); prev?.focus?.(); };
  }, [onClose, closeDisabled]);
  return (
    <div className="backdrop" onClick={(e) => e.stopPropagation()} onMouseDown={(e) => { e.stopPropagation(); if (e.target === e.currentTarget && !closeDisabled) onClose(); }}>
      <div ref={ref} className={`dialog${size ? " " + size : ""}`} role="dialog" aria-modal="true" aria-labelledby={titleId}>
        <div className="dialog-h">
          <span id={titleId} className="grow">{title}</span>
          {headerExtra}
          <button className="btn ghost icon" onClick={onClose} disabled={closeDisabled} aria-label="Close"><X size={16} /></button>
        </div>
        <div className={`dialog-b${flush ? " flush" : ""}`}>{children}</div>
        {footer && <div className="dialog-f">{footer}</div>}
      </div>
    </div>
  );
}

export function Confirm({ title, body, confirmLabel, danger, onConfirm, onCancel }: {
  title: string; body?: ReactNode; confirmLabel: string; danger?: boolean; onConfirm: () => void; onCancel: () => void;
}) {
  return (
    <Dialog title={title} onClose={onCancel} footer={<>
      <button className="btn ghost" onClick={onCancel}>Cancel</button>
      <button className={`btn ${danger ? "danger" : "primary"}`} onClick={onConfirm}>{confirmLabel}</button>
    </>}>
      {body && <div className="muted">{body}</div>}
    </Dialog>
  );
}

export function Toasts() {
  const toasts = useStore((s) => s.toasts);
  return (
    <div className="toasts" aria-live="polite">
      {toasts.map((t) => {
        const I = t.level === "success" ? CheckCircle2 : t.level === "error" ? XCircle : t.level === "warning" ? AlertTriangle : Info;
        return (
          <div key={t.id} className="toast" role={t.level === "error" ? "alert" : "status"}>
            <I size={14} className={`ic ${t.level}`} />
            <span className="selectable">{t.message}</span>
            {t.action && <button className="act" onClick={() => { t.action!.run(); dismissToast(t.id); }}>{t.action.label}</button>}
            <button className="btn ghost sm icon" aria-label="Dismiss" onClick={() => dismissToast(t.id)}><X size={13} /></button>
          </div>
        );
      })}
    </div>
  );
}

export function useInterval(fn: () => void, ms: number, enabled = true) {
  const saved = useRef(fn);
  saved.current = fn;
  useEffect(() => {
    if (!enabled) return;
    saved.current();
    const t = window.setInterval(() => saved.current(), ms);
    return () => window.clearInterval(t);
  }, [ms, enabled]);
}
