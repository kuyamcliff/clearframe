import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { AlertTriangle, CheckCircle2, Info, X, XCircle } from "lucide-react";

import { dismissToast, useStore } from "../state/store";

export function Help({ text }: { text: string }) {
  return (
    <span className="help" tabIndex={0} role="note" aria-label={text}>
      ?<span className="tip">{text}</span>
    </span>
  );
}

export function Field({ label, help, hint, children, htmlFor }: { label: ReactNode; help?: string; hint?: ReactNode; children: ReactNode; htmlFor?: string }) {
  return (
    <div className="field">
      <label htmlFor={htmlFor}>
        {label}
        {help && <Help text={help} />}
      </label>
      {children}
      {hint && <div className="hint">{hint}</div>}
    </div>
  );
}

export function Select<T extends string>({ value, options, onChange, id, disabled, label }: {
  value: T; options: { value: T; label: string; disabled?: boolean }[]; onChange: (v: T) => void; id?: string; disabled?: boolean; label?: string;
}) {
  return (
    <select id={id} className="input" value={value} disabled={disabled} aria-label={label} onChange={(e) => onChange(e.target.value as T)}>
      {options.map((o) => (
        <option key={o.value} value={o.value} disabled={o.disabled}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

/** Number input that only commits valid values within [min, max]. */
export function NumberInput({ value, onChange, min, max, step = 1, id, label, suffix, disabled, width }: {
  value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number; id?: string; label?: string; suffix?: string; disabled?: boolean; width?: number;
}) {
  const [text, setText] = useState(String(value));
  const [focused, setFocused] = useState(false);
  useEffect(() => {
    if (!focused) setText(String(value));
  }, [value, focused]);
  const parsed = Number(text);
  const valid = text.trim() !== "" && isFinite(parsed) && (min === undefined || parsed >= min) && (max === undefined || parsed <= max);
  return (
    <span className="row" style={{ gap: 4 }}>
      <input
        id={id}
        aria-label={label}
        className={`input num${valid ? "" : " invalid"}`}
        style={width ? { width } : undefined}
        inputMode="decimal"
        value={text}
        disabled={disabled}
        aria-invalid={!valid}
        title={!valid ? `Enter a value${min !== undefined ? ` from ${min}` : ""}${max !== undefined ? ` to ${max}` : ""}` : undefined}
        onFocus={() => setFocused(true)}
        onBlur={() => {
          setFocused(false);
          if (!valid) setText(String(value));
        }}
        onChange={(e) => {
          setText(e.target.value);
          const n = Number(e.target.value);
          if (e.target.value.trim() !== "" && isFinite(n) && (min === undefined || n >= min) && (max === undefined || n <= max)) onChange(n);
        }}
        onKeyDown={(e) => {
          if (e.key === "ArrowUp" || e.key === "ArrowDown") {
            e.preventDefault();
            const n = Math.min(max ?? Infinity, Math.max(min ?? -Infinity, +(value + (e.key === "ArrowUp" ? step : -step)).toFixed(6)));
            onChange(n);
          }
        }}
      />
      {suffix && <span className="faint small">{suffix}</span>}
    </span>
  );
}

export function Slider({ value, onChange, min, max, step = 1, label, format, id, disabled }: {
  value: number; onChange: (v: number) => void; min: number; max: number; step?: number; label: string; format?: (v: number) => string; id?: string; disabled?: boolean;
}) {
  return (
    <div className="slider">
      <input id={id} type="range" min={min} max={max} step={step} value={value} disabled={disabled} aria-label={label}
        aria-valuetext={format ? format(value) : String(value)} onChange={(e) => onChange(Number(e.target.value))} />
      <span className="value">
        <NumberInput value={value} onChange={onChange} min={min} max={max} step={step} label={`${label} value`} disabled={disabled} width={64} />
      </span>
    </div>
  );
}

export function Switch({ checked, onChange, label, disabled }: { checked: boolean; onChange: (v: boolean) => void; label: string; disabled?: boolean }) {
  return <button type="button" role="switch" className="switch" aria-checked={checked} aria-label={label} disabled={disabled} onClick={() => onChange(!checked)} />;
}

export function Check({ checked, onChange, label, desc, disabled }: { checked: boolean; onChange: (v: boolean) => void; label: ReactNode; desc?: ReactNode; disabled?: boolean }) {
  const id = useId();
  return (
    <label className="check" htmlFor={id}>
      <input id={id} type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      <span>
        {label}
        {desc && <div className="desc">{desc}</div>}
      </span>
    </label>
  );
}

export function Seg<T extends string>({ value, options, onChange, label }: { value: T; options: { value: T; label: ReactNode; title?: string }[]; onChange: (v: T) => void; label: string }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} type="button" aria-pressed={o.value === value} title={o.title} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Progress({ value, tone, indeterminate, label }: { value: number; tone?: "ok" | "err"; indeterminate?: boolean; label?: string }) {
  return (
    <div className={`progress${tone ? " " + tone : ""}${indeterminate ? " indeterminate" : ""}`} role="progressbar" aria-label={label}
      aria-valuemin={0} aria-valuemax={100} aria-valuenow={indeterminate ? undefined : Math.round(value * 100)}>
      <div style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
    </div>
  );
}

export function Notice({ tone = "info", children }: { tone?: "info" | "warn" | "err" | "ok"; children: ReactNode }) {
  const Icon = tone === "err" ? XCircle : tone === "warn" ? AlertTriangle : tone === "ok" ? CheckCircle2 : Info;
  return (
    <div className={`notice ${tone}`} role={tone === "err" ? "alert" : "status"}>
      <Icon size={16} className={tone === "err" ? "err" : tone === "warn" ? "warn" : tone === "ok" ? "ok" : ""} />
      <div className="grow">{children}</div>
    </div>
  );
}

export function Modal({ title, onClose, children, footer, wide, closeDisabled }: {
  title: ReactNode; onClose: () => void; children: ReactNode; footer?: ReactNode; wide?: boolean; closeDisabled?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const titleId = useId();
  useEffect(() => {
    const prev = document.activeElement as HTMLElement | null;
    const first = ref.current?.querySelector<HTMLElement>("input, select, textarea, button:not([data-close])");
    first?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !closeDisabled) {
        e.stopPropagation();
        onClose();
      }
      if (e.key === "Tab" && ref.current) {
        const f = [...ref.current.querySelectorAll<HTMLElement>("button, input, select, textarea, [tabindex='0']")].filter((x) => !x.hasAttribute("disabled"));
        if (!f.length) return;
        const [a, b] = [f[0], f[f.length - 1]];
        if (e.shiftKey && document.activeElement === a) {
          e.preventDefault();
          b.focus();
        } else if (!e.shiftKey && document.activeElement === b) {
          e.preventDefault();
          a.focus();
        }
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => {
      window.removeEventListener("keydown", onKey, true);
      prev?.focus?.();
    };
  }, [onClose, closeDisabled]);
  return (
    <div className="backdrop" onMouseDown={(e) => e.target === e.currentTarget && !closeDisabled && onClose()}>
      <div ref={ref} className={`modal${wide ? " wide" : ""}`} role="dialog" aria-modal="true" aria-labelledby={titleId}>
        <div className="modal-head">
          <h2 id={titleId} className="grow">{title}</h2>
          <button className="btn ghost icon" data-close onClick={onClose} disabled={closeDisabled} aria-label="Close">
            <X size={18} />
          </button>
        </div>
        <div className="modal-body">{children}</div>
        {footer && <div className="modal-foot">{footer}</div>}
      </div>
    </div>
  );
}

export function Confirm({ title, body, confirmLabel, danger, onConfirm, onCancel }: {
  title: string; body: ReactNode; confirmLabel: string; danger?: boolean; onConfirm: () => void; onCancel: () => void;
}) {
  return (
    <Modal title={title} onClose={onCancel} footer={
      <>
        <button className="btn" onClick={onCancel}>Cancel</button>
        <button className={`btn ${danger ? "danger" : "primary"}`} onClick={onConfirm}>{confirmLabel}</button>
      </>
    }>
      <div className="muted">{body}</div>
    </Modal>
  );
}

export function Toasts() {
  const toasts = useStore((s) => s.toasts);
  return (
    <div className="toasts" aria-live="polite">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.level}`} role={t.level === "error" ? "alert" : "status"}>
          <div className="grow selectable">{t.message}</div>
          {t.action && (
            <button className="btn small" onClick={() => { t.action!.run(); dismissToast(t.id); }}>
              {t.action.label}
            </button>
          )}
          <button className="btn ghost icon small" aria-label="Dismiss" onClick={() => dismissToast(t.id)}>
            <X size={14} />
          </button>
        </div>
      ))}
    </div>
  );
}

/** Calls `fn` every `ms` while mounted (and once immediately). */
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
