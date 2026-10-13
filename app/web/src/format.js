import { t } from "./i18n.js";

export const pct = (v, d = 1) => (v == null || Number.isNaN(v) ? "—" : `${(100 * v).toFixed(d)}%`);
export const num = (v, d = 3) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
export const steps = (n) => (n == null ? "—" : n >= 1e6 ? `${+(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${+(n / 1e3).toFixed(1)}k` : `${n}`);
export function dur(sec) {
  if (sec == null) return "—";
  sec = Math.max(0, Math.round(sec));
  const d = Math.floor(sec / 86400), h = Math.floor((sec % 86400) / 3600), m = Math.floor((sec % 3600) / 60);
  if (d) return t(`${d} 天 ${h} 小时`, `${d}d ${h}h`);
  if (h) return t(`${h} 小时 ${m} 分`, `${h}h ${m}m`);
  if (m) return t(`${m} 分`, `${m}m`);
  return t(`${sec} 秒`, `${sec}s`);
}
export function when(ts) {
  if (!ts) return "—";
  const d = new Date(ts * 1000), p = (x) => String(x).padStart(2, "0");
  const today = new Date().toDateString() === d.toDateString();
  return (today ? "" : `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} `) + `${p(d.getHours())}:${p(d.getMinutes())}`;
}
export const ago = (ts) => (ts ? t(`${dur(Date.now() / 1000 - ts)}前`, `${dur(Date.now() / 1000 - ts)} ago`) : "—");
export const bytes = (b) => (b == null ? "—" : b > 2 ** 40 ? `${(b / 2 ** 40).toFixed(1)} TB` : `${(b / 2 ** 30).toFixed(0)} GB`);
