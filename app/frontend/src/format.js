export const fmt = (v, d = 4) =>
  v === null || v === undefined ? "—" : typeof v === "number" ? v.toFixed(d) : v;

export const pct = (v, d = 1) =>
  v === null || v === undefined ? "—" : (v * 100).toFixed(d) + "%";

export const pm = (mean, std, f = fmt) =>
  mean === null || mean === undefined ? "—" : std ? `${f(mean)} ± ${f(std)}` : f(mean);

export const secToHuman = (s) => {
  if (s === null || s === undefined) return "—";
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (d) return `${d}天${h}小时`;
  if (h) return `${h}小时${m}分`;
  return `${m}分`;
};

export const secToClock = (s) => {
  if (s === null || s === undefined) return "—";
  s = Math.max(0, Math.floor(s));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const pad = (n) => String(n).padStart(2, "0");
  return `${h}:${pad(m)}:${pad(s % 60)}`;
};

// unix seconds or ISO string -> "MM-DD HH:MM"
export const shortTime = (t) => {
  if (!t) return "—";
  const d = typeof t === "number" ? new Date(t * 1000) : new Date(t);
  if (isNaN(d)) return String(t);
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
};

export const steps = (n) => {
  if (!n) return "—";
  if (n >= 1e6) return (n / 1e6).toFixed(n % 1e6 ? 1 : 0) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(0) + "k";
  return String(n);
};
