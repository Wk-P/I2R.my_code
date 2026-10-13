// Bilingual text: every string is written as t("中文", "English") right where it is used.
import { ref } from "vue";

const saved = (() => { try { return localStorage.getItem("v6.lang"); } catch { return null; } })();
export const lang = ref(saved === "en" ? "en" : "zh");
export const t = (zh, en) => (lang.value === "en" ? en ?? zh : zh);
// {zh, en} objects from the backend (manifest titles, report labels)
export const tl = (o) => (o == null ? "" : typeof o === "string" ? o : lang.value === "en" ? o.en ?? o.zh : o.zh ?? o.en);
export function setLang(l) {
  lang.value = l;
  document.documentElement.lang = l;
  try { localStorage.setItem("v6.lang", l); } catch {}
}
