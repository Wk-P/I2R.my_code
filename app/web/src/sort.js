// Column sorting for tables: click a header to sort ascending, again for descending, a third time to restore the
// original order. Rows without a value (null / undefined / NaN) always go last, in either direction.
import { ref, computed } from "vue";

const empty = (v) => v == null || (typeof v === "number" && Number.isNaN(v)) || v === "";
function cmp(a, b) {
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b), undefined, { numeric: true });
}

export function useSort(rows, keys) {
  const by = ref(null);
  const dir = ref(1);
  const toggle = (k) => {
    if (by.value !== k) { by.value = k; dir.value = 1; }
    else if (dir.value === 1) dir.value = -1;
    else { by.value = null; dir.value = 1; }
  };
  const sorted = computed(() => {
    const rs = rows.value || [];
    if (!by.value || !keys[by.value]) return rs;
    const f = keys[by.value];
    return rs.map((r, i) => [r, f(r), i]).sort(([, a, i], [, b, j]) => {
      if (empty(a) || empty(b)) return empty(a) && empty(b) ? i - j : empty(a) ? 1 : -1;
      return cmp(a, b) * dir.value || i - j;
    }).map(([r]) => r);
  });
  return { by, dir, toggle, sorted };
}
