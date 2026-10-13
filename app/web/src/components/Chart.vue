<script setup>
// ECharts wrapper (light theme, like the classic panel): animates only the first render; later updates (polling)
// replace the data in place without animation, so a refresh does not flash. Follows the container width.
import { ref, onMounted, onBeforeUnmount, watch } from "vue";
import * as echarts from "echarts/core";
import { LineChart, ScatterChart, BarChart } from "echarts/charts";
import { GridComponent, TooltipComponent, LegendComponent, MarkLineComponent, MarkPointComponent, MarkAreaComponent,
  TitleComponent, DataZoomComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

echarts.use([LineChart, ScatterChart, BarChart, GridComponent, TooltipComponent, LegendComponent, MarkLineComponent,
  MarkPointComponent, MarkAreaComponent, TitleComponent, DataZoomComponent, CanvasRenderer]);

const props = defineProps({ option: { type: Object, required: true }, height: { type: [Number, String], default: 280 } });
const el = ref(null);
let chart = null, ro = null;
let rendered = false;

function base() {
  const fg = "#5b6478", line = "#e6e8f0";
  return {
    backgroundColor: "transparent",
    textStyle: { fontFamily: "-apple-system, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif", color: fg },
    _axis: { axisLine: { lineStyle: { color: line } }, splitLine: { lineStyle: { color: line } }, axisLabel: { color: fg }, nameTextStyle: { color: fg } },
  };
}
function themed(opt) {
  const b = base();
  const ax = (a) => (a == null ? a : (Array.isArray(a) ? a : [a]).map((x) => ({ ...b._axis, ...x,
    axisLine: { ...b._axis.axisLine, ...(x.axisLine || {}) }, splitLine: { ...b._axis.splitLine, ...(x.splitLine || {}) },
    axisLabel: { ...b._axis.axisLabel, ...(x.axisLabel || {}) } })));
  return { backgroundColor: b.backgroundColor, textStyle: b.textStyle, animation: !rendered, animationDuration: 300, ...opt,
           xAxis: ax(opt.xAxis), yAxis: ax(opt.yAxis) };
}
function render() {
  if (!chart) return;
  chart.setOption(themed(props.option), { notMerge: true, lazyUpdate: true });
  rendered = true;
}
onMounted(() => {
  chart = echarts.init(el.value, null, { renderer: "canvas" });
  render();
  ro = new ResizeObserver(() => chart?.resize());
  ro.observe(el.value);
});
onBeforeUnmount(() => { ro?.disconnect(); chart?.dispose(); chart = null; });
watch(() => props.option, render, { deep: false });
</script>

<template>
  <div ref="el" :style="{ width: '100%', height: typeof height === 'number' ? height + 'px' : height }" />
</template>
