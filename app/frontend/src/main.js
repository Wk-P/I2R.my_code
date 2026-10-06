import { createApp } from "vue";
import App from "./App.vue";
import "./style.css";
import { initI18n } from "./i18n.js";

createApp(App).mount("#app");
initI18n();
