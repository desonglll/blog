// .vitepress/theme/index.ts
import type { Theme } from "vitepress";
import DefaultTheme from "vitepress/theme";
import Mention from "../components/Mention.vue";
import googleAnalytics from "vitepress-plugin-google-analytics";
export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    // register your custom global components
    app.component("Mention", Mention);
    googleAnalytics({
      id: "G-KB2KLVFYQE", // Replace with your GoogleAnalytics ID, which should start with the 'G-'
    });
  },
} satisfies Theme;
