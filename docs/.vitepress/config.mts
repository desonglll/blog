import { defineConfig } from "vitepress";
import getMd from "./utils/get_md";

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Blog",
  description: "A VitePress Site",
  base: "/blog/",
  head: [
    [
      "link",
      {
        rel: "icon",
        href: "https://www.rust-lang.org/logos/rust-logo-blk.svg",
      },
    ],
  ],
  lastUpdated: true,
  markdown: {
    math: true,
    image: {
      lazyLoading: true,
    },
  },

  themeConfig: {
    returnToTopLabel: "Return to top",
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: "Home", link: "/" },
      { text: "Rust", link: "/rust/" },
      { text: "Deep Learning", link: "/deep-learning/" },
    ],

    sidebar: {
      "/rust/": getMd("./docs/rust", false),
      "/deep-learning/": [{ text: "Deep Learning", link: "/deep-learning/" }],
    },

    socialLinks: [{ icon: "github", link: "https://github.com/desonglll" }],
    search: {
      provider: "local",
    },
    editLink: {
      pattern: "https://github.com/desonglll/blog/edit/main/docs/:path",
    },
  },
});
