import { defineConfig } from "vitepress";

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

  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: "Home", link: "/" },
      { text: "Rust", link: "/rust/" },
      { text: "Deep Learning", link: "/deep-learning/" },
    ],

    sidebar: {
      "/rust/": [{ text: "Rust", link: "/rust/" }],
      "/deep-learning/": [{ text: "Deep Learning", link: "/deep-learning/" }],
    },

    socialLinks: [
      { icon: "github", link: "https://github.com/vuejs/vitepress" },
    ],
    search: {
      provider: "local",
    },
    editLink: {
      pattern: "https://github.com/desonglll/blog/edit/main/docs/:path",
    },
  },
});
