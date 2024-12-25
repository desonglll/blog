import fs from "fs";
import { defineConfig } from "vitepress";
function getMd(dir: string, ignore_index = true) {
  const files = fs.readdirSync(dir);
  const mdFiles = files.filter((file: string) => file.endsWith(".md"));
  if (ignore_index) {
    mdFiles.splice(mdFiles.indexOf("index.md"), 1);
  }
  const result = mdFiles.map((file: string) => ({
    text: file
      .replace(".md", "")
      .replace("-", " ")
      .split(" ")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" "),
    link: `${dir.replace("./docs", "")}/${file.replace(".md", "")}`,
  }));
  // console.log(result);
  return result;
}
getMd("./docs/rust");
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
