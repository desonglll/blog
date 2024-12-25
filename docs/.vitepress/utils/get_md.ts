import fs from "node:fs";

interface MenuItem {
  text: string;
  link?: string;
  collapsed?: boolean;
  items?: MenuItem[];
}

export function getMd(dir: string, ignore_index = true): MenuItem[] {
  const files = fs.readdirSync(dir);
  const result: MenuItem[] = [];

  for (const file of files) {
    const fullPath = `${dir}/${file}`;
    const stats = fs.statSync(fullPath);

    if (stats.isDirectory()) {
      // Handle directory
      result.push({
        text: file
          .split("-")
          .map((word: string) => word.charAt(0).toUpperCase() + word.slice(1))
          .join(" "),
        collapsed: true,
        items: getMd(fullPath, ignore_index),
      });
    } else if (file.endsWith(".md")) {
      // Handle markdown file
      if (ignore_index && file === "index.md") continue;

      result.push({
        text: file
          .replace(".md", "")
          .replace("-", " ")
          .split(" ")
          .map((word: string) => word.charAt(0).toUpperCase() + word.slice(1))
          .join(" "),
        link: `${dir.replace("./docs", "")}/${file.replace(".md", "")}`,
      });
    }
  }

  console.log(result);
  return result;
}

export default getMd;
