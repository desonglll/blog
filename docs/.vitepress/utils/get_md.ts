import fs from "node:fs";
import { capitalCase } from "change-case";
import trimEnd from "lodash/trimEnd";
interface MenuItem {
  text: string;
  link?: string;
  collapsed?: boolean;
  items?: MenuItem[];
}
function nameProcess(str: string): string {
  let s = trimEnd(str, ".md");
  s = s.replace(/^\d{4}-\d{2}-\d{2}-/, "");
  s = s.replace("-", " ");
  s = capitalCase(s);
  return s;
}
export function getMd(dir: string, ignore_index = true): MenuItem[] {
  const files: [string] = fs.readdirSync(dir);
  const result: MenuItem[] = [];

  for (const file of files) {
    const fullPath = `${dir}/${file}`;
    const stats = fs.statSync(fullPath);

    if (stats.isDirectory()) {
      // Handle directory
      result.push({
        text: nameProcess(file),
        collapsed: true,
        items: getMd(fullPath, ignore_index),
      });
    } else if (file.endsWith(".md")) {
      // Handle markdown file
      if (ignore_index && file === "index.md") continue;

      result.push({
        text: nameProcess(file),
        link: `${dir.replace("./docs", "")}/${file.replace(".md", "")}`,
      });
    }
  }

  console.log(JSON.stringify(result, null, 2));
  return result;
}

export default getMd;
