import fs from "fs";

export function getMd(dir: string, ignore_index = true) {
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

export default getMd;
