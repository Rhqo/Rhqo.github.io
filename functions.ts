import { Options } from "./quartz/components/ExplorerNode"

export const mapFn: Options["mapFn"] = (node) => {
  if (node.depth > 0) {
    // set emoji for file/folder
    if (node.file) {
      node.displayName = "📄 " + node.displayName
    } else {
      node.displayName = "📁 " + node.displayName
    }
  }
}

export const filterFn: Options["filterFn"] = (node) => {
  // set containing names of everything you want to filter out
  const omit = new Set(["authoring content", "tags", "hosting"])
  return !omit.has(node.name.toLowerCase())
}