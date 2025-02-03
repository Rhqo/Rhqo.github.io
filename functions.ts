import { Options } from "./quartz/components/ExplorerNode"

export const mapFn: Options["mapFn"] = (node) => {
  if (node.depth > 0) {
    if (node.file) {
      node.displayName = "❍ " + node.displayName
    } 
  }
}

export const filterFn: Options["filterFn"] = (node) => {
  // set containing names of everything you want to filter out
  const omit = new Set(["title", "index", "404"])
  return !omit.has(node.name.toLowerCase())
}