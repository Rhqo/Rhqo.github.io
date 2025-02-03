import { Options } from "./quartz/components/ExplorerNode"

export const mapFn: Options["mapFn"] = (node) => {
  // if (node.depth > 0) {
  //   if (node.file) {
  //     node.displayName = "❍ " + node.displayName
  //   } 
  // }
}

// export const filterFn: Options["filterFn"] = (node) => {
//   // set containing names of everything you want to filter out
//   const omit = new Set(["title", "index", "404"])
//   return !omit.has(node.name.toLowerCase())
// }

// export const sortFn: Options["sortFn"] = (a, b) => {
//   const nameOrderMap: Record<string, number> = {
//     "poetry-folder": 100,
//     "essay-folder": 200,
//     "research-paper-file": 201,
//     "dinosaur-fossils-file": 300,
//     "other-folder": 400,
//   }

//   let orderA = 0
//   let orderB = 0

//   if (a.file && a.file.slug) {
//     orderA = nameOrderMap[a.file.slug] || 0
//   } else if (a.name) {
//     orderA = nameOrderMap[a.name] || 0
//   }

//   if (b.file && b.file.slug) {
//     orderB = nameOrderMap[b.file.slug] || 0
//   } else if (b.name) {
//     orderB = nameOrderMap[b.name] || 0
//   }

//   return orderA - orderB
// }