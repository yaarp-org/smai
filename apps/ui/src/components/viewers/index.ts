// Artifact-rendering primitives per 13-frontend.md §10. Stable exports that
// downstream pages (M3 read-side, M4 write-side) can swap their <pre> /
// monospace placeholders onto.
export { ArtifactFrame, type ArtifactFrameProps } from "./artifact-frame";
export { CodeViewer, type CodeViewerProps } from "./code-viewer";
export { DiffViewer, type DiffViewerProps } from "./diff-viewer";
export { DownloadLink, type DownloadLinkProps } from "./download-link";
export { JsonTree, defaultLinkResolver, type JsonTreeProps, type LinkResolver } from "./json-tree";
export { MarkdownViewer, type MarkdownViewerProps } from "./markdown-viewer";
export { detectLanguage } from "./shiki-singleton";
