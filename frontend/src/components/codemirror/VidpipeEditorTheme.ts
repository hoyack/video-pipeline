import { EditorView } from "@codemirror/view";
import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { tags } from "@lezer/highlight";
import type { Extension } from "@codemirror/state";

const vidpipeThemeBase = EditorView.theme(
  {
    "&": {
      backgroundColor: "#030712",
      color: "#d1d5db",
      fontSize: "14px",
      fontFamily:
        "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace",
    },
    ".cm-content": {
      caretColor: "#818cf8",
      padding: "8px 0",
      lineHeight: "1.625",
    },
    ".cm-cursor, .cm-dropCursor": {
      borderLeftColor: "#818cf8",
    },
    "&.cm-focused .cm-selectionBackground, .cm-selectionBackground": {
      backgroundColor: "rgba(79, 70, 229, 0.3)",
    },
    ".cm-activeLine": {
      backgroundColor: "rgba(17, 24, 39, 0.5)",
    },
    ".cm-gutters": {
      backgroundColor: "#111827cc",
      color: "#4b5563",
      borderRight: "1px solid #1f2937",
      minWidth: "3ch",
    },
    ".cm-activeLineGutter": {
      backgroundColor: "rgba(17, 24, 39, 0.8)",
      color: "#9ca3af",
    },
    ".cm-foldPlaceholder": {
      backgroundColor: "#1f2937",
      border: "none",
      color: "#6b7280",
    },
    // Search panel
    ".cm-panels": {
      backgroundColor: "#111827",
      color: "#d1d5db",
    },
    ".cm-panels.cm-panels-top": {
      borderBottom: "1px solid #1f2937",
    },
    ".cm-searchMatch": {
      backgroundColor: "rgba(234, 179, 8, 0.3)",
      outline: "1px solid rgba(234, 179, 8, 0.5)",
    },
    ".cm-searchMatch.cm-searchMatch-selected": {
      backgroundColor: "rgba(234, 179, 8, 0.5)",
    },
    ".cm-panel input[type=text]": {
      backgroundColor: "#030712",
      color: "#d1d5db",
      border: "1px solid #374151",
      borderRadius: "4px",
      padding: "2px 6px",
      fontSize: "12px",
    },
    ".cm-panel input[type=text]:focus": {
      borderColor: "#6366f1",
      outline: "none",
    },
    ".cm-panel button": {
      backgroundColor: "#1f2937",
      color: "#d1d5db",
      border: "1px solid #374151",
      borderRadius: "4px",
      padding: "2px 8px",
      cursor: "pointer",
      fontSize: "12px",
    },
    ".cm-panel button:hover": {
      backgroundColor: "#374151",
    },
    ".cm-panel label": {
      color: "#9ca3af",
      fontSize: "12px",
    },
    ".cm-tooltip": {
      backgroundColor: "#1f2937",
      border: "1px solid #374151",
      color: "#d1d5db",
    },
    ".cm-tag-tooltip": {
      padding: "4px 8px",
      fontSize: "12px",
      lineHeight: "1.4",
      maxWidth: "300px",
    },
    ".cm-tag-tooltip strong": {
      color: "#93c5fd",
    },
  },
  { dark: true },
);

const vidpipeHighlightStyle = HighlightStyle.define([
  { tag: tags.heading, color: "#e5e7eb", fontWeight: "bold" },
  {
    tag: tags.heading1,
    color: "#e5e7eb",
    fontWeight: "bold",
    fontSize: "1.2em",
  },
  {
    tag: tags.heading2,
    color: "#e5e7eb",
    fontWeight: "bold",
    fontSize: "1.1em",
  },
  { tag: tags.emphasis, color: "#c4b5fd", fontStyle: "italic" },
  { tag: tags.strong, color: "#e5e7eb", fontWeight: "bold" },
  { tag: tags.keyword, color: "#c084fc" },
  { tag: tags.atom, color: "#67e8f9" },
  { tag: tags.bool, color: "#67e8f9" },
  { tag: tags.url, color: "#818cf8", textDecoration: "underline" },
  { tag: tags.link, color: "#818cf8" },
  { tag: tags.labelName, color: "#818cf8" },
  { tag: tags.inserted, color: "#4ade80" },
  { tag: tags.deleted, color: "#f87171" },
  { tag: tags.literal, color: "#a5b4fc" },
  { tag: tags.string, color: "#86efac" },
  { tag: tags.number, color: "#fbbf24" },
  { tag: tags.variableName, color: "#93c5fd" },
  {
    tag: [tags.processingInstruction, tags.comment],
    color: "#6b7280",
    fontStyle: "italic",
  },
  { tag: tags.monospace, color: "#a78bfa" },
  {
    tag: tags.strikethrough,
    textDecoration: "line-through",
    color: "#9ca3af",
  },
  { tag: tags.quote, color: "#9ca3af", fontStyle: "italic" },
  { tag: tags.meta, color: "#6b7280" },
  { tag: tags.contentSeparator, color: "#4b5563" },
]);

export const vidpipeEditorTheme: Extension = [
  vidpipeThemeBase,
  syntaxHighlighting(vidpipeHighlightStyle),
];
