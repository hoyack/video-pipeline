import {
  lineNumbers,
  highlightActiveLine,
  highlightActiveLineGutter,
  keymap,
  EditorView,
} from "@codemirror/view";
import { history, historyKeymap, indentWithTab } from "@codemirror/commands";
import {
  bracketMatching,
  indentOnInput,
  defaultHighlightStyle,
  syntaxHighlighting,
} from "@codemirror/language";
import { closeBrackets, closeBracketsKeymap } from "@codemirror/autocomplete";
import { markdown, markdownLanguage } from "@codemirror/lang-markdown";
import { languages } from "@codemirror/language-data";
import { search, searchKeymap } from "@codemirror/search";
import type { Extension } from "@codemirror/state";

export function createEditorExtensions(): Extension[] {
  return [
    lineNumbers(),
    highlightActiveLine(),
    highlightActiveLineGutter(),
    history(),
    bracketMatching(),
    closeBrackets(),
    indentOnInput(),
    syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
    markdown({
      base: markdownLanguage,
      codeLanguages: languages,
    }),
    search({ top: true }),
    EditorView.lineWrapping,
    keymap.of([...searchKeymap, ...historyKeymap, ...closeBracketsKeymap, indentWithTab]),
  ];
}
