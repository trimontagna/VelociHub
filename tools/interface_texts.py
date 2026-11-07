"""List every text of the interface of VelociHub (tooltips, messages, labels, buttons, menus, settings and their help),
with its file and line, in a Markdown file to review and tidy them:

    python tools/interface_texts.py [output.md]

Default output: interface_texts.md next to the repository folder. The texts are found in the source (string literals and
f-strings, their {expressions} kept), sorted by kind:
- Tooltips and help: HelpDot texts, help= arguments, constants named *_HELP / *_HINT / HINTS / *HELP dictionaries,
  the help of the settings of the engines (Parameter) and of the window Batch (Field);
- Messages: dialogs (msgbox, questdlg, messagebox...), the status line and the progress texts;
- Labels, buttons and menus: text= arguments, the titles of the sections, the items of the menus, the labels of the
  settings;
- Composed texts: the parts of the texts built in the code (descriptions, summaries, status);
- Figure texts (titles, axes, colour bars) and the credits.
Docstrings, comments, file names, keys and format strings are left out.
"""

import ast
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = (["velocihub/gui/" + f for f in sorted(os.listdir(os.path.join(REPO, "velocihub", "gui"))) if f.endswith(".py")]
         + ["velocihub/archiving/post_archive_gui.py", "velocihub/archiving/archive.py",
            "velocihub/piv/pairs_engine.py", "velocihub/piv/ptv_engine.py", "velocihub/piv/regression.py",
            "velocihub/piv/base.py", "velocihub/postprocessing.py", "velocihub/lagrangian.py", "velocihub/imaging.py",
            "velocihub/credits.py"])
LONG = 250  # Characters: flagged as long
DIALOGS = {"msgbox", "questdlg", "showerror", "showinfo", "showwarning", "askokcancel", "askyesno", "askyesnocancel",
           "askstring", "update_status", "status_error", "set_status", "message", "warn"}
LABELS = {"Label", "Button", "Checkbutton", "Radiobutton", "LabelFrame", "Menubutton", "section", "toggle",
          "add_command", "add_cascade", "add_checkbutton", "add_radiobutton", "heading", "title", "FigureSelector"}


def text_of(node):
    """The text of a string literal or an f-string ({expressions} kept), None otherwise."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out = []
        for part in node.values:
            if isinstance(part, ast.Constant):
                out.append(part.value)
            else:
                out.append("{" + ast.unparse(part.value) + "}")
        return "".join(out)
    return None


def shown(text):
    """A text meant for the user: words (a space and letters), not a key, a path, a pattern or a format."""
    t = text.strip()
    if len(t) < 3 or " " not in t or not re.search(r"[A-Za-z]{2}", t):
        return False
    if re.fullmatch(r"[\w.*/\\:{}%\-\[\] ]*\.(png|json|npz|csv|tif|tiff|bmp|gif|mp4|zip|tar|py)\b.*", t):
        return False
    if re.fullmatch(r"[\s{}\w.,:%-]*", t) and "{" in t and len(re.sub(r"\{[^}]*\}", "", t).split()) < 2:
        return False
    return True


def call_name(call):
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""


def classify(chain):
    """Kind of a text from its parents (innermost first)."""
    for i, (node, field) in enumerate(chain):
        if isinstance(node, ast.keyword):
            if node.arg in ("help", "tooltip", "hint"):
                return "help"
            if node.arg in ("text", "label", "title", "message", "prompt"):
                return "label"
        if isinstance(node, ast.Call):
            name = call_name(node)
            if name == "HelpDot":
                return "help"
            if name in DIALOGS or name.startswith(("ask", "show")):
                return "message"
            if name == "Parameter" or name == "Field":
                # Parameter(key, label, default, kind, help, ...), Field(path, label, kind, ..., help=)
                arg = chain[i - 1][0] if i else None
                args = node.args
                if arg in args:
                    index = args.index(arg)
                    return "label" if index == 1 else "help" if index >= 4 else "setting"
                return "setting"
            if name in LABELS:
                return "label"
            if name in ("emit", "progress", "wh", "say", "report", "print"):
                return "message"
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(re.search(r"HELP|HINT", n) for n in names):
                return "help"
            if any(n in ("FIGURES", "FIGURE_TITLES") or n.endswith(("LABELS", "_TITLES", "TEXTS")) for n in names):
                return "label"
            if any("SETTINGS" in n or "WIDGETS" in n for n in names):
                return "table"
            if names and names[0].isupper():
                return "other"
        if isinstance(node, ast.FunctionDef) and node.name == "menus":
            return "label"
    return "other"


def collect():
    items = []
    for rel in FILES:
        path = os.path.join(REPO, rel)
        tree = ast.parse(open(path, encoding="utf-8").read())
        docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                      if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef)) and n.body
                      and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        parents = {}
        for node in ast.walk(tree):
            for field, value in ast.iter_fields(node):
                for child in (value if isinstance(value, list) else [value]):
                    if isinstance(child, ast.AST):
                        parents[id(child)] = (node, field)
        for node in ast.walk(tree):
            if id(node) in docstrings or not isinstance(node, (ast.Constant, ast.JoinedStr)):
                continue
            if isinstance(parents.get(id(node), (None,))[0], ast.JoinedStr):
                continue  # A part of an f-string: the f-string is taken whole
            text = text_of(node)
            if text is None or not shown(text):
                continue
            chain, current = [], node
            while id(current) in parents:
                parent, field = parents[id(current)]
                chain.append((parent, field)) if not chain else None
                chain.append((parent, field))
                current = parent
            # Innermost first, each parent once
            seen, ordered = set(), []
            for parent, field in chain:
                if id(parent) not in seen:
                    seen.add(id(parent))
                    ordered.append((parent, field))
            # The direct argument of a call: the node itself, for Parameter/Field positions
            ordered = [(node, None)] + ordered
            kind = classify(ordered[1:])
            if kind == "setting":
                kind = "help"
            elif kind == "table":  # The rows of the tables of settings: label, then help
                kind = "help" if len(text) > 60 else "label"
            elif kind == "other":
                kind = ("figure" if rel in FIGURE_FILES else "credits" if rel.endswith("credits.py")
                        else "composed")
            items.append((rel, node.lineno, kind, text))
    return items


KINDS = [("help", "Tooltips and help texts"), ("message", "Messages (dialogs, status line, progress)"),
         ("label", "Labels, buttons, menus and setting names"),
         ("composed", "Composed texts (parts of the descriptions, summaries and status texts built in the code)"),
         ("figure", "Figure texts (titles, axes, colour bars, messages drawn in the figures)"),
         ("credits", "Credits window (velocihub/credits.py)")]
FIGURE_FILES = {"velocihub/postprocessing.py", "velocihub/lagrangian.py", "velocihub/imaging.py"}


def main(output):
    items = collect()
    counts = {k: sum(1 for i in items if i[2] == k) for k, _ in KINDS}
    base = os.path.dirname(os.path.abspath(output))

    def link(rel, line):
        target = os.path.relpath(os.path.join(REPO, rel), base).replace(os.sep, "/")
        return f"[{rel}:{line}]({target.replace(' ', '%20')}#L{line})"

    lines = ["# VelociHub — interface texts", "",
             f"Generated by `tools/interface_texts.py` from the source: {len(items)} texts "
             + ", ".join(f"{counts[k]} {title.split(' (')[0].lower()}" for k, title in KINDS) + ".",
             f"Texts longer than {LONG} characters are marked ⚠. `{{...}}` are values filled in at run time; "
             "`\\n` are line breaks. Each link opens the line in the source.", ""]
    longest = sorted((i for i in items if len(i[3]) > LONG), key=lambda i: -len(i[3]))
    lines += [f"## The longest texts ({len(longest)} over {LONG} characters)", "",
              "| Characters | Kind | Where | Starts with |", "|---:|---|---|---|"]
    for rel, line, kind, text in longest:
        start = text[:70].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {len(text)} | {kind} | {link(rel, line)} | {start}… |")
    lines.append("")
    for kind, title in KINDS:
        lines += [f"## {title}", ""]
        for rel in FILES:
            mine = sorted((i for i in items if i[2] == kind and i[0] == rel), key=lambda i: i[1])
            if not mine:
                continue
            lines += [f"### {rel}", ""]
            for _, line, _, text in mine:
                flag = " ⚠" if len(text) > LONG else ""
                body = text.replace("\n", "\\n")
                lines += [f"- {link(rel, line)} — {len(text)} chars{flag}", f"  > {body}", ""]
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"{len(items)} texts ({', '.join(f'{k} {v}' for k, v in counts.items())}), {len(longest)} long -> {output}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(REPO), "VelociHub_interface_texts.md"))
