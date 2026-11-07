"""Help > Credits: the author of VelociHub, its licence and the open-source software it uses (credits.py), with their
authors, licences and links (clicked: opened in the web browser)."""

import tkinter as tk
import webbrowser
from tkinter import ttk

from .. import credits

LINK_COLOR = "#1a5fb4"


def show_credits(parent):
    """The window of the credits (one at a time: brought to the front if it is open)."""
    existing = getattr(parent, "_velocihub_credits", None)
    if existing is not None and existing.winfo_exists():
        existing.deiconify()
        existing.lift()
        existing.focus_set()
        return existing
    window = tk.Toplevel(parent)
    parent._velocihub_credits = window
    window.title("Credits - VelociHub")
    window.geometry("820x640")
    window.minsize(560, 360)

    body = ttk.Frame(window, padding=(10, 10, 10, 0))
    body.pack(fill=tk.BOTH, expand=True)
    text = tk.Text(body, wrap=tk.WORD, relief=tk.FLAT, padx=8, pady=8, cursor="arrow",
                   background=window.cget("background"), font=("Segoe UI", 10))
    scroll = ttk.Scrollbar(body, orient=tk.VERTICAL, command=text.yview)
    text.configure(yscrollcommand=scroll.set)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)
    text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    text.tag_configure("title", font=("Segoe UI", 15, "bold"), spacing3=6)
    text.tag_configure("section", font=("Segoe UI", 12, "bold"), spacing1=14, spacing3=4)
    text.tag_configure("name", font=("Segoe UI", 10, "bold"), spacing1=8)
    text.tag_configure("dim", foreground="#555555")
    text.tag_configure("indent", lmargin1=18, lmargin2=18)

    links = {}

    def link(label, url):
        tag = f"link{len(links)}"
        links[tag] = url
        text.tag_configure(tag, foreground=LINK_COLOR, underline=True)
        text.tag_bind(tag, "<Button-1>", lambda event, url=url: webbrowser.open(url))
        text.tag_bind(tag, "<Enter>", lambda event: text.configure(cursor="hand2"))
        text.tag_bind(tag, "<Leave>", lambda event: text.configure(cursor="arrow"))
        text.insert(tk.END, label, (tag, "indent"))

    text.insert(tk.END, f"VelociHub {credits.__version__}\n", "title")
    text.insert(tk.END, credits.ABOUT + "\n")
    text.insert(tk.END, f"Author: {credits.AUTHOR} ({credits.AUTHOR_EMAIL})\n", "dim")
    for i, (label, url) in enumerate(credits.ABOUT_LINKS):
        text.insert(tk.END, "  ·  " if i else "", "indent")
        link(label, url)
    text.insert(tk.END, "\n\nLicence: ", "name")
    link(credits.LICENSE, credits.LICENSE_URL)
    text.insert(tk.END, "\n" + credits.LICENSE_TEXT + "\n", "dim")

    for title, entries in credits.SECTIONS:
        text.insert(tk.END, title + "\n", "section")
        for s in entries:
            text.insert(tk.END, s.name + "\n", "name")
            text.insert(tk.END, s.use + "\n", "indent")
            text.insert(tk.END, f"Authors: {s.authors}\nLicence: {s.license}\n", ("indent", "dim"))
            if s.reference:
                text.insert(tk.END, f"Cite: {s.reference}\n", ("indent", "dim"))
            for i, (label, url) in enumerate(s.links):
                text.insert(tk.END, "  ·  " if i else "", "indent")
                link(f"{label}", url)
            text.insert(tk.END, "\n")
    text.configure(state=tk.DISABLED)

    buttons = ttk.Frame(window, padding=10)
    buttons.pack(fill=tk.X)

    def copy_bibtex():
        window.clipboard_clear()
        window.clipboard_append(credits.POD_BIBTEX)
        status["text"] = "BibTeX of the POD-based background removal copied to the clipboard"

    ttk.Button(buttons, text="Copy BibTeX (POD background removal)", command=copy_bibtex).pack(side=tk.LEFT)
    status = ttk.Label(buttons, foreground="#555555")
    status.pack(side=tk.LEFT, padx=10)
    ttk.Button(buttons, text="Close", command=window.destroy).pack(side=tk.RIGHT)
    window.bind("<Escape>", lambda event: window.destroy())
    window.links = links  # For the tests
    return window
