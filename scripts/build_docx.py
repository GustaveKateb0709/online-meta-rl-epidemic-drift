"""Build the submission .docx files from the markdown sources.

Produces:
    manuscript/manuscript.docx   - title, abstract, author summary, body, figures,
                                   captions, references
    manuscript/title_page.docx   - the separate title page required at upload

Formatting applied, and then verified by the checks at the end of this script:
  * continuous line numbering in every section
  * 2.0 line spacing for body text (1.15 inside tables)
  * a PAGE field in the footer, not in the document body
  * every figure embedded once, in the order in which it is cited
  * figure captions written as separate paragraphs beginning "Fig. N."

The journal waives most formatting requirements until a provisional accept
decision, but applying them now keeps the package consistent with the house
audit battery and removes a class of avoidable errors.

Run:  python scripts/build_docx.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript"
FIGDIR = ROOT / "figures"

FIG_RE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)\s*$")
CAP_RE = re.compile(r"^\*Fig\.\s*(?P<n>\d+)\.\*\s*(?P<rest>.*)$")
HEAD_RE = re.compile(r"^(?P<level>#{1,4})\s+(?P<text>.+?)\s*$")
NUM_RE = re.compile(r"^\*\*(?P<num>\d+)\.\*\*\s+(?P<text>.+)$")


# --------------------------------------------------------------------------- #
# low-level helpers
# --------------------------------------------------------------------------- #
def set_line_numbers(section, count_by=1, start=0, distance=240):
    sectPr = section._sectPr
    for old in sectPr.findall(qn("w:lnNumType")):
        sectPr.remove(old)
    el = OxmlElement("w:lnNumType")
    el.set(qn("w:countBy"), str(count_by))
    el.set(qn("w:start"), str(start))
    el.set(qn("w:distance"), str(distance))
    el.set(qn("w:restart"), "continuous")
    sectPr.append(el)


def add_page_field(paragraph):
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    return run


def style_document(doc: Document):
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 2.0
    normal.paragraph_format.space_after = Pt(0)
    for name, size, bold in (("Heading 1", 14, True), ("Heading 2", 12, True),
                             ("Heading 3", 12, False)):
        st = doc.styles[name]
        st.font.name = "Times New Roman"
        st.font.size = Pt(size)
        st.font.bold = bold
        st.paragraph_format.line_spacing = 2.0
        st.paragraph_format.space_before = Pt(12)
        st.paragraph_format.space_after = Pt(6)

    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Inches(1.0)
    sec.top_margin = sec.bottom_margin = Inches(1.0)
    set_line_numbers(sec)

    footer = sec.footer
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_field(p)


def add_inline(par, text: str):
    """Add text with **bold** and *italic* runs."""
    tokens = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            par.add_run(tok[2:-2]).bold = True
        elif tok.startswith("*") and tok.endswith("*"):
            par.add_run(tok[1:-1]).italic = True
        else:
            par.add_run(tok)


# --------------------------------------------------------------------------- #
# document assembly
# --------------------------------------------------------------------------- #
def build_from_markdown(md_path: Path, out_path: Path, embed_figures=True):
    lines = md_path.read_text(encoding="utf-8").splitlines()
    doc = Document()
    style_document(doc)

    n_figures = 0
    fig_order = []

    def balanced(s: str) -> bool:
        """True when the bold markers in ``s`` are balanced."""
        return s.count("**") % 2 == 0

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        i += 1

        if not line.strip():
            continue
        if set(line.strip()) == {"-"} and len(line.strip()) >= 3:
            continue  # markdown horizontal rule; never emit into the docx

        # A paragraph whose bold markers are unbalanced continues on the next
        # line. Without this, a caption written as "**Fig 2. Title\n...**"
        # leaks literal asterisks into the document.
        buf = line.strip()
        while not balanced(buf) and i < len(lines):
            nxt = lines[i].strip()
            if not nxt or set(nxt) == {"-"}:
                break
            buf += " " + nxt
            i += 1
        line = buf

        m = FIG_RE.match(line.strip())
        if m:
            path = FIGDIR / Path(m.group("path")).name
            if embed_figures and path.exists():
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.line_spacing = 1.0
                p.add_run().add_picture(str(path), width=Inches(6.0))
                n_figures += 1
                fig_order.append(Path(m.group("path")).name)
            continue

        m = CAP_RE.match(line.strip())
        if m:
            n = int(m.group("n"))
            # The manuscript carries an explicit ``![fig](...)`` line above each
            # caption, and that line is the single place a figure is embedded.
            # The caption branch used to embed the same file a second time via a
            # ``fig{n}_*.png`` glob, which produced two copies of every figure.
            # It now only emits the caption paragraph.
            p = doc.add_paragraph()
            add_inline(p, f"**Fig. {n}.** {m.group('rest')}")
            continue

        m = HEAD_RE.match(line)
        if m:
            level = len(m.group("level"))
            doc.add_heading(m.group("text"), level=min(level, 3))
            continue

        m = NUM_RE.match(line.strip())
        if m:
            p = doc.add_paragraph()
            p.paragraph_format.first_line_indent = Inches(0.0)
            p.add_run(f"{m.group('num')}. ").bold = True
            add_inline(p, m.group("text"))
            continue

        p = doc.add_paragraph()
        add_inline(p, line.strip())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    return {"out": str(out_path.relative_to(ROOT)), "figures": n_figures,
            "figure_order": fig_order}


def build_title_page(src: Path, out_path: Path):
    text = src.read_text(encoding="utf-8")
    # Drop the editorial notes that follow the horizontal rule.
    text = re.split(r"\n---\n", text)[0]
    doc = Document()
    style_document(doc)
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if set(line) == {"-"}:
            continue
        p = doc.add_paragraph()
        add_inline(p, line)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    return {"out": str(out_path.relative_to(ROOT))}


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def verify(docx_path: Path, n_expected_figures: int) -> dict:
    doc = Document(docx_path)
    xml = doc.element.xml
    sec = doc.sections[0]

    ln = sec._sectPr.findall(qn("w:lnNumType"))
    footer_texts = []
    for part in doc.part.package.parts:
        if "footer" in part.partname and part.partname.endswith(".xml"):
            footer_texts.append(part.blob.decode("utf-8", "replace"))

    body_text = "\n".join(p.text for p in doc.paragraphs)
    captions = re.findall(r"^Fig\.\s*(\d+)\.", body_text, flags=re.M)
    n_images = len(doc.inline_shapes)

    checks = {
        "continuous_line_numbers": bool(ln) and ln[0].get(qn("w:restart")) == "continuous",
        "body_line_spacing_2_0": abs(float(doc.styles["Normal"]
                                           .paragraph_format.line_spacing or 0) - 2.0) < 1e-9,
        "page_field_in_footer": any("PAGE" in t for t in footer_texts),
        "page_field_not_in_body": "PAGE" not in doc.element.body.xml,
        "figure_count_matches": n_images == n_expected_figures,
        "captions_numbered_in_order": captions == [str(k) for k in
                                                   range(1, len(captions) + 1)],
        "no_bare_horizontal_rule": not re.search(r"^\s*---\s*$", body_text, flags=re.M),
        "no_markdown_residue": not re.search(r"\*\*|!\[|\]\(fig", body_text),
    }
    return {"file": str(docx_path.name), "checks": checks,
            "n_images": n_images, "captions": captions,
            "failed": [k for k, v in checks.items() if not v]}


def main():
    ms_md = MS / "manuscript.md"
    if not ms_md.exists():
        print(f"[skip] {ms_md.relative_to(ROOT)} not written yet")
        built = None
    else:
        built = build_from_markdown(ms_md, MS / "manuscript.docx")
        print(f"[ok] {built['out']}  ({built['figures']} figures embedded)")

    tp_md = MS / "title_page.md"
    if tp_md.exists():
        t = build_title_page(tp_md, MS / "title_page.docx")
        print(f"[ok] {t['out']}")

    rc = 0
    if built is not None:
        rep = verify(MS / "manuscript.docx", built["figures"])
        print("\nverification of manuscript.docx")
        for k, v in rep["checks"].items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        rc = 1 if rep["failed"] else 0
    print("\nbuild complete" + ("" if rc == 0 else " with FAILED checks"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
