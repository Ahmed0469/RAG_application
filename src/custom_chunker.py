#!/usr/bin/env python3
"""
Structure-aware chunker for ISO/IEC 27001:2022 (PDF -> JSONL), built for RAG.

Output: one JSON object per line with
    id, text (clean source text), embed_text (breadcrumb + text), metadata

Chunking strategy
  * Clauses 4-10 (+ Scope, Normative refs, Terms): one chunk per lowest-level
    sub-clause that has a body (e.g. 4.1, 6.1.2, 9.3.2). Lead-in sentence,
    a)/b)/c) items and NOTEs stay together.
  * Annex A: one chunk per control (93 controls), parsed by word coordinates
    because the table's text layer is scrambled (cells interleave).
  * Dropped: cover, copyright page, TOC, foreword, bibliography, ICS page,
    headers/footers, watermark strings.

Usage:
    pip install pdfplumber
    python ingest_iso27001.py iso27001.pdf -o iso27001_chunks.jsonl
    python ingest_iso27001.py iso27001.pdf --include-intro
"""
import argparse
import json
import re
import sys
from collections import Counter

import pdfplumber

STANDARD = "ISO/IEC 27001:2022"
ID_PREFIX = "iso27001-2022"

# ---- page layout of this specific PDF (1-based physical pages) --------------
INTRO_PAGES = [5]
BODY_PAGES = range(7, 17)      # clauses 1-10
ANNEX_PAGES = range(17, 25)    # Annex A (page 17 holds the intro paragraph)
ROMAN = {4: "iv", 5: "v"}


def printed_page(phys: int):
    """Physical PDF page -> printed page label (roman for front matter)."""
    return ROMAN.get(phys, phys - 6 if phys >= 7 else None)


# ---- expected structure (from the TOC + 6.3, which the TOC omits) -----------
EXPECTED_HEADINGS = {
    "1": "Scope", "2": "Normative references", "3": "Terms and definitions",
    "4": "Context of the organization",
    "4.1": "Understanding the organization and its context",
    "4.2": "Understanding the needs and expectations of interested parties",
    "4.3": "Determining the scope of the information security management system",
    "4.4": "Information security management system",
    "5": "Leadership", "5.1": "Leadership and commitment", "5.2": "Policy",
    "5.3": "Organizational roles, responsibilities and authorities",
    "6": "Planning", "6.1": "Actions to address risks and opportunities",
    "6.1.1": "General", "6.1.2": "Information security risk assessment",
    "6.1.3": "Information security risk treatment",
    "6.2": "Information security objectives and planning to achieve them",
    "6.3": "Planning of changes",
    "7": "Support", "7.1": "Resources", "7.2": "Competence", "7.3": "Awareness",
    "7.4": "Communication", "7.5": "Documented information",
    "7.5.1": "General", "7.5.2": "Creating and updating",
    "7.5.3": "Control of documented information",
    "8": "Operation", "8.1": "Operational planning and control",
    "8.2": "Information security risk assessment",
    "8.3": "Information security risk treatment",
    "9": "Performance evaluation",
    "9.1": "Monitoring, measurement, analysis and evaluation",
    "9.2": "Internal audit", "9.2.1": "General", "9.2.2": "Internal audit programme",
    "9.3": "Management review", "9.3.1": "General",
    "9.3.2": "Management review inputs", "9.3.3": "Management review results",
    "10": "Improvement", "10.1": "Continual improvement",
    "10.2": "Nonconformity and corrective action",
}
INTRO_HEADINGS = {"0": "Introduction", "0.1": "General",
                  "0.2": "Compatibility with other management system standards"}

EXPECTED_LEAF_CHUNKS = 33  # leaf sub-clauses with a body in clauses 1-10

THEMES = {"5": "Organizational", "6": "People", "7": "Physical", "8": "Technological"}
EXPECTED_CONTROL_COUNTS = {"5": 37, "6": 8, "7": 14, "8": 34}

# ---- noise ------------------------------------------------------------------
WATERMARK = re.compile(r"--[`,\-]{5,}")
ZW = re.compile("[\u200b\u00ad]")
HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})\s+(\S.*)$")
ITEM_RE = re.compile(r"^([a-z]\)|\d{1,2}\)|—)\s*")
NOTE_RE = re.compile(r"^NOTE(\s+\d+)?\b")
ID_RE = re.compile(r"^[5-8](\.\d{1,2})?$")

KEEP_HYPHEN_PREFIXES = {"re", "non", "co", "pre", "multi", "anti"}

KEYWORDS = [
    "Statement of Applicability", "risk treatment plan", "residual risk",
    "risk owner", "risk acceptance criteria", "risk assessment", "risk treatment",
    "internal audit", "management review", "nonconformity", "corrective action",
    "continual improvement", "interested parties", "information security policy",
    "information security objectives", "documented information", "competence",
    "awareness", "scope", "Annex A", "top management", "monitoring",
    "change management", "supplier", "cryptography", "access control", "incident",
]


# ---- text helpers -----------------------------------------------------------
def clean_line(s: str) -> str:
    s = ZW.sub("", s)
    s = WATERMARK.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def is_noise(line: str) -> bool:
    if not line:
        return True
    if re.fullmatch(r"\d{1,2}|[ivx]+", line):          # bare page number
        return True
    if "© ISO/IEC 2022" in line:
        return True
    if line.startswith("ISO/IEC 27001:2022(E)"):
        return True
    if line.startswith("INTERNATIONAL STANDARD"):
        return True
    return False


def build_vocab(line_groups):
    """Words seen un-broken in the doc; used to decide hyphen-at-EOL joins."""
    vocab = Counter()
    for lines in line_groups:
        for ln in lines:
            toks = ln.split()
            for t in toks[:-1] if ln.endswith("-") else toks:
                vocab[t.strip(".,;:()").lower()] += 1
    return vocab


def join_wrapped(prev: str, nxt: str, vocab) -> str:
    """Join two physical lines, repairing end-of-line hyphenation."""
    if prev.endswith("-") and nxt[:1].islower():
        a = prev[:-1].split(" ")[-1]
        b = nxt.split(" ")[0].strip(".,;:()")
        joined, hyph = (a + b).lower(), (a + "-" + b).lower()
        if vocab.get(hyph):
            return prev + nxt                                   # keep hyphen
        if vocab.get(joined):
            return prev[:-1] + nxt                              # soft hyphen
        if a.lower() in KEEP_HYPHEN_PREFIXES:
            return prev + nxt                                   # re-assignment
        return prev[:-1] + nxt
    return prev + " " + nxt


def reflow(lines, vocab) -> str:
    """Wrapped lines -> paragraphs; list items and NOTEs start new lines."""
    out = []
    for ln in lines:
        starts_new = (
            not out
            or ITEM_RE.match(ln)
            or NOTE_RE.match(ln)
            or (out[-1][-1:] in ".:;" and ln[:1].isupper())
        )
        if starts_new:
            out.append(ln)
        else:
            out[-1] = join_wrapped(out[-1], ln, vocab)
    return "\n".join(out)


def keywords_in(text: str):
    low = text.lower()
    return [k for k in KEYWORDS if k.lower() in low]


def external_refs(text: str):
    refs = set(re.findall(r"ISO(?:/IEC)? \d{4,5}", text))
    if "Annex SL" in text:
        refs.add("Annex SL")
    return sorted(refs)


# ---- clauses ----------------------------------------------------------------
def extract_clause_lines(pdf, pages):
    """[(clean_line, physical_page)] with headers/footers/watermarks removed."""
    rows = []
    for p in pages:
        text = pdf.pages[p - 1].extract_text() or ""
        for raw in text.split("\n"):
            ln = clean_line(raw)
            if not is_noise(ln):
                rows.append((ln, p))
    return rows


def _nospace(t: str) -> str:
    return re.sub(r"\s+", "", t).lower()


def segment(rows, headings):
    """Split lines into sections keyed by heading id.

    Headings are compared with whitespace removed because the PDF's drop-cap
    styling yields lines like '4.3 D etermining the scope ...'.
    """
    sections, cur = [], None
    for ln, page in rows:
        m = HEADING_RE.match(ln)
        if m and m.group(1) in headings and _nospace(m.group(2)) == _nospace(headings[m.group(1)]):
            cur = {"id": m.group(1), "title": headings[m.group(1)], "lines": [], "pages": {page}}
            sections.append(cur)
        elif cur is not None:
            cur["lines"].append(ln)
            cur["pages"].add(page)
    return sections


def clause_chunks(sections, headings, vocab, valid_ids):
    chunks = []
    for s in sections:
        if not s["lines"]:
            continue                    # pure parent heading (e.g. 6.1, 9.2)
        cid = s["id"]
        body = reflow(s["lines"], vocab)
        body = re.sub(r"\s*\[\d\]", "", body)               # drop [5]-style cites
        body = re.sub(r"[ \t]+([,.;:])", r"\1", body)
        crumbs, p = [], cid
        while p:
            crumbs.insert(0, f"{p} {headings[p]}")
            p = p.rsplit(".", 1)[0] if "." in p else None
        parent = cid.rsplit(".", 1)[0] if "." in cid else None
        pages = sorted(s["pages"])
        clause_refs = sorted(
            {r for r in re.findall(r"\b(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\b", body)
             if r in valid_ids and r != cid},
            key=lambda x: [int(n) for n in x.split(".")])
        if re.search(r"\bAnnex A\b", body):
            clause_refs.append("Annex A")
        is_intro = cid.startswith("0")
        meta = {
            "standard": STANDARD, "edition": 3,
            "type": "introduction" if is_intro else "requirement",
            "clause": cid, "title": s["title"], "parent": parent,
            "breadcrumb": crumbs,
            "page_pdf": pages[0], "page_pdf_end": pages[-1],
            "page": printed_page(pages[0]), "page_end": printed_page(pages[-1]),
            "references": clause_refs, "external_references": external_refs(body),
            "has_shall": bool(re.search(r"\bshall\b", body)),
            "mentions_documented_info": "documented information" in body.lower(),
            "has_note": bool(re.search(r"^NOTE", body, re.M)),
            "keywords": keywords_in(body),
        }
        embed = f"{STANDARD} > " + " > ".join(crumbs) + "\n" + body
        chunks.append({"id": f"{ID_PREFIX}-{cid}", "text": body,
                       "embed_text": embed, "metadata": meta})
    return chunks


# ---- Annex A ----------------------------------------------------------------
def lines_from_words(words, tol=3.0):
    """Group word dicts into visual lines, left-to-right."""
    words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines, cur, cur_top = [], [], None
    for w in words:
        if cur_top is None or abs(w["top"] - cur_top) <= tol:
            cur.append(w)
            cur_top = w["top"] if cur_top is None else cur_top
        else:
            lines.append(cur)
            cur, cur_top = [w], w["top"]
    if cur:
        lines.append(cur)
    return [" ".join(x["text"] for x in sorted(l, key=lambda w: w["x0"])) for l in lines]


def annex_intro(pdf, vocab):
    text = pdf.pages[ANNEX_PAGES[0] - 1].extract_text() or ""
    lines = [clean_line(l) for l in text.split("\n")]
    lines = [l for l in lines if l and not is_noise(l)]
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith("The information security controls"))
        end = next(i for i, l in enumerate(lines) if l.startswith("Table A.1"))
    except StopIteration:
        return None
    lines = [l for l in lines if not re.fullmatch(r"\[\d\]", l)]
    end = next(i for i, l in enumerate(lines) if l.startswith("Table A.1"))
    start = next(i for i, l in enumerate(lines) if l.startswith("The information security controls"))
    body = reflow(lines[start:end], vocab)
    body = re.sub(r"\s+([,.;:])", r"\1", re.sub(r"\[\d\]", "", body))
    crumbs = ["Annex A (normative) Information security controls reference"]
    return {
        "id": f"{ID_PREFIX}-A", "text": body,
        "embed_text": f"{STANDARD} > {crumbs[0]}\n{body}",
        "metadata": {
            "standard": STANDARD, "edition": 3, "type": "annex_intro",
            "clause": "Annex A", "title": "Information security controls reference",
            "breadcrumb": crumbs, "page_pdf": ANNEX_PAGES[0],
            "page": printed_page(ANNEX_PAGES[0]), "linked_clause": "6.1.3",
            "references": ["6.1.3"], "external_references": ["ISO/IEC 27002"],
        },
    }


def annex_chunks(pdf, vocab):
    chunks, theme_no = [], None
    for phys in ANNEX_PAGES:
        page = pdf.pages[phys - 1]
        words = []
        for w in page.extract_words(x_tolerance=1.5, y_tolerance=2):
            t = ZW.sub("", WATERMARK.sub("", w["text"])).strip()
            if t and "©" not in t and w["top"] < 800:
                w = dict(w, text=t)
                words.append(w)

        label_x = [w["x0"] for w in words if w["text"] == "Control" and w["x0"] > 150]
        id_cands = [w for w in words if ID_RE.match(w["text"]) and w["x0"] < 120]
        if not label_x or not id_cands:
            continue
        desc_x = min(label_x) - 2
        id_x = min(w["x0"] for w in id_cands)
        anchors = sorted((w for w in id_cands if abs(w["x0"] - id_x) <= 6),
                         key=lambda w: w["top"])

        for i, a in enumerate(anchors):
            top = a["top"] - 3
            bottom = anchors[i + 1]["top"] - 3 if i + 1 < len(anchors) else 10_000
            row = [w for w in words if top <= w["top"] < bottom and w is not a]
            title_w = [w for w in row if id_x + 30 <= w["x0"] < desc_x]
            desc_w = [w for w in row if w["x0"] >= desc_x]

            if "." not in a["text"]:                    # section header row
                theme_no = a["text"]
                continue

            labels = [w for w in desc_w if w["text"] == "Control" and abs(w["x0"] - (desc_x + 2)) < 3]
            if labels:                                  # drop the bold "Control" label
                desc_w.remove(min(labels, key=lambda w: w["top"]))

            title = reflow(lines_from_words(title_w), vocab).replace("\n", " ")
            text = reflow(lines_from_words(desc_w), vocab)
            ctrl = a["text"]
            chunks.append(build_control(ctrl, title, text, theme_no, phys))
    return chunks


def build_control(ctrl, title, text, theme_no, phys):
    theme = THEMES[theme_no]
    cid = f"A.{ctrl}"
    crumbs = ["Annex A", f"{theme_no} {theme} controls", f"{cid} {title}"]
    return {
        "id": f"{ID_PREFIX}-{cid}",
        "text": text,
        "embed_text": f"{STANDARD} > " + " > ".join(crumbs) + f"\nControl: {text}",
        "metadata": {
            "standard": STANDARD, "edition": 3, "type": "control",
            "control_id": cid, "iso27002_ref": ctrl, "title": title,
            "theme": theme, "theme_no": theme_no, "annex_ref": "Table A.1",
            "linked_clause": "6.1.3", "breadcrumb": crumbs,
            "page_pdf": phys, "page": printed_page(phys),
            "keywords": keywords_in(title + " " + text),
        },
    }


# ---- validation -------------------------------------------------------------
def validate(clauses, controls):
    problems = []

    leaf = [c for c in clauses if c["metadata"]["type"] == "requirement"]
    if len(leaf) != EXPECTED_LEAF_CHUNKS:
        problems.append(f"expected {EXPECTED_LEAF_CHUNKS} clause chunks, got {len(leaf)}")
    for must in ("6.3", "6.1.3", "9.3.2", "10.2"):
        if not any(c["metadata"]["clause"] == must for c in leaf):
            problems.append(f"missing clause {must}")

    ids = [c["metadata"]["iso27002_ref"] for c in controls]
    if len(ids) != 93:
        problems.append(f"expected 93 controls, got {len(ids)}")
    if len(set(ids)) != len(ids):
        problems.append("duplicate control ids: " + str([k for k, v in Counter(ids).items() if v > 1]))
    for theme_no, n in EXPECTED_CONTROL_COUNTS.items():
        want = {f"{theme_no}.{i}" for i in range(1, n + 1)}
        got = {i for i in ids if i.startswith(theme_no + ".")}
        if want != got:
            problems.append(f"theme {theme_no}: missing {sorted(want - got)}, unexpected {sorted(got - want)}")
    for c in controls:
        if len(c["text"].split()) < 4:
            problems.append(f"{c['metadata']['control_id']}: description suspiciously short: {c['text']!r}")
        if not c["metadata"]["title"]:
            problems.append(f"{c['metadata']['control_id']}: empty title")
        if re.search(r"\bControl\b", c["text"]):
            problems.append(f"{c['metadata']['control_id']}: stray 'Control' label in text")
    return problems


def count_tokens(text: str) -> int:
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


# ---- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("-o", "--out", default="iso27001_chunks.jsonl")
    ap.add_argument("--include-intro", action="store_true", help="also chunk Introduction 0.1/0.2")
    args = ap.parse_args()

    with pdfplumber.open(args.pdf) as pdf:
        if len(pdf.pages) != 26:
            print(f"WARNING: expected 26 pages, found {len(pdf.pages)}; "
                  "page ranges at the top of this script may need adjusting", file=sys.stderr)

        headings = dict(EXPECTED_HEADINGS)
        pages = list(BODY_PAGES)
        if args.include_intro:
            headings.update(INTRO_HEADINGS)
            pages = INTRO_PAGES + pages

        rows = extract_clause_lines(pdf, pages)
        if args.include_intro:                        # page 5 has a bare "Introduction" title
            rows = [(("0 Introduction" if ln == "Introduction" else ln), p) for ln, p in rows]

        annex_lines = [[clean_line(l) for l in (pdf.pages[p - 1].extract_text() or "").split("\n")]
                       for p in ANNEX_PAGES]
        vocab = build_vocab([[l for l, _ in rows]] + annex_lines)

        sections = segment(rows, headings)
        clauses = clause_chunks(sections, headings, vocab, set(headings))
        intro = annex_intro(pdf, vocab)
        controls = annex_chunks(pdf, vocab)

    all_chunks = clauses + ([intro] if intro else []) + controls

    for c in all_chunks:
        c["metadata"]["tokens"] = count_tokens(c["embed_text"])
        c["metadata"]["chunk_id"] = c["id"]

    with open(args.out, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    toks = [c["metadata"]["tokens"] for c in all_chunks]
    print(f"Wrote {len(all_chunks)} chunks -> {args.out}")
    print(f"  clause chunks : {len(clauses)}")
    print(f"  annex intro   : {1 if intro else 0}")
    print(f"  Annex A ctrls : {len(controls)}")
    print(f"  tokens        : min {min(toks)} / median {sorted(toks)[len(toks)//2]} / max {max(toks)}")

    problems = validate(clauses, controls)
    if problems:
        print("\nVALIDATION PROBLEMS:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("Validation passed (33 clause chunks, 93 controls: 37/8/14/34).")


if __name__ == "__main__":
    main()
