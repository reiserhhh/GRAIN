#!/usr/bin/env python3
"""Extract, locate and deduplicate every citation in the five frozen GPT Pro sources."""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

URL = re.compile(r"https?://[^\s)\]>\"'，。]+")
DOI = re.compile(r"\b(10\.\d{4,9}/[^\s?#&\"'<>，。)]+)", re.IGNORECASE)
ARXIV = re.compile(
    r"arxiv\.org/(?:abs|pdf)/([a-z\-]*/?\d{4}\.?\d{4,5}(?:v\d+)?)", re.IGNORECASE
)
NUMBERED = re.compile(r"^\s*(\d{1,3})\.\s+(\S.*)$")
PAGE = re.compile(r"^## PDF page (\d+)")
SECTION = re.compile(r"^## ([一二三四五六七八])、")
GENERIC_HOSTS = {
    "semanticscholar.org",
    "link.springer.com",
    "methods.sagepub.com",
    "sciencedirect.com",
    "osf.io",
    "researchgate.net",
    "academic.oup.com",
    "consensus.app",
}
TOPICS = (
    (
        "psr_bisimulation",
        r"predictive (state )?represent|\bpsr\b|bisimulation|observable operator|spectral learning",
    ),
    (
        "state_space_observability",
        r"kalman|observab|state[- ]space|system identification|linear dynamical",
    ),
    (
        "gtheory_mtmm_invariance",
        r"generaliz(a)?bility theory|multitrait|multimethod|\bmtmm\b|measurement invariance|differential item|\bdif\b|reliability of behavioral",
    ),
    (
        "selection_partial_identification",
        r"manski|partial identification|missing data|selection bias|heckman|nonignorable|not at random|inverse probability|propensity",
    ),
    (
        "adaptive_design_cat",
        r"adaptive design|adaptive language-based|\balba\b|computeri[sz]ed adaptive|optimal experimental design|bayesian (optimal )?(experimental )?design|item response|\birt\b",
    ),
    ("pragmatics_rsa", r"rational speech|pragmatic|speech act|implicature|politeness"),
    (
        "nonlinear_ica_identifiability",
        r"nonlinear ica|independent component|ivae|identifiab|disentangl|contrastive learning",
    ),
    (
        "information_bottleneck_blackwell",
        r"information bottleneck|blackwell|comparison of experiments|garbling|sufficien",
    ),
    (
        "mori_zwanzig_memory",
        r"mori|zwanzig|memory kernel|optimal prediction|coarse[- ]grain",
    ),
    (
        "counterfactual_explanation",
        r"counterfactual expla|explanations? without opening|actionable recourse|attribution",
    ),
    (
        "validity_theory",
        r"construct validity|concept of validity|validating the interpretations|standards for educational|nomological|validity",
    ),
    (
        "reader_nli_llm",
        r"natural language inference|entailment|\bmnli\b|\bsnli\b|large language model|\bllm\b|chatbot|gpt|zero-shot",
    ),
    (
        "interpretability_sae",
        r"sparse autoencoder|dictionary learning|monosemantic|concept bottleneck|superposition|probing|mechanistic",
    ),
    (
        "text_personality",
        r"personality|big five|language use|open-vocabulary|word use|liwc|social media",
    ),
    (
        "dynamics_ema_state",
        r"ecological momentary|experience sampling|dynamic|time series|within-person|intensive longitudinal|reaction norm|state",
    ),
    (
        "representation_geometry",
        r"word representations|embedding|vector space|geometry|jacobian|fisher",
    ),
)
LOAD_BEARING = {
    "psr_bisimulation",
    "state_space_observability",
    "gtheory_mtmm_invariance",
    "selection_partial_identification",
    "adaptive_design_cat",
    "pragmatics_rsa",
    "nonlinear_ica_identifiability",
    "information_bottleneck_blackwell",
    "mori_zwanzig_memory",
    "counterfactual_explanation",
    "validity_theory",
}
TOOL = {"reader_nli_llm", "interpretability_sae", "representation_geometry"}


def clean_url(url):
    """Strip tracking parameters and trailing punctuation; keep scheme/host/path/query."""
    url = url.rstrip(".,;:")
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query) if not k.startswith("utm_")]
    return urlunsplit(
        (parts.scheme, parts.netloc.lower(), parts.path, urlencode(query), "")
    )


def host(url):
    """Host without www."""
    netloc = urlsplit(url).netloc.lower()
    return netloc.removeprefix("www.")


def is_generic(url):
    """A locator that does not identify a specific work."""
    parts = urlsplit(url)
    path = parts.path.strip("/")
    return (
        host(url) in GENERIC_HOSTS
        and (not path or path in ("rest", "search"))
        or not path
    )


def norm_title(text):
    """Normalized comparison key for titles."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(
        r"\s*[-|]\s*(sciencedirect|pubmed|pmc|springer.*|frontiers|wiley.*|plos.*)$",
        "",
        text,
    )
    return re.sub(r"[^a-z0-9一-鿿]+", "", text)[:90]


def lines_with_locator(text, source):
    """Yield (line_number, locator, line); PDF page for P01-P04, section for P05."""
    page = None
    out = []
    for number, line in enumerate(text.splitlines(), start=1):
        m = PAGE.match(line)
        if m:
            page = int(m.group(1))
        s = SECTION.match(line)
        if s:
            page = "S" + str("一二三四五六七八".index(s.group(1)) + 1)
        out.append(
            (number, f"{source}:{page}" if page is not None else f"{source}:0", line)
        )
    return out


def join_wrapped_url(rows, index):
    """Rejoin a URL broken across a line wrap."""
    _, _, line = rows[index]
    match = URL.search(line)
    url = match.group(0)
    nxt = index + 1
    while (
        nxt < len(rows)
        and line.rstrip().endswith(url)
        and re.fullmatch(r"[A-Za-z0-9\-_/?=&%.#:~+]+", rows[nxt][2].strip() or " ")
        and not rows[nxt][2].strip().startswith("http")
    ):
        url += rows[nxt][2].strip()
        line = rows[nxt][2]
        nxt += 1
    return url


def extract(text, source):
    """Extract URL-anchored and numbered author-year citation records."""
    rows = lines_with_locator(text, source)
    records = []
    for i, (number, locator, line) in enumerate(rows):
        for match in URL.finditer(line):
            url = (
                join_wrapped_url(rows, i)
                if match.start() == URL.search(line).start()
                else match.group(0)
            )
            context = []
            j = i - 1
            before = line[: match.start()].strip()
            if before:
                context.append(before)
            while j >= 0 and len(context) < 3:
                prev = rows[j][2].strip()
                if not prev or URL.search(prev) or PAGE.match(prev):
                    break
                context.insert(0, prev)
                j -= 1
            title = re.sub(r"^\s*(核\s*)?(\d+\s+)+", "", " ".join(context)).strip()
            window = " ".join(r[2] for r in rows[max(0, i - 3) : i + 3])
            records.append(
                {
                    "source": source,
                    "locator": locator,
                    "line": number,
                    "kind": "url",
                    "title": title[:400],
                    "url": clean_url(url),
                    "core_mark": bool(re.search(r"(^|\s)核(\s|$)", window)),
                }
            )
    i = 0
    while i < len(rows):
        m = NUMBERED.match(rows[i][2])
        if m and re.search(r"\(\d{4}\)|\d{4}\.|et al\.|&", m.group(2)):
            parts = [m.group(2).strip()]
            k = i + 1
            while (
                k < len(rows)
                and rows[k][2].strip()
                and not NUMBERED.match(rows[k][2])
                and not PAGE.match(rows[k][2])
                and rows[k][2].startswith(" ")
            ):
                parts.append(rows[k][2].strip())
                k += 1
            raw = re.sub(r"\s+\d{1,3}$", "", " ".join(parts)).strip()
            records.append(
                {
                    "source": source,
                    "locator": rows[i][1],
                    "line": rows[i][0],
                    "kind": "author_year",
                    "title": raw[:400],
                    "url": "",
                    "core_mark": False,
                }
            )
            i = k
        else:
            i += 1
    return records


def author_year_title(raw):
    """Title segment of an 'Authors (year). Title. Venue.' reference."""
    m = re.search(r"\(\d{4}[a-z]?\)\.\s*(.+?)\.\s+[A-Z][^.]*\.?\s*$", raw)
    if m:
        return m.group(1)
    m = re.search(r"\d{4}\.\s*(.+?)\.", raw)
    return m.group(1) if m else raw


def identity(record):
    """Deduplication key: DOI, arXiv id, specific URL, then normalized title."""
    if record["kind"] == "author_year":
        return "title:" + norm_title(author_year_title(record["title"]))
    blob = f"{record['url']} {record['title']}"
    doi = DOI.search(blob)
    if doi:
        return "doi:" + doi.group(1).lower().rstrip(".")
    arxiv = ARXIV.search(record["url"])
    if arxiv:
        return "arxiv:" + re.sub(r"v\d+$", "", arxiv.group(1).lower())
    if record["url"] and not is_generic(record["url"]):
        parts = urlsplit(record["url"])
        return (
            "url:"
            + host(record["url"])
            + parts.path.rstrip("/").lower()
            + ("?" + parts.query if parts.query else "")
        )
    key = norm_title(record["title"])
    return (
        "title:" + key if key else "unlocated:" + record["source"] + str(record["line"])
    )


def topics(text):
    """Keyword topic rule v1 (declared, reviewable, not an automatic truth)."""
    low = text.lower()
    return [name for name, pattern in TOPICS if re.search(pattern, low)]


def classify(entry):
    """load_bearing/context/tool/unavailable by declared rule; overrides applied later."""
    found = set(entry["topics"])
    if found & LOAD_BEARING:
        cls = "load_bearing"
    elif found & TOOL:
        cls = "tool"
    else:
        cls = "context"
    if (
        not entry["doi"]
        and not entry["arxiv"]
        and all(is_generic(u) for u in entry["urls"] or [""])
        and cls != "load_bearing"
        and entry["kind"] == "url"
    ):
        cls = "unavailable_locator"
    return cls


def deduplicate(records):
    """Merge records sharing an identity; keep every source locator."""
    merged = {}
    order = []
    for r in records:
        key = identity(r)
        if key not in merged:
            merged[key] = {
                "citation_key": key,
                "titles": [],
                "urls": [],
                "locators": [],
                "kinds": set(),
                "core_mark": False,
            }
            order.append(key)
        m = merged[key]
        if r["title"] and r["title"] not in m["titles"]:
            m["titles"].append(r["title"])
        if r["url"] and r["url"] not in m["urls"]:
            m["urls"].append(r["url"])
        loc = f"{r['locator']}#L{r['line']}"
        if loc not in m["locators"]:
            m["locators"].append(loc)
        m["kinds"].add(r["kind"])
        m["core_mark"] |= r["core_mark"]
    out = []
    for n, key in enumerate(order, start=1):
        m = merged[key]
        blob = " ".join(m["titles"] + m["urls"])
        doi = DOI.search(blob)
        arxiv = next((ARXIV.search(u) for u in m["urls"] if ARXIV.search(u)), None)
        entry = {
            "citation_id": f"C{n:03d}",
            "citation_key": key,
            "title": max(m["titles"], key=len) if m["titles"] else "",
            "doi": doi.group(1).rstrip(".") if doi else "",
            "arxiv": arxiv.group(1) if arxiv else "",
            "urls": m["urls"],
            "locators": m["locators"],
            "sources": sorted({x.split(":")[0] for x in m["locators"]}),
            "kind": "+".join(sorted(m["kinds"])),
            "core_mark": m["core_mark"],
        }
        entry["topics"] = topics(entry["title"] + " " + " ".join(entry["urls"]))
        entry["class"] = classify(entry)
        out.append(entry)
    return out
