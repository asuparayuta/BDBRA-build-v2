
import re

PRIORITY = [
    "Tracer study",
    "DTI/tractography",
    "Opto/Chemo",
    "Electrophys",
    "Anatomical imaging/clearing",
    "Review",
]

PATTERNS = {
    "Tracer study": [
        r"\bpha[-\s]?l\b", r"phaseolus\s+vulgaris",
        r"\bbda\b", r"biotinylated\s+dextran\s+amine",
        r"\bfluoro[-\s]?gold\b", r"\bfg\b(?![a-z])",
        r"\bctb\b", r"cholera\s+toxin\s+b",
        r"\brabies\b", r"\bpseudorabies\b", r"\bprv\b", r"\bhsv\b",
        r"\bwga\b", r"\bhrp\b",
        r"\baav\b", r"adeno[-\s]?associated\s+virus",
        r"\btracer(s)?\b", r"\binject(ion|ed)\b.*\b(tracer|dye|virus)\b",
        r"\banterograde\b", r"\bretrograde\b"
    ],
    "DTI/tractography": [r"diffusion\s+tensor", r"\bdti\b", r"tractography", r"fiber\s+tracking"],
    "Opto/Chemo": [r"optogenet", r"dreadd", r"\bcno\b", r"channelrhodopsin|chr2|archt|halorhodopsin"],
    "Electrophys": [r"electrophysiolog", r"patch[-\s]?clamp", r"\blfp\b", r"\becog\b"],
    "Anatomical imaging/clearing": [r"clarity", r"idisco|udisco", r"light[-\s]?sheet", r"fmost", r"stpt", r"merfish", r"mapseq", r"memri"],
    "Review": [r"\breview\b", r"meta[-\s]?analys"],
}

def _first_match_label(text: str) -> str:
    t = text.lower()
    for label in PRIORITY:
        for p in PATTERNS[label]:
            if re.search(p, t, flags=re.I):
                return label
    return ""

def infer_method(text: str, start=None, end=None, pad=800) -> str:
    if not text: return "Unspecified"
    n = len(text)
    if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= n:
        s, e = max(0, start - pad), min(n, end + pad)
        window = text[s:e]
        label = _first_match_label(window)
        if label: return label
    return _first_match_label(text[:30000]) or "Unspecified"
