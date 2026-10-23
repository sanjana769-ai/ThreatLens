
import re


TECHNIQUE_ID_PATTERN = re.compile(
    r"\bT\d{4}(?:\.\d{3})?\b",
    flags=re.IGNORECASE,
)


def extract_cited_ids(answer):
    citation_spans = re.findall(
        r"\[([^\]]+)\]|【([^】]+)】",
        answer,
    )

    cited_ids = set()

    for square_span, unicode_span in citation_spans:
        span = square_span or unicode_span
        cited_ids.update(
            item.upper()
            for item in TECHNIQUE_ID_PATTERN.findall(span)
        )

    return cited_ids


def validate_citations(answer, retrieved_ids):
    cited_ids = extract_cited_ids(answer)
    allowed_ids = {item.upper() for item in retrieved_ids}

    return (
        bool(cited_ids) and cited_ids.issubset(allowed_ids),
        cited_ids,
    )


tests = [
    (
        "Credential capture [T1056]",
        {"T1056", "T1606"},
        True,
    ),
    (
        "Forged credentials [T1606]",
        {"T1056", "T1606"},
        True,
    ),
    (
        "Unknown technique [T9999]",
        {"T1056", "T1606"},
        False,
    ),
    (
        "No citations in this answer",
        {"T1056", "T1606"},
        False,
    ),
]

for answer, retrieved_ids, expected in tests:
    actual, cited_ids = validate_citations(answer, retrieved_ids)
    status = "PASS" if actual == expected else "FAIL"
    print(
        f"{status}: cited={sorted(cited_ids)}, "
        f"valid={actual}, expected={expected}"
    )
