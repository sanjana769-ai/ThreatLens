
import re


def normalize(text):
    text = re.sub(
        r"\[(T[0-9]{4})\]\.([0-9]{3})",
        r"[\1.\2]",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\[\[(T[0-9]{4}(?:\.[0-9]{3})?)\]\]",
        r"[\1]",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"【\[(T[0-9]{4}(?:\.[0-9]{3})?)\]】",
        r"[\1]",
        text,
        flags=re.IGNORECASE,
    )

    return text


tests = [
    ("[T1606].001", "[T1606.001]"),
    ("[[T1056]]", "[T1056]"),
    ("【[T1528]】", "[T1528]"),
]

for original, expected in tests:
    result = normalize(original)
    status = "PASS" if result == expected else "FAIL"
    print(f"{status}: {original} -> {result}")
