
import json
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "enterprise-attack.json"

SOURCE_URL = (
    "https://raw.githubusercontent.com/"
    "mitre-attack/attack-stix-data/master/"
    "enterprise-attack/enterprise-attack.json"
)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading MITRE ATT&CK Enterprise data...")

    response = requests.get(
        SOURCE_URL,
        timeout=(15, 180),
    )
    response.raise_for_status()

    bundle = response.json()
    objects = bundle.get("objects", [])

    if not objects:
        raise RuntimeError(
            "Downloaded data contains no STIX objects."
        )

    OUTPUT_FILE.write_text(
        json.dumps(bundle, ensure_ascii=False),
        encoding="utf-8",
    )

    techniques = [
        obj for obj in objects
        if obj.get("type") == "attack-pattern"
        and not obj.get("revoked", False)
        and not obj.get("x_mitre_deprecated", False)
    ]

    print("Download successful!")
    print(f"Total STIX objects: {len(objects)}")
    print(f"Active technique records: {len(techniques)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
