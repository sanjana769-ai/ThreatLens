
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "enterprise-attack.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "techniques.json"
JSONL_FILE = PROJECT_ROOT / "data" / "techniques.jsonl"


def get_external_id(obj):
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id", "")
    return ""


def parse_techniques(bundle):
    records = []

    for obj in bundle.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue

        if obj.get("revoked", False):
            continue

        if obj.get("x_mitre_deprecated", False):
            continue

        external_id = get_external_id(obj)
        name = obj.get("name", "").strip()
        description = obj.get("description", "").strip()

        if not external_id or not name or not description:
            continue

        tactics = [
            phase.get("phase_name", "")
            for phase in obj.get("kill_chain_phases", [])
            if phase.get("kill_chain_name") == "mitre-attack"
        ]

        references = [
            {
                "source": ref.get("source_name", ""),
                "url": ref.get("url", ""),
            }
            for ref in obj.get("external_references", [])
            if ref.get("url")
        ]

        record = {
            "id": external_id,
            "name": name,
            "description": description,
            "tactics": tactics,
            "platforms": obj.get("x_mitre_platforms", []),
            "data_sources": obj.get("x_mitre_data_sources", []),
            "url": next(
                (
                    ref["url"]
                    for ref in references
                    if ref["source"] == "mitre-attack"
                ),
                "",
            ),
            "source": "MITRE ATT&CK Enterprise",
            "stix_id": obj.get("id", ""),
            "references": references,
        }

        records.append(record)

    return records


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    print("Loading STIX dataset...")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        bundle = json.load(file)

    records = parse_techniques(bundle)
    records.sort(key=lambda item: item["id"])

    OUTPUT_FILE.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with JSONL_FILE.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\nParsing complete!")
    print(f"Technique records: {len(records)}")
    print(f"JSON output: {OUTPUT_FILE}")
    print(f"JSONL output: {JSONL_FILE}")

    if records:
        print("\nSample record:")
        print(f"ID: {records[0]['id']}")
        print(f"Name: {records[0]['name']}")
        print(f"Tactics: {', '.join(records[0]['tactics'])}")


if __name__ == "__main__":
    main()
