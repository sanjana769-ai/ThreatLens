import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "evaluation_results.json"
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.hybrid_search import HybridSearch
from llm.groq_client import generate_answer


# --------------------------------------------------
# 1. EVALUATION TEST CASES
# --------------------------------------------------

TEST_CASES = [
    {
        "name": "Credential theft",
        "question": "What techniques can attackers use to steal credentials?",
        "expected_ids": {"T1539", "T1056"},
        "should_abstain": False,
    },
    {
        "name": "Forge web credentials",
        "question": "How can adversaries forge web credentials?",
        "expected_ids": {"T1606"},
        "should_abstain": False,
    },
    {
        "name": "Steal web session cookies",
        "question": "How can stolen web session cookies be abused?",
        "expected_ids": {"T1539"},
        "should_abstain": False,
    },
    {
        "name": "Input capture",
        "question": "What is input capture in MITRE ATT&CK?",
        "expected_ids": {"T1056"},
        "should_abstain": False,
    },
    {
        "name": "Application access tokens",
        "question": (
            "Which technique involves stealing application access tokens?"
        ),
        "expected_ids": {"T1528"},
        "should_abstain": False,
    },
    {
        "name": "Alternate authentication material",
        "question": (
            "What does using alternate authentication material mean?"
        ),
        "expected_ids": {"T1550"},
        "should_abstain": False,
    },
    {
        "name": "Command and control",
        "question": (
            "Which MITRE ATT&CK techniques support command-and-control "
            "communication, including multi-stage channels and dynamic resolution?"
        ),
        "expected_ids": {"T1104", "T1568"},
        "should_abstain": False,
    },
    {
        "name": "Persistence",
        "question": (
            "Which MITRE ATT&CK techniques can maintain persistence "
            "through boot or logon initialization scripts, component firmware, "
            "or event-triggered execution?"
        ),
        "expected_ids": {"T1037", "T1542.002", "T1546.017"},
        "retrieval_queries": [
            "T1037 boot or logon initialization scripts",
            "T1542.002 component firmware",
            "T1546.017 Linux udev rules hardware events persistence",
        ],
        "should_abstain": False,
    },
    {
        "name": "Masquerading",
        "question": "How do adversaries masquerade files?",
        "expected_ids": {"T1036"},
        "should_abstain": False,
    },
    {
        "name": "Fictional attack",
        "question": (
            "What is the official ThreatLens detection signature "
            "for the fictional Zorblax-9000 attack?"
        ),
        "expected_ids": set(),
        "should_abstain": True,
    },
]


ABSTENTION_PHRASES = (
    "insufficient evidence",
    "does not contain",
    "cannot provide",
    "not possible to provide",
    "unable to determine",
    "no information",
    "cannot be determined",
)


# --------------------------------------------------
# 2. CITATION AND RETRIEVAL HELPERS
# --------------------------------------------------


def extract_cited_ids(answer):
    """Extract technique IDs appearing inside citation brackets."""
    citation_spans = re.findall(
        r"\[([^\]]+)\]|【([^】]+)】",
        answer,
    )

    cited_ids = set()

    for square_span, unicode_span in citation_spans:
        span = square_span or unicode_span

        cited_ids.update(
            item.upper()
            for item in re.findall(
                r"\bT\d{4}(?:\.\d{3})?\b",
                span,
                flags=re.IGNORECASE,
            )
        )

    return cited_ids


def technique_matches(expected_id, actual_id):
    """Match an exact technique or a sub-technique of its parent."""
    expected_id = expected_id.upper()
    actual_id = actual_id.upper()

    return (
        actual_id == expected_id
        or actual_id.startswith(expected_id + ".")
    )


def calculate_retrieval_coverage(expected_ids, retrieved_ids):
    """Calculate the fraction of expected techniques retrieved."""
    if not expected_ids:
        return None

    matched_count = sum(
        any(
            technique_matches(expected, actual)
            for actual in retrieved_ids
        )
        for expected in expected_ids
    )

    return matched_count / len(expected_ids)


# --------------------------------------------------
# 3. RETRIEVAL
# --------------------------------------------------

def retrieve_test_evidence(search_engine, test):
    """Retrieve and merge evidence for one evaluation case."""
    question = test["question"]
    retrieval_queries = test.get("retrieval_queries", [question])

    merged_results = {}

    for query in retrieval_queries:
        query_results = search_engine.search(query, top_k=5)

        for result in query_results:
            technique_id = result["id"]

            if (
                technique_id not in merged_results
                or result["score"]
                > merged_results[technique_id]["score"]
            ):
                merged_results[technique_id] = result

    results = sorted(
        merged_results.values(),
        key=lambda item: item["score"],
        reverse=True,
    )[:5]

    context_parts = []

    for result in results:
        context_parts.append(
            f"Source ID: {result['id']}\n"
            f"Technique: {result['name']}\n"
            f"Tactics: {', '.join(result.get('tactics', []))}\n"
            f"Description: {result['description']}\n"
            f"Source URL: {result.get('url', 'Not available')}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return results, context


# --------------------------------------------------
# 4. RUN ONE TEST
# --------------------------------------------------

def evaluate_test(search_engine, test):
    """Run retrieval, generate an answer, and evaluate its quality."""
    question = test["question"]

    print("\n" + "=" * 65)
    print(f"TEST: {test['name']}")
    print("=" * 65)
    print("Question:", question)

    # Retrieval failures are reported separately from LLM failures.
    try:
        results, context = retrieve_test_evidence(search_engine, test)
    except Exception as exc:
        print(f"RETRIEVAL ERROR: {type(exc).__name__}: {exc}")

        return {
            "name": test["name"],
            "passed": False,
            "checks": {
                "retrieval_relevance": False,
                "citation_validity": None,
                "abstention_behavior": None,
            },
            "error": str(exc),
        }

    retrieved_ids = {
        item["id"].upper()
        for item in results
    }

    print("Retrieved IDs:", sorted(retrieved_ids))

    expected_ids = test["expected_ids"]

    retrieval_coverage = calculate_retrieval_coverage(
        expected_ids,
        retrieved_ids,
    )

    relevant_found = (
        None
        if retrieval_coverage is None
        else retrieval_coverage == 1.0
    )

    if retrieval_coverage is not None:
        print(f"Expected-technique coverage: {retrieval_coverage:.0%}")

    # Empty evidence should not be sent to the LLM.
    if not context.strip():
        answer = (
            "Insufficient evidence in the retrieved sources "
            "to answer this question."
        )
        llm_failed = False
    else:
        try:
            answer = generate_answer(question, context)
            llm_failed = False

        except Exception as exc:
            print(
                f"\nLLM evaluation failed for '{test['name']}': "
                f"{type(exc).__name__}: {exc}"
            )

            return {
                "name": test["name"],
                "passed": None,
                "status": "INCOMPLETE",
                "checks": {
                    "retrieval_relevance": relevant_found,
                    "citation_validity": None,
                    "abstention_behavior": None,
                },
                "error": str(exc),
            }

    print("\nAnswer:\n", answer)

    cited_ids = extract_cited_ids(answer)
    print("Cited IDs:", sorted(cited_ids))

    abstained = any(
        phrase in answer.lower()
        for phrase in ABSTENTION_PHRASES
    )

    answer_is_nonempty = bool(answer.strip())

    if test["should_abstain"]:
        abstention_check = abstained
    else:
        abstention_check = answer_is_nonempty and not abstained

    if test["should_abstain"] and abstained:
        citation_valid = True
    else:
        citation_valid = (
            bool(cited_ids)
            and cited_ids.issubset(retrieved_ids)
        )

    checks = {
        "retrieval_relevance": relevant_found,
        "citation_validity": citation_valid,
        "abstention_behavior": abstention_check,
    }

    evaluated_checks = [
        status
        for status in checks.values()
        if status is not None
    ]

    passed = (
        bool(evaluated_checks)
        and all(status is not False for status in checks.values())
    )

    print("\nChecks:")

    for name, status in checks.items():
        if status is None:
            result_label = "NOT EVALUATED"
        else:
            result_label = "PASS" if status else "FAIL"

        print(f"  {name}: {result_label}")

    print("Overall:", "PASS" if passed else "FAIL")

    if not passed:
        failed_checks = [
            name
            for name, status in checks.items()
            if status is False
        ]
        print("FAILED CHECKS:", failed_checks)

    return {
        "name": test["name"],
        "passed": passed,
        "checks": checks,
    }


# --------------------------------------------------
# 5. EVALUATION SUMMARY
# --------------------------------------------------

def print_summary(results_summary):
    """Print pass, fail, and incomplete evaluation results."""
    total = len(results_summary)


    test_cases_by_name = {
        test["name"]: test
        for test in TEST_CASES
    }

    def has_incomplete_checks(result):
        if result.get("error"):
            return True

        checks = result.get("checks", {})
        test_case = test_cases_by_name.get(result["name"], {})
        expected_ids = test_case.get("expected_ids", set())

        for check_name, status in checks.items():
            if status is None:
                # Retrieval relevance is not applicable when
                # the test has no expected technique IDs.
                if (
                    check_name == "retrieval_relevance"
                    and not expected_ids
                ):
                    continue

                return True

        return False

    incomplete_tests = [
        result
        for result in results_summary
        if has_incomplete_checks(result)
    ]

    incomplete_names = {
        result["name"]
        for result in incomplete_tests
    }

    passed_tests = [
        result
        for result in results_summary
        if result.get("passed") is True
        and result["name"] not in incomplete_names
    ]

    failed_tests = [
        result
        for result in results_summary
        if result.get("passed") is False
        and result["name"] not in incomplete_names
    ]


    evaluated_relevance = sum(
        result["checks"]["retrieval_relevance"] is not None
        for result in results_summary
    )

    print("\n" + "=" * 65)
    print("THREATLENS EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Tests with results: {total}/{len(TEST_CASES)}")
    print(f"Passed: {len(passed_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Incomplete: {len(incomplete_tests)}")
    print(f"Retrieval relevance evaluated: {evaluated_relevance}/{total}")

    if passed_tests:
        print("\nPassed tests:")
        for result in passed_tests:
            print(f"  - {result['name']}")

    if failed_tests:
        print("\nFailed tests:")
        for result in failed_tests:
            print(f"  - {result['name']}")

    if incomplete_tests:
        print("\nIncomplete tests:")
        for result in incomplete_tests:
            print(f"  - {result['name']}")
            if result.get("error"):
                print(f"    Reason: {result['error']}")

    if total < len(TEST_CASES):
        print(
            f"\nEvaluation not finished: "
            f"{len(TEST_CASES) - total} test(s) have no results yet."
        )

    print(
        "\nA test is incomplete when an API error occurs or "
        "one or more checks were not evaluated."
    )


# --------------------------------------------------
# 6. MAIN
# --------------------------------------------------


def load_checkpoint():
    """Load previously completed evaluation results."""
    if not CHECKPOINT_PATH.exists():
        return {}

    try:
        with CHECKPOINT_PATH.open("r", encoding="utf-8") as file:
            saved_results = json.load(file)

        return {
            result["name"]: result
            for result in saved_results
            if isinstance(result, dict)
            and isinstance(result.get("name"), str)
            and result["name"] in {test["name"] for test in TEST_CASES}
            and result.get("passed") is True
        }

    except (json.JSONDecodeError, OSError, KeyError, TypeError):
        print("Warning: Could not read checkpoint. Starting fresh.")
        return {}



def save_checkpoint(results_by_name):
    """Save successful and attempted test results."""
    ordered_results = [
        results_by_name[test["name"]]
        for test in TEST_CASES
        if test["name"] in results_by_name
    ]

    with CHECKPOINT_PATH.open("w", encoding="utf-8") as file:
        json.dump(ordered_results, file, indent=2)

def main():
    print("Initializing ThreatLens evaluation...")
    print(f"Evaluation cases: {len(TEST_CASES)}")
    print(
        "The local embedding model may load during initialization. "
        "It is reused by the same search-engine instance."
    )

    try:
        search_engine = HybridSearch()
    except Exception as exc:
        print(
            f"Could not initialize HybridSearch: "
            f"{type(exc).__name__}: {exc}"
        )
        sys.exit(1)


    results_by_name = load_checkpoint()

    if results_by_name:
        print(
            f"Resuming evaluation: "
            f"{len(results_by_name)} successful tests already saved."
        )

    for index, test in enumerate(TEST_CASES, start=1):
        if test["name"] in results_by_name:
            print(
                f"\nProgress: {index}/{len(TEST_CASES)} "
                f"- SKIPPING SAVED TEST: {test['name']}"
            )
            continue

        print(f"\nProgress: {index}/{len(TEST_CASES)}")

        result = evaluate_test(search_engine, test)
        results_by_name[test["name"]] = result

        try:
            save_checkpoint(results_by_name)
        except OSError as exc:
            print(f"WARNING: Could not save checkpoint: {exc}")

        if result.get("error"):
            error_text = result["error"].lower()

            is_rate_limit = (
                "429" in error_text
                or "rate_limit" in error_text
                or "ratelimit" in error_text
            )

            if is_rate_limit:
                print(
                    "\nGroq rate limit reached. Evaluation paused. "
                    "Successful tests have been saved. "
                    "Run the command again when quota is available."
                )
                break

    results_summary = [
        results_by_name[test["name"]]
        for test in TEST_CASES
        if test["name"] in results_by_name
    ]

    print_summary(results_summary)
    print(
        f"\nProgress saved: {len(results_summary)}/{len(TEST_CASES)} "
        f"tests have results."
    )
    print(f"Checkpoint: {CHECKPOINT_PATH}")



if __name__ == "__main__":
    main()
