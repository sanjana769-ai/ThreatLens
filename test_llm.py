
from llm.groq_client import generate_answer

evidence = """
[Source: DEMO-001]
A fictional security report states that an exposed
administrative interface may allow unauthorized access.
The report does not identify a CVE or a threat actor.
"""

question = (
    "What risk does the report describe, and "
    "does it identify a CVE or threat actor?"
)

answer = generate_answer(question, evidence)

print("\n" + "=" * 60)
print("THREATLENS ANALYSIS")
print("=" * 60)
print(answer)
print("=" * 60)
