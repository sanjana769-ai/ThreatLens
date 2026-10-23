
import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from pypdf import PdfReader


from document_retrieval import (
    clean_pdf_text,
    index_document,
    search_document,
)

# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent
EVALUATION_FILE = ROOT / "evaluation" / "evaluate_rag.py"
RESULTS_FILE = ROOT / "evaluation_results.json"
TECHNIQUES_FILE = ROOT / "data" / "techniques.json"

st.set_page_config(
    page_title="ThreatLens",
    page_icon="🛡️",
    layout="wide",
)

# ==================================================
# SESSION STATE
# ==================================================

DEFAULTS = {
    "collection": None,
    "document_name": None,
    "document_hash": None,
    "document_pages": None,
    "document_answer": None,
    "document_results": [],
    "threat_query": "",
    "threat_answer": None,
    "threat_results": [],
    "evaluation_output": None,
    "evaluation_return_code": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# BACKEND HELPERS
# ==================================================

@st.cache_resource(show_spinner=False)
def get_hybrid_engine():
    from retrieval.hybrid_search import HybridSearch
    return HybridSearch()


def get_technique_count():
    try:
        with TECHNIQUES_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data)
    except (OSError, json.JSONDecodeError):
        return 0


def get_evaluation_results():
    try:
        with RESULTS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def extract_pages(uploaded_file):
    reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    pages = []

    for number, pdf_page in enumerate(reader.pages, start=1):
        text = clean_pdf_text(pdf_page.extract_text() or "")
        pages.append({
            "page_number": number,
            "text": text,
        })

    return pages


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("🛡️ ThreatLens")
st.sidebar.caption("Cybersecurity Intelligence")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Document Analysis",
        "Threat Intelligence",
        "Evaluation",
    ],
)

st.sidebar.divider()
st.sidebar.write("**System status**")

if st.session_state.collection:
    st.sidebar.success("Document indexed")
else:
    st.sidebar.info("Ready for analysis")

st.sidebar.caption("ThreatLens MVP · Evidence-grounded AI")



# ==================================================
# OVERVIEW
# ==================================================

if page == "Overview":

    st.title("🛡️ ThreatLens Command Center")
    st.write(
        "Investigate cybersecurity threats using hybrid retrieval, "
        "MITRE ATT&CK intelligence, and evidence-grounded AI."
    )

    # ------------------------------
    # DYNAMIC METRICS
    # ------------------------------

    results = get_evaluation_results()
    total_tests = len(results)
    passed = sum(
        item.get("passed") is True for item in results
    )

    total_records = get_technique_count()
    indexed_documents = (
        1 if st.session_state.collection else 0
    )
    retrieved = len(st.session_state.threat_results)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("ATT&CK Records", f"{total_records:,}")
    col2.metric("Indexed Documents", indexed_documents)
    col3.metric("Retrieved Techniques", retrieved)
    col4.metric(
        "Evaluation Pass Rate",
        f"{passed / total_tests * 100:.0f}%"
        if total_tests else "—",
    )

    st.divider()

    # ------------------------------
    # ANALYTICS DASHBOARD
    # ------------------------------

    st.subheader("📊 Security Analytics")

    # Load ATT&CK techniques safely.
    try:
        with TECHNIQUES_FILE.open("r", encoding="utf-8") as f:
            technique_data = json.load(f)

        if isinstance(technique_data, dict):
            if isinstance(technique_data.get("techniques"), list):
                techniques = technique_data["techniques"]
            else:
                techniques = list(technique_data.values())
        elif isinstance(technique_data, list):
            techniques = technique_data
        else:
            techniques = []

    except (OSError, json.JSONDecodeError):
        techniques = []

    left, right = st.columns(2)

    # ------------------------------
    # PIE CHART: TACTIC DISTRIBUTION
    # ------------------------------

    with left:
        st.markdown("#### ATT&CK Tactic Distribution")

        tactic_counts = {}

        for technique in techniques:
            if not isinstance(technique, dict):
                continue

            tactics = technique.get("tactics", [])

            if isinstance(tactics, str):
                tactics = [tactics]

            if not isinstance(tactics, list):
                continue

            # Count each technique once per associated tactic.
            for tactic in set(
                str(t).strip()
                for t in tactics
                if t and str(t).strip()
            ):
                tactic_counts[tactic] = (
                    tactic_counts.get(tactic, 0) + 1
                )

        if tactic_counts:
            tactic_df = pd.DataFrame(
                [
                    {"Tactic": tactic, "Techniques": count}
                    for tactic, count in tactic_counts.items()
                ]
            ).sort_values("Techniques", ascending=False)

            pie = px.pie(
                tactic_df,
                names="Tactic",
                values="Techniques",
                hole=0.48,
                title="Techniques by tactic",
            )
            pie.update_layout(
                template="plotly_dark",
                margin=dict(t=55, b=15, l=10, r=10),
                legend_title_text="Tactics",
            )
            st.plotly_chart(pie, use_container_width=True)

        else:
            st.info(
                "No tactic data was found in techniques.json. "
                "Check the dataset's tactics field."
            )

    # ------------------------------
    # BAR CHART: TOP RETRIEVED TECHNIQUES
    # ------------------------------

    with right:
        st.markdown("#### Retrieved Techniques")

        retrieved_items = st.session_state.threat_results

        if retrieved_items:
            retrieval_counts = {}

            for item in retrieved_items:
                technique_id = item.get("id", "Unknown")
                technique_name = item.get("name", "Unnamed")

                label = f"{technique_id} · {technique_name}"
                retrieval_counts[label] = (
                    retrieval_counts.get(label, 0) + 1
                )

            retrieval_df = pd.DataFrame(
                [
                    {"Technique": name, "Results": count}
                    for name, count in retrieval_counts.items()
                ]
            ).sort_values("Results", ascending=True)

            bar = px.bar(
                retrieval_df.tail(10),
                x="Results",
                y="Technique",
                orientation="h",
                title="Techniques in the latest search",
            )
            bar.update_layout(
                template="plotly_dark",
                margin=dict(t=55, b=15, l=10, r=10),
                yaxis_title="",
                xaxis_title="Result count",
            )
            st.plotly_chart(bar, use_container_width=True)

        else:
            st.info(
                "Run a search in Threat Intelligence to populate "
                "this chart."
            )

    # ------------------------------
    # LINE CHART: EVALUATION RESULTS
    # ------------------------------

    st.divider()
    st.subheader("📈 Evaluation Results by Test")

    if results:
        evaluation_df = pd.DataFrame(
            [
                {
                    "Test": item.get("name", f"Test {i + 1}"),
                    "Pass rate (%)": (
                        100 if item.get("passed") is True else 0
                    ),
                    "Status": (
                        "Passed"
                        if item.get("passed") is True
                        else "Not passed"
                    ),
                }
                for i, item in enumerate(results)
            ]
        )

        line = px.line(
            evaluation_df,
            x="Test",
            y="Pass rate (%)",
            markers=True,
            color="Status",
            title="Pass/fail status of saved evaluation tests",
            hover_data=["Status"],
        )
        line.update_layout(
            template="plotly_dark",
            margin=dict(t=55, b=100, l=10, r=10),
            xaxis_title="Evaluation test",
            yaxis_title="Status (%)",
            yaxis_range=[0, 105],
        )
        line.update_xaxes(tickangle=-35)

        st.plotly_chart(line, use_container_width=True)
        st.caption(
            "This chart shows the saved status of each test, "
            "not performance over time. A historical trend requires "
            "saving results from multiple evaluation runs."
        )

    else:
        st.info(
            "No evaluation results are available yet. "
            "Open Evaluation and run the evaluation suite first."
        )

    st.divider()

    # ------------------------------
    # AI INVESTIGATION COPILOT
    # ------------------------------

    st.subheader("🔎 AI Investigation Copilot")

    with st.form("overview_form"):
        question = st.text_area(
            "What do you want to investigate?",
            placeholder=(
                "Example: How do attackers steal credentials?"
            ),
        )

        investigate = st.form_submit_button(
            "Investigate with AI",
            type="primary",
        )

    if investigate:
        if question.strip():
            st.session_state.threat_query = question.strip()
            st.session_state.threat_answer = None
            st.session_state.threat_results = []

            st.info(
                "Your question is ready. Open Threat Intelligence "
                "from the sidebar and select Search and Analyze."
            )
        else:
            st.warning("Enter a question first.")

    # ------------------------------
    # WORKSPACES
    # ------------------------------

    st.divider()
    st.subheader("Workspaces")

    left, right = st.columns(2)

    with left:
        st.markdown("### 📄 Document Intelligence")
        st.write(
            "Upload reports, index PDF content, "
            "and retrieve relevant evidence."
        )

        st.markdown("### 📊 RAG Evaluation")
        st.write(
            "Inspect retrieval relevance, citation validity, "
            "and insufficient-evidence behavior."
        )

    with right:
        st.markdown("### 🛡️ MITRE ATT&CK Explorer")
        st.write(
            "Explore relevant techniques, tactics, platforms, "
            "and source references."
        )

        st.markdown("### 🔍 Evidence-first analysis")
        st.write(
            "Keep source evidence separate from "
            "AI-generated explanations."
        )

# ==================================================
# DOCUMENT ANALYSIS
# ==================================================

elif page == "Document Analysis":

    st.title("Document Intelligence")
    st.write(
        "Upload a cybersecurity PDF, index it locally, "
        "and ask questions about its contents."
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
    )

    if uploaded_file is not None:

        import hashlib

        file_hash = hashlib.sha256(
            uploaded_file.getvalue()
        ).hexdigest()

        if file_hash != st.session_state.document_hash:
            st.session_state.collection = None
            st.session_state.document_hash = file_hash
            st.session_state.document_name = uploaded_file.name
            st.session_state.document_pages = None
            st.session_state.document_answer = None
            st.session_state.document_results = []

        try:
            if st.session_state.document_pages is None:
                with st.spinner("Extracting PDF text..."):
                    st.session_state.document_pages = extract_pages(
                        uploaded_file
                    )

            pages = st.session_state.document_pages
            word_count = sum(
                len(p["text"].split()) for p in pages
            )

            a, b = st.columns(2)
            a.metric("PDF Pages", len(pages))
            b.metric("Extracted Words", f"{word_count:,}")

            with st.expander("Preview extracted text"):
                for p in pages[:3]:
                    st.markdown(f"**Page {p['page_number']}**")
                    st.write(p["text"][:1500] or "No text extracted.")

            if st.session_state.collection is None:
                if st.button(
                    "Build Search Index",
                    type="primary",
                ):
                    with st.spinner("Creating embeddings and indexing..."):
                        name, count = index_document(
                            pages,
                            uploaded_file.name,
                        )

                    st.session_state.collection = name
                    st.success(f"Indexed successfully: {count} chunks.")
                    st.rerun()
            else:
                st.success("Document indexed and ready for questions.")

        except Exception as exc:
            st.error("Unable to process this PDF.")
            st.exception(exc)

    if st.session_state.collection:
        st.divider()
        st.subheader("Ask your document")

        with st.form("document_question_form"):
            question = st.text_input(
                "Question",
                placeholder="What threats are described in this report?",
            )
            ask = st.form_submit_button(
                "Analyze Document",
                type="primary",
            )

        if ask:
            if not question.strip():
                st.warning("Enter a question.")
            else:
                try:
                    with st.spinner("Retrieving document evidence..."):
                        evidence = search_document(
                            st.session_state.collection,
                            question.strip(),
                            top_k=4,
                        )

                    st.session_state.document_results = evidence
                    st.session_state.document_answer = None

                    if evidence:
                        context = "\n\n---\n\n".join(
                            f"Source: {item['source']}\n"
                            f"Page: {item['page_number']}\n"
                            f"Evidence: {item['text']}"
                            for item in evidence
                        )

                        from llm.groq_client import generate_document_answer

                        with st.spinner("Generating AI analysis..."):
                            answer = generate_document_answer(
                                question.strip(),
                                context,
                            )

                        st.session_state.document_answer = answer
                    else:
                        st.info("No relevant evidence was retrieved.")

                except Exception as exc:
                    st.error("Document analysis failed.")
                    st.exception(exc)

        if st.session_state.document_answer:
            st.subheader("AI Analysis")
            st.markdown(st.session_state.document_answer)

        if st.session_state.document_results:
            st.subheader("Retrieved Evidence")

            for rank, item in enumerate(
                st.session_state.document_results,
                start=1,
            ):
                with st.expander(
                    f"Evidence {rank} · Page {item['page_number']}"
                ):
                    st.caption(f"Source: {item['source']}")
                    if item.get("distance") is not None:
                        st.caption(
                            f"Distance: {item['distance']:.4f}"
                        )
                    st.write(item["text"])


# ==================================================
# THREAT INTELLIGENCE
# ==================================================

elif page == "Threat Intelligence":

    st.title("MITRE ATT&CK Explorer")
    st.write(
        "Search the technique knowledge base using hybrid retrieval."
    )

    with st.form("threat_form"):
        query = st.text_area(
            "Investigation question",
            value=st.session_state.threat_query,
            placeholder=(
                "Example: How do attackers use PowerShell?"
            ),
        )

        a, b = st.columns(2)

        with a:
            top_k = st.slider(
                "Number of results",
                1,
                10,
                5,
            )

        with b:
            alpha = st.slider(
                "Semantic search weight",
                0.0,
                1.0,
                0.5,
                0.1,
            )

        search = st.form_submit_button(
            "Search and Analyze",
            type="primary",
        )

    if search:
        if not query.strip():
            st.warning("Enter a threat question.")
        else:
            try:
                with st.spinner("Searching ATT&CK intelligence..."):
                    engine = get_hybrid_engine()
                    techniques = engine.search(
                        query.strip(),
                        top_k=top_k,
                        alpha=alpha,
                    )

                st.session_state.threat_query = query.strip()
                st.session_state.threat_results = techniques
                st.session_state.threat_answer = None

                if techniques:
                    context = "\n\n---\n\n".join(
                        f"Source ID: {item['id']}\n"
                        f"Technique: {item['name']}\n"
                        f"Tactics: {', '.join(item.get('tactics', []))}\n"
                        f"Platforms: {', '.join(item.get('platforms', []))}\n"
                        f"Description: {item.get('description', '')}\n"
                        f"Source URL: {item.get('url', '')}"
                        for item in techniques
                    )

                    from llm.groq_client import generate_answer

                    with st.spinner("Generating grounded analysis..."):
                        answer = generate_answer(
                            query.strip(),
                            context,
                        )

                    st.session_state.threat_answer = answer
                else:
                    st.info("No matching techniques found.")

            except Exception as exc:
                st.error("Threat intelligence search failed.")
                st.exception(exc)

    if st.session_state.threat_answer:
        st.subheader("AI-Generated Analysis")
        st.markdown(st.session_state.threat_answer)

    if st.session_state.threat_results:
        st.divider()
        st.subheader("Retrieved ATT&CK Techniques")

        for rank, item in enumerate(
            st.session_state.threat_results,
            start=1,
        ):
            with st.expander(
                f"{rank}. {item.get('id', 'Unknown')} · "
                f"{item.get('name', 'Unnamed technique')}"
            ):
                st.write(
                    item.get("description", "No description available.")
                )

                a, b = st.columns(2)

                with a:
                    st.markdown("**Tactics**")
                    st.write(
                        ", ".join(item.get("tactics", []))
                        or "Not specified"
                    )

                with b:
                    st.markdown("**Platforms**")
                    st.write(
                        ", ".join(item.get("platforms", []))
                        or "Not specified"
                    )

                score = item.get("score")
                if isinstance(score, (int, float)):
                    st.caption(f"Retrieval score: {score:.4f}")

                if item.get("url"):
                    st.markdown(
                        f"[Open source reference]({item['url']})"
                    )


# ==================================================
# EVALUATION
# ==================================================

elif page == "Evaluation":

    st.title("RAG Evaluation Lab")
    st.write(
        "Evaluate retrieval quality, citation validity, and abstention behavior."
    )

    st.warning(
        "Running evaluations may call the configured Groq model and consume API quota."
    )

    if not EVALUATION_FILE.exists():
        st.error(f"Evaluation script not found: {EVALUATION_FILE}")
    elif st.button("Run Evaluation Suite", type="primary"):
        try:
            with st.spinner("Running evaluation tests..."):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "evaluation.evaluate_rag",
                    ],
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                    timeout=900,
                    check=False,
                )

            output = completed.stdout or ""

            if completed.stderr:
                output += "\n\n--- STDERR ---\n" + completed.stderr

            st.session_state.evaluation_output = output
            st.session_state.evaluation_return_code = completed.returncode

        except subprocess.TimeoutExpired:
            st.error("Evaluation timed out after 15 minutes.")
        except Exception as exc:
            st.error("Unable to run the evaluation suite.")
            st.exception(exc)

    results = get_evaluation_results()

    if results:
        total = len(results)
        passed = sum(
            item.get("passed") is True for item in results
        )
        failed = total - passed

        a, b, c = st.columns(3)
        a.metric("Tests Passed", passed)
        b.metric("Tests Not Passed", failed)
        c.metric("Pass Rate", f"{passed / total * 100:.0f}%")

        if failed == 0:
            st.success("All saved evaluation tests passed.")
        else:
            st.warning("Some tests need attention.")

        rows = []

        for item in results:
            checks = item.get("checks") or {}

            rows.append({
                "Test": item.get("name", "Unnamed test"),
                "Status": (
                    "PASS" if item.get("passed") is True else "NOT PASSED"
                ),
                "Retrieval": checks.get("retrieval_relevance"),
                "Citations": checks.get("citation_validity"),
                "Abstention": checks.get("abstention_behavior"),
            })

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("View raw evaluation records"):
            st.json(results)
    else:
        st.info("No saved results found. Run the evaluation suite.")

    if st.session_state.evaluation_output is not None:
        if st.session_state.evaluation_return_code == 0:
            st.success("Evaluation process completed.")
        else:
            st.error("Evaluation process reported an error.")

        with st.expander("Evaluation logs"):
            st.code(st.session_state.evaluation_output)


# ==================================================
# FOOTER
# ==================================================

st.divider()
st.caption(
    "ThreatLens · Evidence-grounded cybersecurity intelligence"
)
