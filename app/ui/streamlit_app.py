from __future__ import annotations

from html import escape

import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="RepoMind", layout="wide")

API_URL = st.sidebar.text_input("Backend URL", value="http://127.0.0.1:8000")
st.title("RepoMind")
st.caption("Repository intelligence assistant with hybrid retrieval, code graph analysis, evidence-grounded answers, and quality scoring.")
st.markdown(
    """
    <style>
    .citation-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        margin: 0.45rem 0;
        background: #fafafa;
    }
    .citation-path {
        font-weight: 700;
        color: #111827;
    }
    .citation-lines {
        color: #4b5563;
        font-size: 0.88rem;
    }
    .review-note {
        border-left: 4px solid #f59e0b;
        padding: 0.65rem 0.85rem;
        background: #fffbeb;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def table(items: list[dict], fallback: str = "Nothing detected.") -> None:
    if items:
        st.table(items)
    else:
        st.info(fallback)


def render_citation_cards(citations: list[dict], limit: int = 5) -> None:
    if not citations:
        st.caption("No source citations returned for this answer.")
        return
    for citation in citations[:limit]:
        path = citation.get("file_path") or citation.get("path") or "unknown file"
        start = citation.get("start_line")
        end = citation.get("end_line")
        line_label = f"lines {start}-{end}" if start and end else "line information unavailable"
        excerpt = citation.get("excerpt") or citation.get("text") or ""
        st.markdown(
            f"""
            <div class="citation-card">
              <div class="citation-path">{escape(str(path))}</div>
              <div class="citation-lines">{escape(str(line_label))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if excerpt:
            st.code(str(excerpt)[:1200])


def render_llm_status(payload: dict) -> None:
    llm = payload.get("llm") or {}
    if llm.get("used") and llm.get("provider") == "gemini":
        model = llm.get("model", "Gemini")
        st.success(f"Generated with Gemini ({model}).")
    elif llm:
        reason = llm.get("reason", "local fallback was used.")
        st.warning(f"LLM fallback used: {reason}")


repo_url = st.text_input("Public GitHub repository URL", placeholder="https://github.com/owner/repo")
force = st.checkbox("Force rebuild analysis", value=False)

for key, default in {
    "repo_id": None,
    "summary": None,
    "weaknesses": [],
    "graph": {},
    "last_chat": None,
    "chat_history": [],
    "run_answer": None,
    "run_answer_repo_id": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

col1, col2 = st.columns([1, 1])
with col1:
    analyze_clicked = st.button("Analyze repository", type="primary", disabled=not repo_url)
with col2:
    if st.button("Clear cached repo", disabled=not st.session_state.repo_id):
        response = requests.delete(f"{API_URL}/repo/{st.session_state.repo_id}", timeout=30)
        if response.ok:
            for key in ("repo_id", "summary", "weaknesses", "graph", "last_chat", "chat_history", "run_answer", "run_answer_repo_id"):
                st.session_state[key] = None if key in {"repo_id", "summary", "last_chat", "run_answer", "run_answer_repo_id"} else ([] if key in {"weaknesses", "chat_history"} else {})
            st.success("Cache cleared.")
        else:
            st.error(response.text)

if analyze_clicked:
    status_box = st.empty()
    status_box.info("Analyzing repository. Larger repos can take a few minutes.")
    try:
        response = requests.post(f"{API_URL}/analyze_repo", json={"repo_url": repo_url, "force": force}, timeout=900)
        response.raise_for_status()
        payload = response.json()
        st.session_state.repo_id = payload["repo_id"]
        st.session_state.summary = payload["summary"]
        st.session_state.weaknesses = payload["weaknesses"]
        st.session_state.graph = payload["graph"]
        st.session_state.run_answer = None
        st.session_state.run_answer_repo_id = None
        status_box.success("Analysis complete: " + " -> ".join(payload.get("statuses", [])))
    except requests.RequestException as exc:
        status_box.error(f"Analysis failed: {exc}")

if not st.session_state.summary:
    st.info("Enter a GitHub repository URL and run analysis to begin.")
    st.stop()

summary = st.session_state.summary
facts = summary.get("repo_facts", {})
identity = summary.get("project_identity", {})
quality = summary.get("quality_score", {})
overview_tab, arch_tab, run_tab, quality_tab, chat_tab, sources_tab = st.tabs(
    ["Overview", "Architecture", "Run Instructions", "Quality / Weaknesses", "Chat", "Sources"]
)

with overview_tab:
    st.subheader(identity.get("project_name", summary.get("project_name", "Repository")))
    st.markdown(f"**One-line summary:** {identity.get('one_line_summary', summary.get('one_line_summary', 'Not detected'))}")
    st.markdown(f"**What this repo is for:** {identity.get('what_this_repo_is_for', summary.get('what_it_does', 'Not detected'))}")
    st.markdown(f"**Likely target:** {identity.get('target', 'developers or maintainers')}")
    st.markdown("### Tech Stack")
    table(summary.get("tech_stack", []))
    st.markdown("### Important Files")
    table(summary.get("important_files", []))
    st.markdown("### Suggested Questions")
    for question in summary.get("suggested_questions", []):
        st.markdown(f"- {question}")

with arch_tab:
    architecture = summary.get("architecture", {})
    st.markdown("### Architecture Explanation")
    st.write(architecture.get("explanation", "No architecture explanation was generated."))
    st.markdown("### Entrypoint Flow")
    for item in architecture.get("entrypoint_flow", []):
        st.markdown(f"- {item}")
    st.markdown("### Important Modules")
    table(architecture.get("important_modules", []))
    st.markdown("### Diagram")
    mermaid = architecture.get("mermaid") or st.session_state.graph.get("mermaid", "flowchart TD\nRepo[Repository]")
    components.html(
        f"""
        <pre class="mermaid">{escape(mermaid)}</pre>
        <script type="module">
          import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
          mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
        </script>
        """,
        height=420,
        scrolling=True,
    )
    with st.expander("Mermaid source"):
        st.code(f"```mermaid\n{mermaid}\n```", language="markdown")

with run_tab:
    st.markdown("### How To Install / Run")
    if st.session_state.run_answer_repo_id != st.session_state.repo_id:
        try:
            with st.spinner("Generating run instructions with the assistant..."):
                response = requests.post(
                    f"{API_URL}/repo/{st.session_state.repo_id}/chat",
                    json={"question": "How do I install and run this repository locally? Give exact commands and cite the files."},
                    timeout=180,
                )
                response.raise_for_status()
                st.session_state.run_answer = response.json()
                st.session_state.run_answer_repo_id = st.session_state.repo_id
                st.session_state.last_chat = st.session_state.run_answer
        except requests.RequestException as exc:
            st.session_state.run_answer = {
                "answer": f"Could not generate run answer: {exc}",
                "citations": [],
                "llm": {"provider": "gemini", "used": False, "reason": str(exc)},
            }
            st.session_state.run_answer_repo_id = st.session_state.repo_id
    if st.session_state.run_answer:
        st.markdown("### Assistant Run Answer")
        render_llm_status(st.session_state.run_answer)
        st.markdown(st.session_state.run_answer.get("answer", "No answer returned."))
        with st.expander("Sources", expanded=True):
            render_citation_cards(st.session_state.run_answer.get("citations", []))
    official_docs = facts.get("run_doc_paths") or facts.get("install_doc_paths") or facts.get("official_doc_paths", [])
    if official_docs:
        st.markdown("**Official instructions found in:**")
        for path in official_docs:
            st.markdown(f"- `{path}`")
    else:
        st.warning("No README/INSTALL-style official documentation was detected. Commands below may be inferred.")
    run_rows = summary.get("run_instructions", [])
    official_rows = [row for row in run_rows if row.get("official")]
    inferred_rows = [row for row in run_rows if not row.get("official")]
    if official_rows:
        st.markdown("### Official Installation / Run Steps")
        table(official_rows)
    elif inferred_rows:
        st.markdown("### Inferred Run Signals")
        table(inferred_rows)
    if any(row.get("command") == "Not clearly documented." for row in run_rows):
        st.warning("Run instructions are not clearly documented. Add a README quickstart to improve this.")
    st.markdown("### Environment / Package Evidence")
    table([{"path": path, "type": "dependency/package"} for path in facts.get("dependency_files", []) + facts.get("package_files", [])])
    st.markdown("### Helper / Developer Scripts")
    if facts.get("official_helper_commands"):
        st.markdown("Official docs also mention these optional/helper commands:")
        table(facts.get("official_helper_commands", []))
    table(facts.get("helper_scripts", []), "No helper scripts detected.")

with quality_tab:
    st.markdown("### Quality Score")
    st.metric("Total", quality.get("total", 0))
    categories = [{"category": key, **value} for key, value in quality.get("categories", {}).items()]
    table(categories)
    security_review = quality.get("security_review", {})
    st.markdown("### Automated Static Review")
    st.markdown(
        f"""
        <div class="review-note">
        <strong>{escape(str(security_review.get("label", "automated static review findings")).title())}</strong><br>
        Findings: {security_review.get("finding_count", 0)} · Highest severity: {escape(str(security_review.get("highest_severity", "none")))}<br>
        {escape(str(security_review.get("note", "Static findings are not a formal security audit.")))}
        </div>
        """,
        unsafe_allow_html=True,
    )
    confidence = security_review.get("confidence_summary", {})
    if confidence:
        table([{"confidence": key, "count": value} for key, value in confidence.items()])
    st.markdown("### Top Improvements")
    for item in quality.get("top_improvements", []):
        st.markdown(f"- {item}")
    st.markdown("### Weakness Report")
    grouped: dict[str, list[dict]] = {}
    for weakness in st.session_state.weaknesses:
        grouped.setdefault(f"{weakness['severity'].upper()} / {weakness.get('category', 'general')}", []).append(weakness)
    if not grouped:
        st.success("No weaknesses detected by the static analyzer.")
    for group, items in grouped.items():
        with st.expander(group, expanded=group.startswith("HIGH")):
            for weakness in items:
                location = weakness.get("file_path") or "repository-wide"
                if weakness.get("line_number"):
                    location = f"{location}:{weakness['line_number']}"
                st.markdown(f"**{weakness.get('title') or weakness.get('issue')}**")
                st.markdown(f"- Location: `{location}`")
                st.markdown(f"- Severity: `{weakness.get('severity', 'unknown')}` · Confidence: `{weakness.get('confidence', 'medium')}`")
                if weakness.get("evidence"):
                    st.code(weakness["evidence"])
                st.markdown(f"- Why it matters: {weakness['why_it_matters']}")
                st.markdown(f"- Suggested fix: {weakness['suggested_fix']}")

with chat_tab:
    st.markdown("### Ask RepoMind")
    st.caption("Answers are grounded in retrieved repository chunks, graph context, and cited file lines.")
    question = st.chat_input("Ask about the project, run commands, architecture, important files, or weaknesses")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        try:
            with st.spinner("Reading the repo evidence and composing an answer..."):
                response = requests.post(f"{API_URL}/repo/{st.session_state.repo_id}/chat", json={"question": question}, timeout=180)
                response.raise_for_status()
                payload = response.json()
                st.session_state.last_chat = payload
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": payload["answer"],
                        "citations": payload.get("citations", []),
                        "llm": payload.get("llm", {}),
                    }
                )
        except requests.RequestException as exc:
            st.session_state.chat_history.append({"role": "assistant", "content": f"Chat failed: {exc}", "citations": []})

    for index, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_llm_status(message)
                citations = message.get("citations", [])
                if citations:
                    with st.expander("Sources", expanded=index == len(st.session_state.chat_history) - 1):
                        render_citation_cards(citations)

with sources_tab:
    st.markdown("### Source Citations")
    if st.session_state.last_chat:
        render_citation_cards(st.session_state.last_chat.get("citations", []), limit=20)
    else:
        st.info("Ask a question in the Chat tab to see source citations.")
    st.markdown("### Graph Metrics")
    metrics = summary.get("graph_metrics", {})
    metric_rows = [
        {"metric": "nodes", "value": metrics.get("nodes", 0)},
        {"metric": "edges", "value": metrics.get("edges", 0)},
    ]
    metric_rows.extend({"metric": f"node:{key}", "value": value} for key, value in metrics.get("node_kinds", {}).items())
    metric_rows.extend({"metric": f"edge:{key}", "value": value} for key, value in metrics.get("edge_relations", {}).items())
    table(metric_rows)
