from __future__ import annotations

import json
import os
import time
from html import escape
from typing import Any

import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="RepoMind", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    .citation-card { border: 1px solid rgba(128,128,128,.35); border-radius: 8px; padding: .6rem .8rem; margin: .4rem 0; }
    .citation-path { font-weight: 700; }
    .citation-meta { opacity: .75; font-size: .85rem; }
    .badge { display:inline-block; padding: .1rem .5rem; border-radius: 999px; font-size: .8rem; margin-right: .3rem; border: 1px solid rgba(128,128,128,.4); }
    .badge-good { background: rgba(34,197,94,.15); } .badge-warn { background: rgba(245,158,11,.18); } .badge-bad { background: rgba(239,68,68,.18); }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- sidebar / connection ------------------------------------------------------------------------
API_URL = st.sidebar.text_input("Backend URL", value=os.getenv("REPOMIND_API_URL", "http://127.0.0.1:8000")).rstrip("/")
API_KEY = st.sidebar.text_input("API key (if required)", value=os.getenv("REPOMIND_API_KEY", ""), type="password")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}


def api(method: str, path: str, timeout: int = 60, **kwargs: Any) -> requests.Response:
    return requests.request(method, f"{API_URL}{path}", headers=HEADERS, timeout=timeout, **kwargs)


def error_detail(response: requests.Response, fallback: str) -> str:
    """Readable message from an API error: {"detail": {"code", "message"}} or legacy string/list details."""
    try:
        payload = response.json()
    except ValueError:
        return response.text or fallback
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, dict):
        message = str(detail.get("message") or detail)
        return f"{message} ({detail['code']})" if detail.get("code") else message
    if isinstance(detail, list):  # pre-normalisation FastAPI validation errors
        return "; ".join(str(item.get("msg", item)) if isinstance(item, dict) else str(item) for item in detail) or fallback
    return str(detail or fallback)


def coverage_banner_text(coverage: dict[str, Any] | None) -> str:
    """Warning text for a partially indexed repository ("" when coverage is complete or unknown)."""
    if not isinstance(coverage, dict) or not coverage.get("partial"):
        return ""
    seen = coverage.get("files_seen") or 0
    not_analyzed = (coverage.get("files_skipped_by_cap") or 0) + (coverage.get("files_dropped_by_chunk_cap") or 0)
    fraction = coverage.get("analyzed_fraction")
    share = f"about {fraction:.0%}" if isinstance(fraction, (int, float)) else "only part"
    counts = f" ({seen - not_analyzed:,} of {seen:,} files)" if seen else ""
    reasons = "; ".join(str(reason.get("message") or reason.get("code")) for reason in coverage.get("reasons", []) if isinstance(reason, dict))
    text = f"**Partial analysis:** {share} of this repository was analyzed{counts}. Answers and findings may miss the rest."
    if any(coverage.get(key) for key in ("truncated_files", "unreadable_files", "parse_failures")):
        text = "**Partial analysis:** Some files were truncated, unreadable, or could not be parsed. File counts do not measure content coverage. Answers may miss relevant code."
    if reasons:
        text += f" Why: {reasons}."
    if not coverage.get("files_skipped_count_exact", True):
        text += " The number of skipped files is a lower bound."
    return text


for key, default in {
    "repo_id": None, "summary": None, "graph": {}, "chat_history": [], "last_citations": [],
    "run_answer": None, "detective": None, "file_view": None,
    "findings": [], "findings_repo": None,
}.items():
    st.session_state.setdefault(key, default)


def load_repo(repo_id: str) -> None:
    summary = api("GET", f"/repo/{repo_id}/summary")
    if not summary.ok:
        st.error(f"Could not load analysis: {error_detail(summary, summary.reason)}")
        return
    st.session_state.repo_id = repo_id
    st.session_state.summary = summary.json()
    st.session_state.graph = api("GET", f"/repo/{repo_id}/graph").json()
    st.session_state.chat_history = []
    st.session_state.run_answer = None
    st.session_state.detective = None
    st.session_state.file_view = None
    st.session_state.findings_repo = None


with st.sidebar:
    try:
        ready = api("GET", "/ready", timeout=5)
        if ready.ok:
            info = ready.json()
            st.success(f"Backend ready · v{info.get('version')}")
            storage = info.get("storage") or {}
            if storage.get("degraded"):
                st.warning(f"Metadata DB unavailable; using JSON files ({storage.get('fallback_reason') or 'unknown reason'}).")
            st.caption(
                f"LLM: **{info.get('llm_provider')}** · embeddings: `{info.get('embedding_model')}` · vectors: {info.get('vector_backend')} · "
                f"reranker: {info.get('reranker')} · tree-sitter: {'on' if info.get('tree_sitter') else 'off'}"
            )
        else:
            st.warning(f"Backend responded {ready.status_code}: {error_detail(ready, ready.reason)}")
        repos = api("GET", "/repos", timeout=5).json() if ready.ok else []
    except requests.RequestException:
        st.error("Backend unreachable. Start it with `bash scripts/run_backend.sh`.")
        repos = []
    if repos:
        st.markdown("**Analyzed repositories**")
        labels = {f"{r['repo_url'].removeprefix('https://github.com/')}": r["repo_id"] for r in repos}
        choice = st.selectbox("Open", ["—"] + sorted(labels), label_visibility="collapsed")
        if choice != "—" and labels[choice] != st.session_state.repo_id:
            load_repo(labels[choice])

st.title("🧠 RepoMind")
st.caption("Repository intelligence with source references, explicit coverage, documentation drift and static findings.")

# --- analysis ----------------------------------------------------------------------------------------
col_url, col_force, col_go = st.columns([6, 1.3, 1.2])
repo_url = col_url.text_input("Public GitHub repository URL", placeholder="https://github.com/owner/repo", label_visibility="collapsed")
force = col_force.checkbox("Force refresh", help="Fetch the latest commit and rebuild every artifact.")
if col_go.button("Analyze", type="primary", disabled=not repo_url, width="stretch"):
    try:
        submitted = api("POST", "/analyze_repo_async", json={"repo_url": repo_url, "force": force}, timeout=30)
        if not submitted.ok:
            st.error(f"Analysis failed: {error_detail(submitted, submitted.reason)}")
            st.stop()
        job = submitted.json()
        bar = st.progress(0, text="Queued…")
        started = time.time()
        while job["state"] not in {"completed", "failed"} and time.time() - started < 1800:
            time.sleep(0.6)
            job = api("GET", f"/jobs/{job['job_id']}", timeout=10).json()
            stage = (job.get("statuses") or ["queued"])[-1]
            bar.progress(min(100, int(job.get("progress") or 0)), text=f"{stage} · {int(time.time() - started)}s")
        if job["state"] == "failed":
            st.error(f"Analysis failed: {job.get('error')} ({job.get('error_code')})")
            st.stop()
        bar.progress(100, text=f"Done in {int(time.time() - started)}s")
        load_repo(job["repo_id"])
    except requests.RequestException as exc:
        st.error(f"Backend request failed: {exc}")

if not st.session_state.summary:
    st.info("Enter a GitHub repository URL (or open an analyzed repo from the sidebar) to begin.")
    st.stop()

summary: dict[str, Any] = st.session_state.summary
repo_id: str = st.session_state.repo_id
facts = summary.get("repo_facts", {})
identity = summary.get("project_identity", {})
quality = summary.get("quality_score", {})
insights = summary.get("insights", {})
ingestion = summary.get("ingestion", {})


def table(items: list[dict], fallback: str = "Nothing detected.") -> None:
    if items:
        st.dataframe(items, width="stretch", hide_index=True)
    else:
        st.caption(fallback)


def mermaid(source: str, height: int = 460) -> None:
    components.html(
        f"""<pre class="mermaid">{escape(source)}</pre>
        <script type="module">
          import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
          mermaid.initialize({{ startOnLoad: true, theme: 'neutral', securityLevel: 'strict' }});
        </script>""",
        height=height,
        scrolling=True,
    )


def open_file(path: str, start: int = 1, end: int | None = None, run_id: str | None = None) -> None:
    params = {"path": path, "start": max(1, start - 3), "end": (end or start) + 3}
    if run_id:
        params["run_id"] = run_id
    response = api("GET", f"/repo/{repo_id}/file", params=params)
    st.session_state.file_view = response.json() if response.ok else {"error": error_detail(response, response.reason), "path": path}


def render_citations(citations: list[dict], key: str, limit: int = 8) -> None:
    if not citations:
        st.caption("No source citations returned.")
        return
    for index, citation in enumerate(citations[:limit]):
        cited = "✅ cited" if citation.get("cited") else "retrieved"
        symbol = f" · <code>{escape(str(citation['symbol']))}</code>" if citation.get("symbol") else ""
        st.markdown(
            f"""<div class="citation-card"><span class="citation-path">[{escape(str(citation.get('id', '')))}] {escape(citation['file_path'])}</span>
            <span class="citation-meta"> · lines {citation.get('start_line')}-{citation.get('end_line')}{symbol} · {cited}</span></div>""",
            unsafe_allow_html=True,
        )
        cols = st.columns([1, 5])
        if cols[0].button("Open", key=f"{key}-open-{index}"):
            open_file(citation["file_path"], citation.get("start_line") or 1, citation.get("end_line"), citation.get("run_id"))
        if citation.get("source_url"):
            cols[0].link_button("GitHub", citation["source_url"])
        with cols[1].expander("Excerpt"):
            st.code(citation.get("excerpt", "")[:1200])


EVIDENCE_BADGES = {"strong": ("badge-good", "strong retrieval match"), "weak": ("badge-warn", "weak retrieval match"), "none": ("badge-bad", "no retrieval support"), "unknown": ("badge-warn", "support unassessed")}


def grounding_badge(grounding: dict | None, llm: dict | None, evidence: dict | None = None) -> None:
    if evidence and evidence.get("level") in EVIDENCE_BADGES:
        css, label = EVIDENCE_BADGES[evidence["level"]]
        st.markdown(f'<span class="badge {css}" title="{escape(str(evidence.get("reason", "")))}">{label}</span>', unsafe_allow_html=True)
        if evidence["level"] != "strong" and evidence.get("reason"):
            st.caption(f"Evidence check: {evidence['reason']}")
    if not grounding:
        return
    score = grounding.get("grounding_score", 0)
    css = "badge-good" if score >= 0.7 else "badge-warn" if score >= 0.4 else "badge-bad"
    provider = (llm or {}).get("provider", "local")
    used = "LLM" if (llm or {}).get("used") else "deterministic"
    invalid = grounding.get("invalid_source_ids", []) + grounding.get("unverified_path_refs", [])
    st.markdown(
        f'<span class="badge {css}">reference checks {score:.2f}</span><span class="badge">{escape(provider)} · {used}</span>'
        f'<span class="badge">{len(grounding.get("cited_sources", []))} cited sources</span>'
        + (f'<span class="badge badge-bad">{len(invalid)} unverified refs</span>' if invalid else ""),
        unsafe_allow_html=True,
    )
    if (llm or {}).get("reason") and not (llm or {}).get("used"):
        st.caption(f"LLM not used: {llm['reason']}")
    st.caption("Reference checks validate visible source locations. Whether each cited source supports the claim is unassessed.")


# --- header metrics ------------------------------------------------------------------------------------
st.subheader(identity.get("project_name", "Repository"))
st.write(identity.get("one_line_summary", ""))
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Assessed health", f"{quality.get('total', 0)}/{quality.get('assessed_max', 100)}", quality.get("grade", ""))
m2.metric("Files indexed", ingestion.get("files_indexed", "—"))
m3.metric("Chunks", ingestion.get("chunks", "—"))
drift_accuracy = insights.get("docs_drift", {}).get("docs_accuracy")
m4.metric("Docs accuracy", f"{drift_accuracy:.0%}" if isinstance(drift_accuracy, (int, float)) else "n/a")
vulns = insights.get("dependencies", {}).get("vulnerabilities", [])
deps_checked = insights.get("dependencies", {}).get("scan_status") == "ok"
m5.metric("Reported advisories", len(vulns) if deps_checked else "unknown", f"{sum(1 for v in vulns if v.get('reachable'))} imported" if vulns else None, delta_color="inverse")
if ingestion.get("commit_sha"):
    st.caption(f"Commit `{ingestion['commit_sha'][:10]}` · embeddings `{ingestion.get('embedding_model')}` · parsers {ingestion.get('parser_backends')}")
if ingestion.get("source_kind") == "local":
    st.caption("Source: a local checkout. The repository id is a local namespace; it is not a public GitHub repository link.")
# Analyses cached before coverage reporting have no "coverage" key; the banner is simply skipped for them.
coverage = summary.get("coverage") or {}
banner = coverage_banner_text(coverage)
if banner:
    st.warning(banner, icon="⚠️")
    sample = coverage.get("sample_dropped_paths") or []
    if sample:
        with st.expander(f"Files left out of the analysis (sample of {len(sample)})"):
            st.code("\n".join(sample), language="text")
if coverage.get("duplicates_collapsed"):
    st.caption(f"{coverage['duplicates_collapsed']} byte-identical duplicate file(s) were indexed once and cited via their first copy.")

tabs = st.tabs(["Overview", "Architecture", "Run", "Health", "Insights", "Chat", "Detective", "Code viewer"])

with tabs[0]:
    # Analyses cached before the "about" block existed fall back to the older identity fields.
    about = identity.get("about") or {
        "what_it_is": identity.get("one_line_summary", ""),
        "why_it_exists": identity.get("what_this_repo_is_for", ""),
        "why_it_exists_source": None,
        "what_it_does": [],
        "who_it_is_for": identity.get("target", ""),
    }
    with st.container(border=True):
        st.markdown("### About this project")
        st.markdown(f"**What it is.** {about.get('what_it_is', '')}")
        st.markdown("**Why it exists.**")
        if about.get("why_it_exists"):
            if about.get("why_it_exists_source"):
                st.markdown(f"> {about['why_it_exists']}")
                st.caption(f"Quoted from `{about['why_it_exists_source']}`.")
            else:
                st.markdown(about["why_it_exists"])
        else:
            st.caption("The README does not state the project's purpose, so it is not claimed here. See *What it does* for what the code shows.")
        st.markdown("**What it does.**")
        if about.get("what_it_does"):
            for item in about["what_it_does"]:
                st.markdown(f"- {item['text']}")
            if about.get("what_it_does_basis") == "readme":
                st.caption("Listed in " + ", ".join(sorted({f"`{item['source']}`" for item in about["what_it_does"]})) + ".")
            else:
                st.caption("Inferred from the repository structure (routes, entrypoints, frameworks, config). The README has no feature list.")
        else:
            st.caption("No feature list was found. Re-analyze with **Force refresh** to infer one from the code.")
        st.markdown(f"**Who it's for.** Likely {about.get('who_it_is_for', '')}.")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Tech stack")
        table(summary.get("tech_stack", []))
    with right:
        st.markdown("#### Read these first")
        table(summary.get("important_files", [])[:12])
    st.markdown("#### How it likely works")
    for step in summary.get("how_it_likely_works", []):
        st.markdown(f"- {step}")

with tabs[1]:
    architecture = summary.get("architecture", {})
    st.write(architecture.get("explanation", ""))
    st.caption(architecture.get("entrypoint_flow_basis", "Suggested reading order inferred from repository structure."))
    for step in architecture.get("entrypoint_flow", []):
        st.markdown(f"- {step}")
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Most central modules (PageRank over imports + calls)")
        table(architecture.get("central_files", []))
    with right:
        st.markdown("#### Import cycles")
        cycles = architecture.get("dependency_cycles", [])
        if cycles:
            for cycle in cycles:
                st.markdown("- " + " → ".join(f"`{p}`" for p in cycle))
        else:
            st.caption("No circular dependencies detected.")
    st.markdown("#### Module dependency diagram")
    mermaid(architecture.get("dependency_mermaid") or st.session_state.graph.get("mermaid", "flowchart TD\nA[Repository]"))
    with st.expander("Interactive file dependency graph"):
        files = api("GET", f"/repo/{repo_id}/graph", params={"view": "files"})
        if files.ok:
            data = files.json()
            nodes = [{"id": n["id"], "label": n["id"].rsplit("/", 1)[-1], "title": n["id"], "value": n["degree"], "group": n["id"].split("/")[0]} for n in data["nodes"]]
            edges = [{"from": e["source"], "to": e["target"], "arrows": "to", "value": e["weight"]} for e in data["edges"]]
            components.html(
                f"""<div id="g" style="height:560px;border:1px solid #8884"></div>
                <script src="https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
                <script>new vis.Network(document.getElementById('g'), {{nodes: new vis.DataSet({json.dumps(nodes)}), edges: new vis.DataSet({json.dumps(edges)})}},
                {{nodes: {{shape: 'dot', scaling: {{min: 6, max: 30}}, font: {{size: 12}}}}, physics: {{stabilization: true}}, interaction: {{hover: true}}}});</script>""",
                height=580,
            )
    with st.expander("Mermaid source"):
        st.code(architecture.get("dependency_mermaid", ""), language="text")

with tabs[2]:
    official_docs = facts.get("run_doc_paths") or facts.get("install_doc_paths") or facts.get("official_doc_paths", [])
    if official_docs:
        st.markdown("Official instructions found in: " + ", ".join(f"`{p}`" for p in official_docs))
    else:
        st.warning("No README/INSTALL-style documentation was detected; commands below may be inferred.")
    run_rows = summary.get("run_instructions", [])
    table([{k: row.get(k) for k in ("command", "source", "start_line", "heading", "confidence", "official")} for row in run_rows], "No run commands detected.")
    if st.button("Ask the assistant for step-by-step run instructions"):
        with st.spinner("Composing a grounded answer…"):
            response = api("POST", f"/repo/{repo_id}/chat", json={"question": "How do I install and run this repository locally? Give exact commands and cite the files."}, timeout=180)
            st.session_state.run_answer = response.json() if response.ok else {"answer": error_detail(response, response.reason)}
    if st.session_state.run_answer:
        grounding_badge(st.session_state.run_answer.get("grounding"), st.session_state.run_answer.get("llm"), st.session_state.run_answer.get("evidence"))
        st.markdown(st.session_state.run_answer.get("answer", ""))
        render_citations(st.session_state.run_answer.get("citations", []), "run")
    if facts.get("helper_scripts"):
        with st.expander("Helper / developer scripts"):
            table(facts["helper_scripts"])

with tabs[3]:
    st.markdown(f"#### Assessed health: {quality.get('total', 0)}/{quality.get('assessed_max', 100)}")
    st.caption(quality.get("method", ""))
    if quality.get("assessment_coverage", 1) < 1:
        st.warning(f"Assessment is incomplete. Possible total: {quality.get('possible_total')}. Unavailable checks are not clean results.")
    table([{"category": key, "score": f"{value['score']}/{value.get('assessed_max', value.get('max', 20))}", "signals": value["reason"]} for key, value in quality.get("categories", {}).items()])
    st.markdown("#### Static findings")
    if st.session_state.findings_repo != repo_id:
        try:
            response = api("GET", f"/repo/{repo_id}/weaknesses")
            payload = response.json() if response.ok else None
            st.session_state.findings = payload if isinstance(payload, list) else summary.get("risks", [])
            st.session_state.findings_repo = repo_id
            if not response.ok:
                st.warning("Could not load the full findings list; showing summary findings.")
        except requests.RequestException:
            st.warning("Could not load the full findings list; showing summary findings.")
            st.session_state.findings = summary.get("risks", [])
    findings = st.session_state.findings
    severity = st.selectbox("Finding severity", ["all", "critical", "high", "medium", "low"])
    category = st.selectbox("Finding category", ["all"] + sorted({f.get("category", "other") for f in findings}))
    filtered = [f for f in findings if (severity == "all" or f.get("severity") == severity)
                and (category == "all" or f.get("category") == category)]
    for index, finding in enumerate(filtered):
        with st.expander(f"{finding.get('severity', '').upper()} · {finding.get('title')}"):
            st.write(finding.get("why_it_matters", ""))
            st.code(finding.get("evidence") or "No source excerpt recorded.")
            st.write(f"Suggested fix: {finding.get('suggested_fix', '')}")
            if finding.get("file_path") and st.button("View finding source", key=f"finding-{index}"):
                open_file(finding["file_path"], finding.get("line_number") or 1,
                          run_id=ingestion.get("run_id"))
    if not filtered:
        st.caption("No findings match these filters. This is an automated static review.")

with tabs[4]:
    drift = insights.get("docs_drift", {})
    st.markdown("#### 📚 Docs vs code drift")
    st.caption("Commands, files, env vars, endpoints and links the documentation mentions, checked against the actual repository.")
    table([{k: issue.get(k) for k in ("severity", "title", "doc_path", "doc_line", "evidence")} for issue in drift.get("top_issues", [])], "Documentation matches the code for every check.")
    deps = insights.get("dependencies", {})
    st.markdown("#### 🛡️ Dependency vulnerabilities (OSV.dev)")
    st.caption(f"Scan status: {deps.get('scan_status')} · {deps.get('packages_scanned', 0)} versions checked / {deps.get('inventory_packages', 0)} inventoried · {deps.get('unpinned_count', 0)} unpinned. {deps.get('note', '')}")
    table([{"package": v["package"], "version": v["version"], "id": v["id"], "severity": v["severity"], "fixed in": ", ".join(v.get("fixed_versions", [])), "imported by repo code": v["reachable"], "summary": v.get("summary", "")[:120]} for v in vulns],
          "No advisories returned for the checked pinned versions." if deps_checked else "Dependency vulnerability status is unknown; the scan did not complete.")
    history = insights.get("history", {})
    st.markdown("#### 🔥 Change hotspots (git history × complexity)")
    st.caption(f"{history.get('commits_analyzed', 0)} commits analyzed · bus factor: {history.get('bus_factor') or 'n/a'}. {history.get('note', '')}")
    table(history.get("hotspots", []), "No history available.")
    code_health = insights.get("code_health", {})
    left, right = st.columns(2)
    with left:
        st.markdown("#### 🧩 Most complex functions")
        table(code_health.get("complex_functions", []), "No highly complex functions.")
    with right:
        st.markdown("#### 🪦 Unreferenced private functions")
        table(code_health.get("unreferenced_private_functions", []), "None detected.")

with tabs[5]:
    for index, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                grounding_badge(message.get("grounding"), message.get("llm"), message.get("evidence"))
                with st.expander("Sources", expanded=index == len(st.session_state.chat_history) - 1):
                    render_citations(message.get("citations", []), f"chat-{index}")
    question = st.chat_input("Ask about architecture, a function, run commands, risks…")
    if question:
        history = [{"role": m["role"], "content": m["content"][:4000]} for m in st.session_state.chat_history[-6:]]
        st.session_state.chat_history.append({"role": "user", "content": question})
        try:
            with st.spinner("Retrieving evidence and composing a grounded answer…"):
                response = api("POST", f"/repo/{repo_id}/chat", json={"question": question, "history": history}, timeout=180)
            payload = response.json() if response.ok else {"answer": f"Chat failed: {error_detail(response, response.reason)}", "citations": []}
        except requests.RequestException as exc:
            payload = {"answer": f"Chat failed: {exc}", "citations": []}
        st.session_state.chat_history.append({"role": "assistant", **payload, "content": payload.get("answer", "")})
        st.session_state.last_citations = payload.get("citations", [])
        st.rerun()

with tabs[6]:
    st.markdown("#### 🕵️ Repo Detective")
    st.caption("Multi-step investigation: the agent searches, reads code, follows callers/callees and cites only lines it actually inspected.")
    detective_question = st.text_input("Investigation", placeholder="e.g. How does a request to POST /orders reach the payment provider?")
    steps = st.slider("Max tool steps", 1, 10, 6)
    if st.button("Investigate", disabled=not detective_question):
        with st.spinner("Investigating…"):
            response = api("POST", f"/repo/{repo_id}/investigate", json={"question": detective_question, "max_steps": steps}, timeout=300)
            st.session_state.detective = response.json() if response.ok else {"answer": error_detail(response, response.reason), "steps": []}
    result = st.session_state.detective
    if result:
        st.caption(f"Mode: {result.get('mode')} · {result.get('notes', '')}")
        grounding_badge(result.get("grounding"), result.get("llm") or {"provider": "deterministic", "used": False})
        st.markdown(result.get("answer", ""))
        with st.expander(f"Investigation trace ({len(result.get('steps', []))} steps)"):
            for step in result.get("steps", []):
                st.markdown(f"**{step['step']}. `{step['tool']}`** `{json.dumps(step['args'])}` → {step['result']}" + (f"  \n_{step['thought']}_" if step.get("thought") and step["thought"] != "deterministic plan" else ""))
        with st.expander("Evidence inspected"):
            table(result.get("evidence", []))

with tabs[7]:
    view = st.session_state.file_view
    path_col, start_col, end_col, go_col = st.columns([5, 1, 1, 1])
    path = path_col.text_input("File path", value=(view or {}).get("path", ""))
    start = start_col.number_input("From", min_value=1, value=int((view or {}).get("start_line") or 1))
    end = end_col.number_input("To", min_value=1, value=int((view or {}).get("end_line") or 80))
    if go_col.button("Show", disabled=not path):
        open_file(path, int(start), int(end))
        view = st.session_state.file_view
    if view:
        if view.get("error"):
            st.error(view["error"])
        else:
            st.caption(f"`{view['path']}` lines {view['start_line']}-{view['end_line']} of {view['total_lines']}")
            language = {"py": "python", "ts": "typescript", "js": "javascript", "md": "markdown", "json": "json", "yml": "yaml", "yaml": "yaml"}.get(view["path"].rsplit(".", 1)[-1], None)
            st.code(view["text"], language=language, line_numbers=True)
    else:
        st.caption("Click **Open** on any citation, or enter a path above.")
