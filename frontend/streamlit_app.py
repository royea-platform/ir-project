"""Streamlit frontend for the IR Search Engine."""

import streamlit as st
import httpx
import pandas as pd

GATEWAY_URL = "http://localhost:8000"

st.set_page_config(page_title="IR Search Engine", page_icon="🔍", layout="wide")
st.title("🔍 Information Retrieval Search Engine")

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")

    dataset = st.selectbox("Dataset", ["quora", "touche"], help="Select corpus to search")
    mode = st.radio("Execution Mode", ["basic", "basic+extra"],
                    help="basic+extra enables query refinement")

    st.subheader("BM25 Parameters")
    bm25_k1 = st.slider("k1 (term frequency saturation)", 0.0, 3.0, 1.5, 0.1)
    bm25_b = st.slider("b (document length normalization)", 0.0, 1.0, 0.75, 0.05)

    top_k = st.slider("Top-K Results", 5, 50, 10, 5)

    # Hybrid settings
    st.subheader("Hybrid Settings")
    hybrid_mode = st.radio("Hybrid Mode", ["serial", "parallel"])
    fusion_method = st.selectbox("Fusion Method", ["rrf", "weighted", "combmnz"])
    fusion_weights = {"bm25": 0.5, "dense": 0.5}
    if fusion_method == "weighted":
        bm25_w = st.slider("BM25 Weight", 0.0, 1.0, 0.5, 0.1)
        dense_w = st.slider("Dense Weight", 0.0, 1.0, 0.5, 0.1)
        fusion_weights = {"bm25": bm25_w, "dense": dense_w}

# ── Document viewer ──────────────────────────────────────────────────────────

if "view_doc" in st.session_state and st.session_state.view_doc:
    doc_info = st.session_state.view_doc
    st.markdown("---")
    st.subheader(f"📄 Document: {doc_info['doc_id']}")

    with st.spinner("Loading full document..."):
        try:
            resp = httpx.get(
                f"{GATEWAY_URL}/doc/{doc_info['dataset']}/{doc_info['doc_id']}",
                timeout=30.0,
            )
            resp.raise_for_status()
            doc = resp.json()

            if doc.get("error"):
                st.error(doc["error"])
            else:
                if doc.get("title"):
                    st.markdown(f"### {doc['title']}")

                meta_cols = st.columns(4)
                meta_cols[0].markdown(f"**Doc ID:** `{doc['doc_id']}`")
                meta_cols[1].markdown(f"**Dataset:** `{doc.get('dataset', doc_info['dataset'])}`")
                if doc.get("doc_number"):
                    meta_cols[2].markdown(f"**Doc #:** `{doc['doc_number']}` / `{doc.get('total_docs', '?')}`")
                meta_cols[3].markdown(f"**Score:** `{doc_info.get('score', 'N/A')}`  |  **Rank:** `{doc_info.get('rank', 'N/A')}`")

                st.markdown("---")
                st.markdown("#### Document Content")
                st.text_area("", value=doc["text"], height=400, disabled=True)

                st.download_button(
                    label="⬇️ Download Document",
                    data=doc["text"],
                    file_name=f"{doc['doc_id']}.txt",
                    mime="text/plain",
                )

        except Exception as e:
            st.error(f"Failed to load document: {e}")

    if st.button("← Back to results"):
        st.session_state.view_doc = None
        st.rerun()

    st.stop()

# ── Main tabs ────────────────────────────────────────────────────────────────

tab_search, tab_compare, tab_eval, tab_clusters, tab_federated = st.tabs([
    "🔍 Search", "⚖️ Compare Methods", "📊 Evaluation Dashboard",
    "🗂️ Clusters & Topics", "🌐 Federated Search",
])


def _do_search(q, ds, m, repr_t, tk, k1, b, hm, fm, fw):
    """Run a search and return results dict."""
    resp = httpx.post(f"{GATEWAY_URL}/search", json={
        "dataset": ds, "query": q, "mode": m, "repr_type": repr_t,
        "top_k": tk, "bm25_k1": k1, "bm25_b": b,
        "hybrid_mode": hm, "fusion_method": fm, "fusion_weights": fw,
    }, timeout=120.0)
    resp.raise_for_status()
    return resp.json()


def _display_results(data, ds, prefix=""):
    """Display search results."""
    if data.get("refined_query"):
        st.info(f"Query refined: **{data['query']}** → **{data['refined_query']}**")

    for doc in data.get("results", []):
        doc_id = doc["doc_id"]
        title = doc.get("title") or doc_id
        score = doc["score"]
        rank = doc["rank"]
        snippet = doc.get("text", "")[:300]

        col_info, col_btn = st.columns([5, 1])
        with col_info:
            st.markdown(f"**#{rank}** — **{title}**  \n`score: {score:.4f}` | `{doc_id}`")
            st.caption(snippet + ("..." if len(doc.get("text", "")) > 300 else ""))
        with col_btn:
            if st.button("📄", key=f"{prefix}view_{doc_id}_{rank}"):
                st.session_state.view_doc = {
                    "doc_id": doc_id, "dataset": ds, "score": score, "rank": rank,
                }
                st.rerun()
        st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Single Search
# ══════════════════════════════════════════════════════════════════════════════

with tab_search:
    repr_type = st.selectbox("Representation", ["tfidf", "bm25", "dense", "hybrid"], key="search_repr")
    query = st.text_input("Enter your query:", placeholder="Type a question or keywords...", key="search_q")

    if st.button("🔍 Search", type="primary", width="stretch", key="search_btn"):
        if query:
            with st.spinner("Searching..."):
                try:
                    data = _do_search(query, dataset, mode, repr_type, top_k,
                                      bm25_k1, bm25_b, hybrid_mode, fusion_method, fusion_weights)
                    st.session_state.search_results = data
                    st.session_state.search_dataset = dataset
                except httpx.ConnectError:
                    st.error("Cannot connect to the gateway. Start the services first: `make run` (or `python app.py`).")
                except Exception as e:
                    st.error(f"Search failed: {e}")

    if "search_results" in st.session_state and st.session_state.get("search_results"):
        data = st.session_state.search_results
        st.subheader(f"Results — {data['repr_type'].upper()} on {data['dataset']} ({len(data['results'])} docs)")
        _display_results(data, st.session_state.get("search_dataset", dataset), prefix="s_")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Compare Methods (same query → all 4 methods side by side)
# ══════════════════════════════════════════════════════════════════════════════

with tab_compare:
    st.markdown("### Compare all retrieval methods on the same query")
    st.caption("Runs TF-IDF, BM25, Dense, and Hybrid simultaneously so you can see how results differ.")

    compare_query = st.text_input("Query to compare:", placeholder="e.g., how to learn programming", key="cmp_q")

    if st.button("⚖️ Compare All Methods", type="primary", width="stretch", key="cmp_btn"):
        if compare_query:
            methods = ["tfidf", "bm25", "dense", "hybrid"]
            results = {}
            progress = st.progress(0, text="Running comparisons...")

            for i, method in enumerate(methods):
                progress.progress((i + 1) / len(methods), text=f"Running {method.upper()}...")
                try:
                    data = _do_search(compare_query, dataset, mode, method, top_k,
                                      bm25_k1, bm25_b, hybrid_mode, fusion_method, fusion_weights)
                    results[method] = data
                except Exception as e:
                    results[method] = {"error": str(e), "results": []}

            progress.empty()
            st.session_state.compare_results = results
            st.session_state.compare_dataset = dataset

    if "compare_results" in st.session_state and st.session_state.get("compare_results"):
        results = st.session_state.compare_results
        ds = st.session_state.get("compare_dataset", dataset)

        # Summary table: which docs appear in top-K for each method
        st.markdown("#### 📋 Results Overlap")
        all_doc_ids = {}
        for method, data in results.items():
            if "error" not in data:
                for doc in data.get("results", []):
                    did = doc["doc_id"]
                    if did not in all_doc_ids:
                        all_doc_ids[did] = {"doc_id": did, "title": doc.get("title") or did}
                    all_doc_ids[did][method] = doc["rank"]

        if all_doc_ids:
            overlap_df = pd.DataFrame(all_doc_ids.values())
            for m in ["tfidf", "bm25", "dense", "hybrid"]:
                if m not in overlap_df.columns:
                    overlap_df[m] = None
            overlap_df = overlap_df[["doc_id", "title", "tfidf", "bm25", "dense", "hybrid"]]
            overlap_df = overlap_df.sort_values(
                by=["tfidf", "bm25", "dense", "hybrid"],
                na_position="last"
            ).head(20)

            st.dataframe(
                overlap_df.rename(columns={
                    "doc_id": "Doc ID", "title": "Title",
                    "tfidf": "TF-IDF Rank", "bm25": "BM25 Rank",
                    "dense": "Dense Rank", "hybrid": "Hybrid Rank",
                }),
                width="stretch", hide_index=True,
            )

            # Count docs found by multiple methods
            overlap_df["methods_found"] = overlap_df[["tfidf", "bm25", "dense", "hybrid"]].notna().sum(axis=1)
            found_by_all = (overlap_df["methods_found"] == 4).sum()
            found_by_3 = (overlap_df["methods_found"] == 3).sum()
            found_by_2 = (overlap_df["methods_found"] == 2).sum()
            found_by_1 = (overlap_df["methods_found"] == 1).sum()

            st.markdown(f"**Overlap:** {found_by_all} docs in all 4 | {found_by_3} in 3 | {found_by_2} in 2 | {found_by_1} unique to 1 method")

        # Side-by-side results
        st.markdown("#### 📊 Side-by-Side Results")
        col1, col2 = st.columns(2)

        method_names = {"tfidf": "TF-IDF", "bm25": "BM25", "dense": "Dense Embeddings", "hybrid": "Hybrid"}

        for i, (method, data) in enumerate(results.items()):
            target_col = col1 if i % 2 == 0 else col2
            with target_col:
                st.markdown(f"##### {method_names.get(method, method)}")
                if "error" in data:
                    st.error(data["error"])
                else:
                    for doc in data.get("results", [])[:5]:
                        title = doc.get("title") or doc["doc_id"]
                        st.markdown(f"`#{doc['rank']}` **{title}** — score: {doc['score']:.4f}")
                        st.caption(doc.get("text", "")[:150] + "...")
                    st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Evaluation Dashboard
# ══════════════════════════════════════════════════════════════════════════════

with tab_eval:
    st.markdown("### 📊 Evaluation Dashboard")
    st.caption("Run evaluation on test queries with official qrels. Compare metrics across methods and datasets.")

    eval_col1, eval_col2 = st.columns(2)
    with eval_col1:
        eval_dataset = st.selectbox("Dataset", ["quora", "touche"], key="eval_ds")
    with eval_col2:
        num_queries = st.selectbox("Number of test queries", [10, 25, 50, 100], index=1, key="eval_nq")

    # Load saved results if available
    import os
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "evaluation_results.csv")

    if os.path.exists(csv_path):
        st.markdown("#### 📈 Saved Evaluation Results")
        saved_df = pd.read_csv(csv_path)
        st.dataframe(saved_df, width="stretch", hide_index=True)

        # Bar chart
        if not saved_df.empty:
            metric_cols = [c for c in saved_df.columns if c not in ["dataset", "repr_type"]]
            if metric_cols:
                chart_metric = st.selectbox("Metric to chart:", metric_cols, key="chart_m")
                chart_data = saved_df.pivot(index="repr_type", columns="dataset", values=chart_metric)
                st.bar_chart(chart_data)

    st.markdown("---")

    if st.button("🚀 Run Full Evaluation (all methods)", type="primary", key="eval_all_btn"):
        methods = ["tfidf", "bm25", "dense", "hybrid"]
        eval_results = []
        progress = st.progress(0, text="Evaluating...")

        for i, method in enumerate(methods):
            progress.progress((i + 1) / len(methods), text=f"Evaluating {method.upper()} on {eval_dataset}...")
            try:
                resp = httpx.post(f"{GATEWAY_URL}/evaluate", json={
                    "dataset": eval_dataset,
                    "repr_type": method,
                    "mode": mode,
                    "top_k": 100,
                    "bm25_k1": bm25_k1,
                    "bm25_b": bm25_b,
                    "hybrid_mode": hybrid_mode,
                    "fusion_method": fusion_method,
                    "fusion_weights": fusion_weights,
                    "num_queries": num_queries,
                }, timeout=600.0)
                resp.raise_for_status()
                data = resp.json()
                metrics = data.get("metrics", {})
                eval_results.append({"method": method.upper(), **metrics})
            except Exception as e:
                eval_results.append({"method": method.upper(), "error": str(e)})

        progress.empty()

        if eval_results:
            eval_df = pd.DataFrame(eval_results)
            st.session_state.eval_df = eval_df

    if "eval_df" in st.session_state and st.session_state.get("eval_df") is not None:
        eval_df = st.session_state.eval_df
        st.markdown(f"#### Results for **{eval_dataset}** ({num_queries} queries)")
        st.dataframe(eval_df, width="stretch", hide_index=True)

        # Highlight best per metric
        metric_cols = [c for c in eval_df.columns if c not in ["method", "error"]]
        if metric_cols:
            st.markdown("#### 🏆 Best method per metric:")
            best_cols = st.columns(len(metric_cols))
            for i, metric in enumerate(metric_cols):
                try:
                    best_idx = eval_df[metric].astype(float).idxmax()
                    best_method = eval_df.loc[best_idx, "method"]
                    best_val = eval_df.loc[best_idx, metric]
                    best_cols[i].metric(metric, f"{best_val:.4f}", delta=best_method)
                except Exception:
                    pass

            # Bar chart comparison
            chart_df = eval_df.set_index("method")[metric_cols].astype(float)
            st.bar_chart(chart_df)

    # Single method evaluation
    st.markdown("---")
    st.markdown("#### Evaluate single configuration")
    eval_repr = st.selectbox("Method", ["tfidf", "bm25", "dense", "hybrid"], key="eval_single_repr")

    if st.button("📊 Evaluate", key="eval_single_btn"):
        with st.spinner(f"Evaluating {eval_repr}..."):
            try:
                resp = httpx.post(f"{GATEWAY_URL}/evaluate", json={
                    "dataset": eval_dataset,
                    "repr_type": eval_repr,
                    "mode": mode,
                    "top_k": 100,
                    "bm25_k1": bm25_k1,
                    "bm25_b": bm25_b,
                    "hybrid_mode": hybrid_mode,
                    "fusion_method": fusion_method,
                    "fusion_weights": fusion_weights,
                    "num_queries": num_queries,
                }, timeout=600.0)
                resp.raise_for_status()
                data = resp.json()
                metrics = data.get("metrics", {})

                if metrics:
                    cols = st.columns(len(metrics))
                    for i, (metric, value) in enumerate(metrics.items()):
                        cols[i].metric(metric, f"{value:.4f}")

            except httpx.ConnectError:
                st.error("Cannot connect to API. Run `python app.py` first.")
            except Exception as e:
                st.error(f"Evaluation failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Clusters & Topics  (bonus #15 clustering + #17 topic detection)
# ══════════════════════════════════════════════════════════════════════════════

with tab_clusters:
    st.markdown("### 🗂️ Document Clusters & Detected Topics")
    st.caption("KMeans over document embeddings groups the corpus; the top TF-IDF "
               "terms per cluster are its detected topic. Build with `make build-clusters`.")

    if st.button("Load clusters", type="primary", key="clusters_btn"):
        try:
            resp = httpx.get(f"{GATEWAY_URL}/clusters/{dataset}", timeout=60.0)
            resp.raise_for_status()
            st.session_state.clusters = resp.json()
            st.session_state.clusters_ds = dataset
        except httpx.HTTPStatusError as e:
            st.error(f"{e.response.status_code}: {e.response.json().get('detail', e)}")
        except httpx.ConnectError:
            st.error("Cannot connect to the gateway. Run `make run` first.")
        except Exception as e:
            st.error(f"Failed to load clusters: {e}")

    if st.session_state.get("clusters"):
        data = st.session_state.clusters
        cds = st.session_state.get("clusters_ds", dataset)
        st.markdown(f"**{data['n_clusters']} clusters** on `{cds}`")

        rows = [{"Cluster": c["cluster_id"], "Topic": ", ".join(c["topic"]), "Docs": c["size"]}
                for c in data["clusters"]]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        ids = [c["cluster_id"] for c in data["clusters"]]
        sel = st.selectbox("Inspect cluster:", ids, key="cluster_sel",
                           format_func=lambda i: f"#{i}")
        if st.button("Show sample docs", key="cluster_docs_btn"):
            try:
                r = httpx.get(f"{GATEWAY_URL}/cluster/{cds}/{sel}",
                              params={"limit": 10}, timeout=60.0)
                r.raise_for_status()
                cd = r.json()
                st.markdown(f"#### Cluster #{sel} — topic: *{', '.join(cd['topic'])}*")
                for doc in cd["docs"]:
                    title = doc.get("title") or doc["doc_id"]
                    st.markdown(f"**{title}**  \n`{doc['doc_id']}`")
                    st.caption(doc.get("text", "")[:200] + "...")
                    st.markdown("---")
            except Exception as e:
                st.error(f"Failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: Federated Search  (bonus #14 distributed information retrieval)
# ══════════════════════════════════════════════════════════════════════════════

with tab_federated:
    st.markdown("### 🌐 Federated (Distributed) Search")
    st.caption("Query every dataset shard at once. Per-shard scores are min-max "
               "normalized so they can be merged into a single ranked list.")

    fed_datasets = st.multiselect("Datasets (shards)", ["quora", "touche"],
                                  default=["quora", "touche"], key="fed_ds")
    fed_repr = st.selectbox("Representation", ["tfidf", "bm25", "dense", "hybrid"], key="fed_repr")
    fed_query = st.text_input("Query:", placeholder="searches all selected shards...", key="fed_q")

    if st.button("🌐 Federated Search", type="primary", width="stretch", key="fed_btn"):
        if fed_query and fed_datasets:
            with st.spinner("Searching all shards..."):
                try:
                    resp = httpx.post(f"{GATEWAY_URL}/search_distributed", json={
                        "query": fed_query, "datasets": fed_datasets, "mode": mode,
                        "repr_type": fed_repr, "top_k": top_k,
                        "bm25_k1": bm25_k1, "bm25_b": bm25_b,
                        "hybrid_mode": hybrid_mode, "fusion_method": fusion_method,
                        "fusion_weights": fusion_weights,
                    }, timeout=180.0)
                    resp.raise_for_status()
                    st.session_state.fed_results = resp.json()
                except httpx.ConnectError:
                    st.error("Cannot connect to the gateway. Run `make run` first.")
                except Exception as e:
                    st.error(f"Federated search failed: {e}")

    if st.session_state.get("fed_results"):
        data = st.session_state.fed_results
        split = " | ".join(f"{k}: {v}" for k, v in data.get("per_dataset", {}).items())
        st.markdown(f"**{len(data['results'])} merged results** — shard split: {split}")
        for doc in data["results"]:
            title = doc.get("title") or doc["doc_id"]
            st.markdown(f"**#{doc['rank']}** — **{title}**  "
                        f"`[{doc.get('dataset', '?')}]`  \n"
                        f"`norm score: {doc['score']:.4f}` | `{doc['doc_id']}`")
            st.caption(doc.get("text", "")[:250] + "...")
            st.markdown("---")
