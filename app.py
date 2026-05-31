"""
LegalSetu Learn — Educational RAG-Powered Explorer for Indian MSME Law

Architecture: Strict RAG (Retrieval-Augmented Generation)
- User queries are answered ONLY from a curated knowledge base
- Claude API generates responses grounded in retrieved chunks
- Citations are shown for every response
- Mandatory agreement gate prevents misuse

This is an OPEN SOURCE EDUCATIONAL TOOL for law students and curious learners.
NOT FOR REAL LEGAL DECISIONS.
"""

import streamlit as st
import os
from datetime import datetime
from anthropic import Anthropic

from rag_engine import (
    build_index,
    retrieve,
    format_retrieved_context,
    get_collection_stats,
)
from prompts import STRICT_RAG_PROMPT, STRUCTURED_TOPICS

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="LegalSetu Learn — Educational MSME Law Explorer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Default model — Haiku is cheap and sufficient for grounded RAG
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


# =============================================================================
# ANTHROPIC CLIENT
# =============================================================================

def get_anthropic_client():
    """Initialize Anthropic client from secrets or env."""
    api_key = None
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except (FileNotFoundError, KeyError):
        api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        st.error(
            "⚠️ ANTHROPIC_API_KEY not found. "
            "Set it in `.streamlit/secrets.toml` (local) or "
            "Streamlit Cloud Secrets (deployed)."
        )
        st.stop()
    return Anthropic(api_key=api_key)


# =============================================================================
# RAG QUERY HANDLER
# =============================================================================

def answer_with_rag(user_query: str, model: str = DEFAULT_MODEL) -> dict:
    """
    Run the full RAG pipeline:
    1. Retrieve relevant chunks
    2. Format context
    3. Call Claude with strict RAG prompt
    4. Return response with sources

    Returns dict with: answer, sources, retrieved_chunks
    """
    client = get_anthropic_client()

    # Step 1: Retrieve
    retrieved = retrieve(user_query, top_k=4)

    if not retrieved:
        return {
            "answer": "❌ The knowledge base appears to be empty. Please build the index first by running `python rag_engine.py` from the terminal, or click the 'Rebuild Knowledge Base' button in the sidebar.",
            "sources": [],
            "retrieved_chunks": [],
        }

    # Step 2: Format context
    context_block = format_retrieved_context(retrieved)

    # Step 3: Build user message — context must be the bulk
    user_message = f"""=== RETRIEVED EDUCATIONAL CONTEXT (your ONLY source of truth) ===

{context_block}

=== END OF RETRIEVED CONTEXT ===

USER'S QUESTION: {user_query}

Answer the question using ONLY the retrieved context above. If the context doesn't contain enough information to answer, refuse as specified in your instructions. Always cite sources at the end."""

    # Step 4: Call Claude
    try:
        with st.spinner("🔍 Retrieving from knowledge base and generating grounded answer..."):
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                system=STRICT_RAG_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            answer = response.content[0].text
    except Exception as e:
        return {
            "answer": f"❌ Error calling Claude API: {str(e)}",
            "sources": [],
            "retrieved_chunks": retrieved,
        }

    # Compile unique sources used
    sources = []
    seen = set()
    for chunk in retrieved:
        key = (chunk["source"], chunk["heading"])
        if key not in seen:
            seen.add(key)
            sources.append(chunk)

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved,
    }


# =============================================================================
# AGREEMENT GATE (Mandatory Click-Through)
# =============================================================================

def show_agreement_gate():
    """Mandatory agreement before user can access the tool."""
    st.title("📚 LegalSetu Learn")
    st.subheader("Educational Explorer for Indian MSME Law")

    st.markdown("---")

    st.markdown("""
    ### Before you proceed, please understand:

    This is an **educational tool** designed for:
    - 📖 **Law students** studying Indian commercial law
    - 🔍 **Curious learners** wanting to understand MSME legal concepts
    - 💻 **Technologists** interested in RAG architecture

    ### What this tool IS:
    - A retrieval-based educational explainer
    - Powered by a curated knowledge base of public statutes
    - Built using strict RAG (Retrieval-Augmented Generation)
    - Open source under MIT License

    ### What this tool is NOT:
    - ❌ A legal advisor
    - ❌ A tax consultant
    - ❌ A substitute for an advocate or Chartered Accountant
    - ❌ Capable of handling your specific legal situation
    """)

    st.markdown("---")
    st.markdown("### Mandatory Acknowledgements")

    ack1 = st.checkbox(
        "I understand this is an **educational tool only**, not legal or tax advice.",
        key="ack1",
    )
    ack2 = st.checkbox(
        "I understand the AI **may produce errors** despite its safeguards, "
        "and I will verify against primary sources.",
        key="ack2",
    )
    ack3 = st.checkbox(
        "I will **consult a qualified professional** (advocate / CA) before any legal or tax decision.",
        key="ack3",
    )
    ack4 = st.checkbox(
        "I will **NOT use the outputs** to directly file, submit, or rely upon any legal/tax document "
        "(notices, replies, contracts).",
        key="ack4",
    )
    ack5 = st.checkbox(
        "I understand that in 2025, the Gujarat High Court reprimanded a GST officer for relying on "
        "AI-generated fabricated case laws. **AI-generated legal content must always be verified.**",
        key="ack5",
    )

    all_acknowledged = all([ack1, ack2, ack3, ack4, ack5])

    if st.button("Proceed to Educational Tool", type="primary", disabled=not all_acknowledged):
        if all_acknowledged:
            st.session_state.agreement_accepted = True
            st.rerun()
        else:
            st.warning("Please check all acknowledgements to proceed.")

    if not all_acknowledged:
        st.info("👆 Please check all 5 acknowledgements above to enable the 'Proceed' button.")


# =============================================================================
# MAIN APP (After Agreement)
# =============================================================================

def show_main_app():
    """The main RAG application after agreement is accepted."""

    # Sidebar
    with st.sidebar:
        st.title("📚 LegalSetu Learn")
        st.caption("Educational RAG explorer for Indian MSME law")

        st.markdown("---")

        # Knowledge base stats
        st.markdown("### 📊 Knowledge Base")
        stats = get_collection_stats()
        if stats["total_chunks"] == 0:
            st.warning("⚠️ Knowledge base not built yet.")
            if st.button("🔨 Build Index Now"):
                with st.spinner("Building vector index... (one-time setup, ~30 sec)"):
                    count, msg = build_index(force_rebuild=False)
                st.success(msg)
                st.rerun()
        else:
            st.success(f"✅ Indexed: {stats['total_chunks']} chunks")
            st.caption(f"From {stats['unique_sources']} verified documents")
            with st.expander("📁 Source documents"):
                for src in stats['source_files']:
                    st.markdown(f"- `{src}`")

            if st.button("🔄 Rebuild Index"):
                with st.spinner("Rebuilding..."):
                    count, msg = build_index(force_rebuild=True)
                st.success(msg)
                st.rerun()

        st.markdown("---")

        st.markdown("### ⚙️ Model")
        model_choice = st.selectbox(
            "AI Model",
            [
                "claude-haiku-4-5-20251001",
                "claude-sonnet-4-6",
            ],
            help="Haiku is cheaper. Sonnet may give more nuanced explanations.",
        )

        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown(
            "Open-source RAG demonstration. All answers come from a curated knowledge base of "
            "Indian statutes and educational case studies. Claude is instructed to refuse if "
            "context is insufficient."
        )

        st.markdown("### 🔗 Primary Sources")
        st.markdown(
            "- [MSMED Act 2006 (PDF)](https://msme.gov.in/sites/default/files/MSMED2006act.pdf)\n"
            "- [CGST Act 2017 (PDF)](https://www.cbic.gov.in/resources//htdocs-cbec/gst/CGST-act-updated.pdf)\n"
            "- [Indian Contract Act 1872](https://legislative.gov.in/sites/default/files/A1872-09.pdf)\n"
            "- [MSME ODR Portal](https://msmeodr.gov.in)\n"
            "- [Indian Kanoon (Case Law)](https://indiankanoon.org)"
        )

        st.markdown("### 💻 Project")
        st.markdown("[GitHub Repo](https://github.com/yourusername/legalsetu-learn)")
        st.caption("MIT License | Built with Claude API + ChromaDB")

        st.markdown("---")
        if st.button("🚪 Reset & Logout"):
            st.session_state.clear()
            st.rerun()

    # Main content
    st.title("📚 LegalSetu Learn")
    st.markdown("**Educational explorer for Indian MSME law — Strict RAG Architecture**")

    # Persistent disclaimer
    st.warning(
        "⚠️ **Educational tool only.** All responses are generated from a fixed knowledge base. "
        "AI may still produce errors. Verify against primary sources. Not legal advice."
    )

    # Check if index is built
    stats = get_collection_stats()
    if stats["total_chunks"] == 0:
        st.error(
            "🚨 Knowledge base index is empty. Please click 'Build Index Now' in the sidebar to set it up. "
            "This is a one-time process that takes about 30 seconds."
        )
        return

    # Mode selector
    tab1, tab2, tab3 = st.tabs([
        "📋 Topic Explorer (Recommended)",
        "❓ Q&A Mode",
        "🔍 Knowledge Base Browser",
    ])

    # ========================================================================
    # TAB 1: TOPIC EXPLORER (Structured Inputs)
    # ========================================================================
    with tab1:
        st.header("Browse Topics")
        st.markdown(
            "Select a topic below to get a grounded educational explanation. "
            "Each answer is generated **only** from the curated knowledge base, with full citations."
        )

        # Group topics by category
        categories = {
            "MSMED Act 2006": ["msmed_overview", "msmed_payment_recovery", "msmed_odr_portal", "udyam_registration"],
            "GST Law": ["gst_section_73", "gst_section_74", "gst_section_132", "gst_notice_types", "itc_mismatch"],
            "Indian Contract Act": ["contract_red_flags", "non_compete", "liquidated_damages"],
            "Other Commercial Law": ["cheque_bounce", "lease_registration"],
        }

        for category, topic_keys in categories.items():
            st.markdown(f"#### {category}")

            cols = st.columns(2)
            for i, topic_key in enumerate(topic_keys):
                topic = STRUCTURED_TOPICS[topic_key]
                with cols[i % 2]:
                    if st.button(
                        f"📖 {topic['title']}",
                        key=f"topic_{topic_key}",
                        use_container_width=True,
                        help=topic['description'],
                    ):
                        st.session_state.selected_topic = topic_key

            st.markdown("")  # Spacing

        # Show selected topic response
        if "selected_topic" in st.session_state and st.session_state.selected_topic:
            topic_key = st.session_state.selected_topic
            topic = STRUCTURED_TOPICS[topic_key]

            st.markdown("---")
            st.subheader(f"📖 {topic['title']}")

            result = answer_with_rag(topic["query"], model=model_choice)

            st.markdown(result["answer"])

            # Show retrieved context (transparency)
            with st.expander("🔍 View raw retrieved context (transparency)"):
                for i, chunk in enumerate(result["retrieved_chunks"], 1):
                    st.markdown(f"**Chunk {i}** — Source: `{chunk['source']}`")
                    st.markdown(f"Section: *{chunk['heading']}*")
                    st.text(chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'])
                    st.markdown("---")

            # Download
            st.download_button(
                "📥 Download answer",
                result["answer"],
                file_name=f"legalsetu_{topic_key}_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
            )

    # ========================================================================
    # TAB 2: Q&A MODE (Free-form, but still RAG-grounded)
    # ========================================================================
    with tab2:
        st.header("Ask a Question")
        st.markdown(
            "Type a question about Indian MSME law. The system will retrieve relevant passages "
            "from the knowledge base and Claude will answer **strictly** from those passages. "
            "If your question isn't covered, Claude will refuse rather than guess."
        )

        st.info(
            "💡 **Tip**: Questions about general legal concepts work best. "
            "Avoid asking about your specific case — this tool is educational, not advisory."
        )

        user_query = st.text_input(
            "Your question:",
            placeholder="e.g., What is the interest rate under MSMED Act for delayed payments?",
            key="qa_input",
        )

        if st.button("🔍 Get Answer", type="primary", key="qa_button"):
            if not user_query.strip():
                st.warning("Please enter a question.")
            elif len(user_query) > 500:
                st.warning("Question is too long. Keep it under 500 characters for best results.")
            else:
                result = answer_with_rag(user_query, model=model_choice)

                st.markdown("---")
                st.subheader("Answer")
                st.markdown(result["answer"])

                # Show retrieval transparency
                with st.expander("🔍 View raw retrieved context (transparency)"):
                    for i, chunk in enumerate(result["retrieved_chunks"], 1):
                        st.markdown(f"**Chunk {i}** — Source: `{chunk['source']}`")
                        st.markdown(f"Section: *{chunk['heading']}*")
                        st.text(chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'])
                        st.markdown("---")

                st.download_button(
                    "📥 Download answer",
                    f"# Question\n\n{user_query}\n\n# Answer\n\n{result['answer']}",
                    file_name=f"legalsetu_qa_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                )

    # ========================================================================
    # TAB 3: KNOWLEDGE BASE BROWSER (Full Transparency)
    # ========================================================================
    with tab3:
        st.header("Browse the Knowledge Base")
        st.markdown(
            "Full transparency: see exactly what documents the AI has access to. "
            "Nothing else is used to generate answers."
        )

        from pathlib import Path
        kb_dir = Path(__file__).parent / "knowledge_base"
        md_files = sorted(kb_dir.glob("*.md"))

        if not md_files:
            st.warning("No knowledge base documents found.")
        else:
            file_choice = st.selectbox(
                "Select a document to view:",
                [f.name for f in md_files],
            )

            selected_file = next((f for f in md_files if f.name == file_choice), None)
            if selected_file:
                with open(selected_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                st.markdown(content)

    # Footer
    st.markdown("---")
    st.markdown(
        """<div style="text-align: center; color: #888; font-size: 0.85em;">
        LegalSetu Learn — Open Source Educational Tool |
        <a href="https://github.com/yourusername/legalsetu-learn">GitHub</a> |
        Strict RAG Architecture | Built with Claude API
        </div>""",
        unsafe_allow_html=True,
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    # Initialize session state
    if "agreement_accepted" not in st.session_state:
        st.session_state.agreement_accepted = False

    if not st.session_state.agreement_accepted:
        show_agreement_gate()
    else:
        show_main_app()


if __name__ == "__main__":
    main()
