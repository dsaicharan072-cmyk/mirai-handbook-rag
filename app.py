"""Streamlit Web Chatbot UI for Mirai Academic & Hostel Policy Advisor."""

import streamlit as st
from src.rag import answer, retrieve_context
from src.config import TOP_K, DISTANCE_THRESHOLD

st.set_page_config(
    page_title="Mirai Policy Advisor Chatbot",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Mirai Academic & Hostel Policy Advisor")
st.caption("A strictly grounded RAG Chatbot powered by Google Gemini and ChromaDB")

# Sidebar
with st.sidebar:
    st.header("📌 About")
    st.markdown(
        """
        This chatbot provides factual answers regarding:
        - **Attendance & Debarment Rules**
        - **Medical Leave Submissions**
        - **Hostel Curfew & Mess Governance**
        - **Student Clubs & Event Approvals**
        - **Examinations & Grading Tiers**
        
        *Strict Grounding Notice*: If a policy or fact is not in the official 50-page handbook, the advisor responds with `[not found]`.
        """
    )
    st.divider()
    st.subheader("💡 Try Asking:")
    sample_queries = [
        "I have 72% attendance this semester. How many marks do I get and can I write the final exam?",
        "Within how many days after recovery must a student submit medical leave documents?",
        "What is the monthly fee for the swimming pool and indoor squash court?",
        "What percentage of batch support is needed to propose a new student club?",
        "What is the default password for the Mirai_Student_5G Wi-Fi?",
    ]
    for q in sample_queries:
        if st.button(q, use_container_width=True):
            st.session_state["user_prompt"] = q

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your Mirai University Academic & Hostel Policy Advisor. Ask me anything about academic integrity, attendance thresholds, medical leaves, hostel rules, or club guidelines.",
            "sources": [],
            "retrieved": [],
        }
    ]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown(f"**📚 Cited Sources:** `{', '.join(str(s) for s in msg['sources'])}`")
        if msg.get("retrieved"):
            with st.expander("🔍 View Retrieved Context Chunks"):
                for idx, c in enumerate(msg["retrieved"], 1):
                    st.markdown(f"**Chunk [{idx}] {c['chunk_id']} (Page {c['page']})** - Distance: `{c['distance']:.4f}`")
                    st.text(c["text"])

# Handle user input from text box or sidebar button
prompt = st.chat_input("Ask a question about Mirai University policies...")
if "user_prompt" in st.session_state and st.session_state["user_prompt"]:
    prompt = st.session_state["user_prompt"]
    st.session_state["user_prompt"] = None

if prompt:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching handbook and generating answer..."):
            retrieved = retrieve_context(prompt, top_k=TOP_K, distance_threshold=DISTANCE_THRESHOLD)
            res = answer(prompt)
            ans_text = res["answer"]
            sources = res["sources"]

            st.markdown(ans_text)
            if sources:
                st.markdown(f"**📚 Cited Sources:** `{', '.join(str(s) for s in sources)}`")
            elif ans_text == "[not found]":
                st.info("ℹ️ The official handbook does not contain information on this topic.")

            if retrieved:
                with st.expander("🔍 View Retrieved Context Chunks"):
                    for idx, c in enumerate(retrieved, 1):
                        st.markdown(f"**Chunk [{idx}] {c['chunk_id']} (Page {c['page']})** - Distance: `{c['distance']:.4f}`")
                        st.text(c["text"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": ans_text,
            "sources": sources,
            "retrieved": retrieved,
        }
    )
