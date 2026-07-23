#pip install streamlit

#streamlit run ui/app.py

#add this as model_path in the ui for now : C:\Users\gunn\models\Phi-3-mini-4k-instruct-q4.gguf

import sys
import time
from pathlib import Path
import streamlit as st
sys.path.insert(0,str(Path(__file__).parent.parent))
from phase_three.agent import CrisisAgent
from sync.sync_manager import SyncManager

DEFAULT_DB_PATH="data/lancedb"
SYNC_POLL_INTERVAL_SEC=1800

INTENT_COLORS = {
    "MEDICAL": "#d9534f",
    "EVACUATION": "#f0ad4e",
    "SURVIVAL": "#5bc0de",
    "EMOTIONAL": "#9b6bcc",
    "GENERAL": "#6c757d",
    "NO_TRIAGE": "#6c757d",
}
 

st.set_page_config(page_title="Crisis RAG", page_icon="🆘", layout="centered")

@st.cache_resource(show_spinner="Loading the model and knowledge base (first load can take a minute)...")
def load_agent(db_path: str, model_path: str) -> CrisisAgent:
    return CrisisAgent(db_path=db_path, model_path=model_path)
 
 
@st.cache_resource(show_spinner=False)
def load_sync_manager(db_path: str) -> SyncManager:
    manager = SyncManager(db_path=db_path, poll_interval_sec=SYNC_POLL_INTERVAL_SEC)
    manager.start()
    return manager


#sidebar
with st.sidebar:
    st.header("Setup")
    model_path=st.text_input(
        "Model Path (.gguf)",
        value=st.session_state.get("model_path", ""),
        placeholder=r"C:\Users\you\models\Phi-3-mini-4k-instruct-q4.gguf",
        help="Path to your Phi-3 Mini GGUF file — same one you'd pass with --model on the CLI.",
    )
    db_path= st.text_input("Vector database path", value=DEFAULT_DB_PATH)
    load_clicked=st.button("Load Crisis Rag", type="primary", use_container_width=True)
    if load_clicked:
        if not model_path or not Path(model_path).exists():
            st.error("Model file not founf at that path. Double check it and try again.")
        elif not Path(db_path).exists():
            st.error(f"database not found at :'{db_path}'. Run phase_one/ingest.py first.")
        else:
            st.session_state.model_path=model_path
            st.session_state.db_path=db_path
            st.session_state.system_ready=True
    st.divider()
    if st.session_state.get("system_ready"):
        st.success("System Laoded and ready")
        st.subheader("background sync")
        sync_manager=load_sync_manager(st.session_state.db_path)
        st.caption(f"Check every {SYNC_POLL_INTERVAL_SEC // 60} min when online"
                   "Offline is unaffected either ways."
                   )
        if st.button("Sync now", use_container_width=True):
            with st.spinner("Check for updates.."):
                result=sync_manager.check_and_sync_once()
            if result["status"]=="offline":
                st.info("Currently offline, nothing to sync.")
            elif result["status"]=="no_sources":
                st.info("No sources configured yet , check your sync/sources.json")
            else:
                st.success(
                    f"Updated :{result['updated'] or 'none'} |"
                    f"Unchanged: {len(result['unchanged'])} |"
                    f"Failed: {result['failed'] or 'none'}"
                )
        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state["messages"]=[]
            st.rerun()
        else:
            st.info("Enter your model path above and click **Load Crisis RAG** to begin." )


st.title("Crisis Rag")
st.caption("Offline-first crisis assistance. Answers are grounded in the local knowledge base and are not a substitute for professiobal emergency services")
if "messages" not in st.session_state:
    st.session_state.messages=[]

if not st.session_state.get("system_ready"):
    st.warning("Load the system from teh sidebar before asking any question")
    st.stop()

agent=load_agent(st.session_state.db_path, st.session_state.model_path)

# Replay conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]
            color = INTENT_COLORS.get(meta["intent"], "#6c757d")
            st.markdown(
                f"<span style='background-color:{color};color:white;padding:2px 8px;"
                f"border-radius:10px;font-size:0.8em;'>{meta['intent']}</span> "
                f"<span style='font-size:0.8em;color:#888;'>"
                f"confidence: {meta['confidence']} · {meta['iterations']} search iteration(s) · "
                f"{meta['latency']:.1f}s</span>",
                unsafe_allow_html=True,
            )
            if meta["sources"]:
                with st.expander(f"Sources ({len(meta['sources'])})"):
                    for s in meta["sources"]:
                        st.markdown(f"- **{s['source']}** — {s['score']*100:.0f}% match") 

#next question
query=st.chat_input("Describe your situation...")
if query:
    st.session_state.messages.append({"role":"user","content":query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Classifying and searching"):
            t0=time.time()
            try:
                response=agent.run(query)
                latency=time.time()-t0
                error=None 
            except Exception as e:
                response=None 
                latency=time.time()-t0
                error=str(e)
        if error:
            st.error(f"something went wrong generating a response: {error}")
            st.session_state.messages.append({
                "role":"assistant",
                "content":f"Error:{error}",
                "meta":None ,
            })
        else:
            st.markdown(response.answer)
            color=INTENT_COLORS.get(response.intent,"#6c757d")
            st.markdown(
                f"<span style='background-color:{color};color:white;padding:2px 8px;'>{response.intent}</span>"
                f"<span style='font-size:0.8em;color:#888;'>"
                f"confidence:{response.confidence}."
                f"{response.iterations} search iterations"
                f"{latency:.1f}s</span>",
                unsafe_allow_html=True,
            )
            if response.sources:
                with st.expander(f"Sources ({len(response.sources)})"):
                    for s in response.sources:
                        st.markdown(f"- **{s['source']}** -{s['score']*100:.0f}% match")
            st.session_state.messages.append({
                "role":"assistant",
                "content":response.answer,
                "meta":{
                    "intent":response.intent,
                    "confidence":response.confidence,
                    "iterations":response.iterations,
                    "latency":latency,
                    "sources":response.sources,
                },
            })



