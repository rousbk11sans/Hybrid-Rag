import streamlit as st

from src.pipeline import MODES, Pipeline


@st.cache_resource
def get_pipeline():
    return Pipeline()


st.set_page_config(layout="wide", page_title="Hybrid RAG Search Engine")
st.title("Hybrid RAG Search Engine")
p = get_pipeline()

q = st.text_input("Enter a scientific claim or query")
modes = st.multiselect("Methods", MODES, default=MODES)

if q and modes:
    cols = st.columns(len(modes))
    for col, m in zip(cols, modes):
        col.subheader(m)
        try:
            for d in p.run(q, m, k=5):
                col.markdown(f"**{d}**  \n{p.doc_texts[p.doc_index[d]][:200]}...")
        except Exception as e:
            col.error(str(e))
