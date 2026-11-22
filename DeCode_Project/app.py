import os
import json
import streamlit as st
from pathlib import Path
from src.pipeline import run_extraction_pipeline

st.set_page_config(page_title="DeCode — Meeting Notes to Action Items", page_icon="📋", layout="wide")

st.title("📋 DeCode: Meeting Notes to Action Items Extractor")
st.caption("Course: Prompt Engineering (SCSE4041) | Team DeCode (Bennett University)")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    model = st.selectbox("GPT Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    version = st.radio("Prompt Version", ["V1 (Baseline)", "V2 (Structured JSON)", "V3 (Strict Grounded)"], index=2)
    enable_verify = st.checkbox("Enable Step-2 Transcript Verifier", value=True)
    st.markdown("---")
    st.markdown("**Team DeCode:** Divyansh Solanki, Bhoomi Chaudhary, Rohan Khare, Govind Dixit, Krrish Taneja")

dataset_path = Path(__file__).parent / "data" / "synthetic_dataset.json"
samples = json.loads(dataset_path.read_text(encoding="utf-8"))
sample_titles = ["Custom Input"] + [f"{s['id']} — {s['title']} ({s['category']})" for s in samples]

selected_sample = st.selectbox("Load Test Transcript from 10-Sample Benchmark Dataset:", sample_titles)
default_text = ""
gold_ref = None
if selected_sample != "Custom Input":
    idx = sample_titles.index(selected_sample) - 1
    default_text = samples[idx]["transcript"]
    gold_ref = samples[idx]["gold"]

transcript = st.text_area("Meeting Transcript Input", value=default_text, height=200, placeholder="Paste raw meeting transcript here...")

if st.button("🚀 Extract & Verify", type="primary"):
    if not transcript.strip():
        st.warning("Please enter a meeting transcript first.")
    elif not api_key:
        st.error("Please enter your OpenAI API key in the sidebar or .env file.")
    else:
        ver_code = version.split()[0]
        with st.spinner(f"Running {ver_code} pipeline on {model}..."):
            try:
                result = run_extraction_pipeline(
                    transcript=transcript,
                    version=ver_code,
                    enable_verification=enable_verify,
                    api_key=api_key,
                    model=model,
                )
                if ver_code == "V1":
                    st.subheader("V1 Baseline Output (Unstructured)")
                    st.write(result["raw_output"])
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("✅ Verified Final Output")
                        st.json(result["final"])
                    with col2:
                        if gold_ref:
                            st.subheader("🎯 Gold Standard Reference")
                            st.json(gold_ref)
                        else:
                            st.subheader("🔍 Step-1 Candidate Output")
                            st.json(result["candidate"])
            except Exception as e:
                st.error(f"API Error: {e}")
