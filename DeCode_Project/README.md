# DeCode — Meeting Notes to Action Items Extractor
**Course:** Prompt Engineering (SCSE4041) | **Institution:** Bennett University  
**Team DeCode:** Divyansh Solanki (S24CSEU1975), Bhoomi Chaudhary (S24CSEU2667), Rohan Khare (S24CSEU1755), Govind Dixit (S24CSEU1851), Krrish Taneja (S24CSEU1884)

## Overview
A multi-step prompt-engineered pipeline powered by the **OpenAI GPT API** and **Streamlit** that converts raw meeting transcripts into verified, strictly grounded JSON (`decisions`, `action_items`, `open_items`) with verbatim evidence quotes.

## Quickstart
```bash
pip install -r requirements.txt
cp .env.example .env   # Add your OPENAI_API_KEY
streamlit run app.py
```
