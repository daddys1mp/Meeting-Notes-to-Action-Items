import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"


def load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def get_client(api_key: str | None = None) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY missing. Set it in .env or Streamlit sidebar.")
    return OpenAI(api_key=key)


def call_gpt(client: OpenAI, prompt: str, model: str = "gpt-4o-mini", json_mode: bool = True) -> str:
    kwargs = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": "You are a precise enterprise meeting extraction engine."},
            {"role": "user", "content": prompt},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def run_extraction_pipeline(
    transcript: str,
    version: str = "V3",
    enable_verification: bool = True,
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
) -> dict:
    client = get_client(api_key)

    if version == "V1":
        raw_prompt = load_prompt("v1_baseline.txt").format(transcript=transcript)
        raw_text = call_gpt(client, raw_prompt, model=model, json_mode=False)
        return {"version": "V1", "raw_output": raw_text, "verified": False}

    prompt_file = "v2_structured.txt" if version == "V2" else "v3_strict_grounded.txt"
    extract_prompt = load_prompt(prompt_file).format(transcript=transcript)
    candidate_str = call_gpt(client, extract_prompt, model=model, json_mode=True)
    candidate_json = json.loads(candidate_str)

    if not enable_verification:
        return {
            "version": version,
            "verified": False,
            "candidate": candidate_json,
            "final": candidate_json,
        }

    verify_prompt = load_prompt("verifier.txt").format(
        transcript=transcript,
        candidate_json=json.dumps(candidate_json, indent=2),
    )
    verified_str = call_gpt(client, verify_prompt, model=model, json_mode=True)
    verified_json = json.loads(verified_str)

    return {
        "version": version,
        "verified": True,
        "candidate": candidate_json,
        "final": verified_json,
    }
