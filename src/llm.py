"""Thin Gemini wrapper. Swap this file to switch to Claude/OpenAI later."""
from __future__ import annotations
from typing import Iterable, Iterator
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from src import config


_initialized = False


def _init():
    global _initialized
    if not _initialized:
        if not config.GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY missing. Set it in .env (get free key at "
                "https://aistudio.google.com/apikey)"
            )
        genai.configure(api_key=config.GOOGLE_API_KEY)
        _initialized = True


def _model(system_prompt: str | None = None):
    _init()
    return genai.GenerativeModel(
        config.GEMINI_MODEL,
        system_instruction=system_prompt or config.SYSTEM_PROMPT,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def complete(prompt: str, system_prompt: str | None = None, temperature: float = 0.2) -> str:
    """One-shot completion."""
    model = _model(system_prompt)
    resp = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=4096,
        ),
    )
    return (resp.text or "").strip()


def stream(prompt: str, system_prompt: str | None = None, temperature: float = 0.2) -> Iterator[str]:
    """Streaming completion, yields chunks of text."""
    model = _model(system_prompt)
    resp = model.generate_content(
        prompt,
        stream=True,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=4096,
        ),
    )
    for chunk in resp:
        if chunk.text:
            yield chunk.text


def is_on_topic(question: str) -> bool:
    """Quick gatekeeper: is this an underwater systems question?
    
    Uses cheap model regardless of GEMINI_MODEL setting, to save tokens.
    """
    _init()
    cheap_model = genai.GenerativeModel(
        "gemini-2.5-flash-lite",
        system_instruction="You classify questions.",
    )
    prompt = (
        "Is the following question about underwater systems engineering "
        "(AUVs, ROVs, underwater drones, marine robotics, ocean engineering, "
        "subsea systems, hydrodynamics, underwater acoustics, marine vehicle "
        "design)? Answer ONLY 'yes' or 'no'.\n\n"
        f"Question: {question}"
    )
    try:
        resp = cheap_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=10,
            ),
        )
        ans = (resp.text or "").lower().strip()
        return ans.startswith("y")
    except Exception:
        return True