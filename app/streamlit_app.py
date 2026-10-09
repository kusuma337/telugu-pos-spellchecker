# -*- coding: utf-8 -*-
"""Telugu POS Tagging & Intelligent Spellchecking — Advanced 7-Tab UI."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path bootstrap — works from project root OR from app/ sub-directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import MissingArtifactError, analyze_sentence, predict_with_confidence
from src.preprocessing import tokenize, contains_telugu
from app.analytics import (
    detect_script,
    get_text_analytics,
    get_pos_distribution,
    introduce_noise,
    load_eval_metrics,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_SENTENCE = "నేను పాఠశాలకు వెళ్తున్నాను."

TAG_ENGLISH: dict[str, str] = {
    "PRON":  "Pronoun",
    "NOUN":  "Noun",
    "PROPN": "Proper Noun",
    "VERB":  "Verb",
    "ADJ":   "Adjective",
    "ADV":   "Adverb",
    "ADP":   "Adposition",
    "DET":   "Determiner",
    "CONJ":  "Conjunction",
    "CCONJ": "Conjunction",
    "SCONJ": "Conjunction",
    "PUNCT": "Punctuation",
    "NUM":   "Number",
    "INTJ":  "Interjection",
    "PART":  "Particle",
    "X":     "Other",
}

TAG_TELUGU: dict[str, str] = {
    "PRON":  "సర్వనామం",
    "NOUN":  "నామవాచకం",
    "PROPN": "సంజ్ఞావాచకం",
    "VERB":  "క్రియ",
    "ADJ":   "విశేషణం",
    "ADV":   "క్రియా విశేషణం",
    "ADP":   "విభక్తి",
    "DET":   "నిర్ణాయకం",
    "CONJ":  "సంయోగం",
    "CCONJ": "సమన్వయ సంయోగం",
    "SCONJ": "అధీన సంయోగం",
    "PUNCT": "విరామ చిహ్నం",
    "NUM":   "సంఖ్య",
    "INTJ":  "విస్మయాది బోధకం",
    "PART":  "నిపాతం",
    "X":     "ఇతర",
}

PIPELINE_STEPS = [
    ("📝", "Telugu Input",     "Raw Unicode text entered by the user"),
    ("⚙️", "Preprocessing",   "Tokenization, NFC normalization, whitespace collapse"),
    ("🔍", "Spell Checking",  "Edit-distance + unigram log-probability ranking"),
    ("🧠", "BiLSTM",          "Bidirectional LSTM — corrected tokens → tag logits"),
    ("🏷️", "POS Tagging",     "Per-token part-of-speech classification"),
    ("✅", "Results",          "Corrected sentence + POS tag table"),
]

TECH_STACK = [
    ("Python",       "Core language"),
    ("PyTorch",      "Deep learning framework"),
    ("BiLSTM",       "Sequence tagging model"),
    ("NLP",          "Natural Language Processing"),
    ("scikit-learn", "Evaluation metrics"),
    ("Streamlit",    "Interactive web UI"),
]

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Telugu:wght@400;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', 'Noto Sans Telugu', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #fdf6f0 0%, #f0f4ff 50%, #f5fff8 100%);
}
[data-testid="stHeader"]    { background: transparent !important; }
[data-testid="stSidebar"]   { background: #1a1f3c !important; }
[data-testid="stSidebar"] * { color: #e8eaf6 !important; }
[data-testid="stSidebar"] hr { border-color: #3d4270 !important; }
[data-testid="stSidebar"] .stMarkdown code {
    background: #2d3561;
    color: #a5d8ff !important;
}
.te { font-family: 'Noto Sans Telugu', 'Noto Sans', serif; font-size: 1.05em; }

/* Hero */
.hero {
    background: linear-gradient(125deg, #0d1b6e 0%, #1a3a8f 45%, #0e6655 100%);
    border-radius: 20px; padding: 40px 48px 36px;
    margin-bottom: 36px; position: relative; overflow: hidden;
    box-shadow: 0 8px 32px rgba(13,27,110,.22);
}
.hero::before {
    content: "🪷"; position: absolute; right: 40px; top: 24px;
    font-size: 7rem; opacity: .12; pointer-events: none;
}
.hero-title {
    font-family: 'Noto Sans Telugu', 'Inter', sans-serif;
    font-size: 2.1rem; font-weight: 800; color: #ffffff;
    margin: 0 0 10px; line-height: 1.25; text-shadow: 0 2px 8px rgba(0,0,0,.25);
}
.hero-sub {
    font-family: 'Noto Sans Telugu', 'Inter', sans-serif;
    font-size: 1.05rem; color: #cfd8ff; margin: 0 0 18px; line-height: 1.5;
}
.hero-pills { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 4px; }
.hero-pill {
    background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.28);
    border-radius: 20px; padding: 4px 14px; font-size: .8rem; color: #e8f0ff;
}

/* Section header */
.sec-hdr {
    font-family: 'Noto Sans Telugu', 'Inter', sans-serif;
    font-size: 1.3rem; font-weight: 700; color: #0d1b6e;
    border-left: 5px solid #ff6f00; padding-left: 12px; margin: 32px 0 14px;
}

/* Result cards */
.res-card {
    background: #ffffff; border-radius: 14px; padding: 18px 22px;
    box-shadow: 0 2px 10px rgba(0,0,0,.07); margin-bottom: 14px;
    border-left: 5px solid var(--card-accent, #1565c0);
}
.res-card-lbl {
    font-size: .72rem; font-weight: 700; color: #888;
    text-transform: uppercase; letter-spacing: .1em; margin-bottom: 6px;
}
.res-card-body {
    font-family: 'Noto Sans Telugu', serif; font-size: 1.15rem;
    color: #1a1a2e; line-height: 1.8;
}

/* Spell badges */
.spell-row { display: flex; align-items: center; flex-wrap: wrap; gap: 6px; margin: 4px 0; }
.badge-bad  { font-family: 'Noto Sans Telugu', serif; background: #fff3e0; color: #bf360c;
              border: 1px solid #ffcc80; border-radius: 8px; padding: 3px 12px; font-size: .95rem; }
.badge-good { font-family: 'Noto Sans Telugu', serif; background: #e8f5e9; color: #1b5e20;
              border: 1px solid #a5d6a7; border-radius: 8px; padding: 3px 12px; font-size: .95rem; }
.badge-arrow { color: #888; font-size: 1.1rem; }

/* Metric cards */
.met-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin: 10px 0 24px; }
@media (max-width:700px) { .met-grid { grid-template-columns: repeat(2,1fr); } }
.met-card {
    background: #fff; border-radius: 14px; padding: 22px 16px 16px;
    text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,.07);
    border-top: 5px solid var(--m-color);
}
.met-val { font-size: 2.1rem; font-weight: 800; color: var(--m-color); margin: 6px 0 2px; }
.met-lbl { font-size: .82rem; color: #555; text-transform: uppercase; letter-spacing: .07em; }
.met-note { font-size: .68rem; color: #aaa; margin-top: 4px; }

/* Analytics cards */
.analytics-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin: 12px 0 20px; }
@media (max-width:700px) { .analytics-grid { grid-template-columns: repeat(2,1fr); } }
.analytics-card {
    background: #fff; border-radius: 12px; padding: 16px 14px;
    text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,.06);
    border-top: 4px solid var(--ac-color, #1565c0);
}
.analytics-val { font-size: 1.7rem; font-weight: 800; color: var(--ac-color, #1565c0); margin: 4px 0 2px; }
.analytics-lbl { font-size: .78rem; color: #666; text-transform: uppercase; letter-spacing: .07em; }

/* Confidence bar */
.conf-row { display: flex; align-items: center; gap: 10px; margin: 3px 0; }
.conf-label { font-family: 'Noto Sans Telugu', serif; min-width: 120px; font-size: .95rem; color: #222; }
.conf-bar-outer { flex: 1; background: #f0f0f0; border-radius: 6px; height: 12px; overflow: hidden; }
.conf-bar-inner { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #1565c0, #42a5f5); }
.conf-pct { min-width: 48px; font-size: .82rem; font-weight: 700; color: #1565c0; text-align: right; }

/* Comparison table */
.cmp-table { width: 100%; border-collapse: collapse; margin: 14px 0; font-size: .92rem; }
.cmp-table th {
    background: #1a1f3c; color: #fff; padding: 10px 14px;
    text-align: left; font-weight: 600; font-size: .82rem; text-transform: uppercase;
}
.cmp-table td { padding: 10px 14px; border-bottom: 1px solid #e8eaf6; }
.cmp-table tr:hover td { background: #f5f7ff; }
.best-cell { color: #1b5e20; font-weight: 800; }

/* Pipeline */
.pipe-step {
    display: flex; align-items: flex-start; gap: 14px; background: #fff;
    border-radius: 12px; padding: 13px 16px; margin-bottom: 8px;
    box-shadow: 0 1px 6px rgba(0,0,0,.06); border-left: 4px solid #0d47a1;
}
.pipe-step:hover { border-color: #ff6f00; }
.pipe-icon { font-size: 1.4rem; flex-shrink: 0; margin-top: 1px; }
.pipe-title { font-weight: 700; font-size: .97rem; color: #1a1a2e; }
.pipe-desc  { font-size: .8rem; color: #666; margin-top: 2px; line-height: 1.45; }

/* Tech badges */
.tech-wrap { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 4px; }
.tech-badge {
    background: linear-gradient(135deg, #e8f0fe, #d7e3fc);
    color: #1a3a8f; border-radius: 20px; padding: 5px 16px;
    font-size: .87rem; font-weight: 600; border: 1px solid #c5d5f5;
}

textarea {
    font-family: 'Noto Sans Telugu', serif !important;
    font-size: 1.1rem !important; line-height: 1.7 !important;
}

.te-divider { border: none; border-top: 1px dashed #ccd; margin: 32px 0 8px; }

/* Noisy comparison */
.noisy-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 14px 0; }
@media (max-width:700px) { .noisy-grid { grid-template-columns: 1fr; } }
.noisy-panel {
    background: #fff; border-radius: 14px; padding: 18px 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,.07); border-top: 4px solid var(--np-color);
}
.noisy-panel-title {
    font-weight: 700; font-size: .9rem; text-transform: uppercase;
    letter-spacing: .07em; color: var(--np-color); margin-bottom: 10px;
}

/* Warning banner */
.warn-banner {
    background: #fff8e1; border: 1px solid #ffe082; border-radius: 10px;
    padding: 12px 18px; font-size: .9rem; color: #795548; margin: 10px 0 16px;
}

/* Footer */
.footer {
    text-align: center; font-size: .78rem; color: #aaa;
    padding: 20px 0 8px; border-top: 1px solid #e0e0e0;
    margin-top: 48px;
}
</style>
"""

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model artifacts…")
def _load_artifacts():
    """Load and cache the model, vocabularies, and spellchecker."""
    from src.inference import load_trained_artifacts
    return load_trained_artifacts()


@st.cache_data(show_spinner=False)
def _load_eval_metrics() -> dict:
    return load_eval_metrics()


# ---------------------------------------------------------------------------
# Helper renderers
# ---------------------------------------------------------------------------

def translate_tag(tag: str) -> str:
    return TAG_ENGLISH.get(tag.upper(), tag)


def translate_tag_telugu(tag: str) -> str:
    return TAG_TELUGU.get(tag.upper(), tag)


def _metric_card(label: str, value: str, color: str, note: str = "Test-set result") -> str:
    return (
        f'<div class="met-card" style="--m-color:{color}">'
        f'  <div class="met-lbl">{label}</div>'
        f'  <div class="met-val">{value}</div>'
        f'  <div class="met-note">{note}</div>'
        f'</div>'
    )


def _analytics_card(label: str, value: str, color: str) -> str:
    return (
        f'<div class="analytics-card" style="--ac-color:{color}">'
        f'  <div class="analytics-val">{value}</div>'
        f'  <div class="analytics-lbl">{label}</div>'
        f'</div>'
    )


def _result_card(label: str, body: str, accent: str = "#1565c0") -> str:
    return (
        f'<div class="res-card" style="--card-accent:{accent}">'
        f'  <div class="res-card-lbl">{label}</div>'
        f'  <div class="res-card-body">{body}</div>'
        f'</div>'
    )


def _pipeline_html() -> str:
    parts = []
    for icon, title, desc in PIPELINE_STEPS:
        parts.append(
            f'<div class="pipe-step">'
            f'  <div class="pipe-icon">{icon}</div>'
            f'  <div>'
            f'    <div class="pipe-title">{title}</div>'
            f'    <div class="pipe-desc">{desc}</div>'
            f'  </div>'
            f'</div>'
        )
    return "\n".join(parts)


def _tech_badges_html() -> str:
    badges = "".join(
        f'<span class="tech-badge" title="{role}">{name}</span>'
        for name, role in TECH_STACK
    )
    return f'<div class="tech-wrap">{badges}</div>'


def _confidence_bars_html(tagged: list[dict]) -> str:
    rows = ""
    for item in tagged:
        conf = item.get("confidence", 1.0)
        pct = conf * 100
        rows += (
            f'<div class="conf-row">'
            f'  <div class="conf-label">{item["word"]}</div>'
            f'  <div class="conf-bar-outer">'
            f'    <div class="conf-bar-inner" style="width:{pct:.1f}%"></div>'
            f'  </div>'
            f'  <div class="conf-pct">{pct:.1f}%</div>'
            f'</div>'
        )
    return rows


def _spell_diff_html(original_tokens: list[str], corrected_tokens: list[str]) -> str:
    """Render before/after comparison highlighting changed words."""
    orig_set = {item["original"]: item["corrected"] for item in []}
    # Build a word-level mapping using index alignment
    parts_orig = []
    parts_corr = []
    for o, c in zip(original_tokens, corrected_tokens):
        if o != c:
            parts_orig.append(f'<span class="badge-bad">{o}</span>')
            parts_corr.append(f'<span class="badge-good">{c}</span>')
        else:
            parts_orig.append(f'<span style="font-family:\'Noto Sans Telugu\',serif">{o}</span>')
            parts_corr.append(f'<span style="font-family:\'Noto Sans Telugu\',serif">{c}</span>')
    return (
        f'<div style="margin-bottom:6px"><strong>Before:</strong> '
        + " ".join(parts_orig) + "</div>"
        f'<div><strong>After:</strong> '
        + " ".join(parts_corr) + "</div>"
    )


def _validate_script(text: str) -> bool:
    """Return True if text contains Telugu; show a warning if not."""
    script = detect_script(text)
    if script == "empty":
        st.warning("⚠️ Please enter some text before running analysis.")
        return False
    if script == "english":
        st.warning(
            "⚠️ The entered text appears to be English, not Telugu. "
            "This model is trained on Telugu text and may produce incorrect results."
        )
        return True  # Allow processing, but warn
    if script == "numeric":
        st.warning("⚠️ The entered text appears to be numeric only.")
        return True
    return True


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<p style="font-family:\'Noto Sans Telugu\',serif;font-size:1.25rem;'
            'font-weight:800;color:#fff;margin:0 0 4px">🪷 Telugu NLP</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="font-size:.8rem;color:#9fa8da;margin:0 0 16px">'
            'Telugu Language Analysis System</p>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("**🧠 Model Details**")
        for k, v in [
            ("Architecture", "BiLSTM"),
            ("Embedding",    "128 dims"),
            ("Hidden dim",   "128"),
            ("Dropout",      "0.3"),
            ("Language",     "Telugu (te)"),
            ("Framework",    "PyTorch"),
        ]:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:.83rem;padding:3px 0;border-bottom:1px solid #2d3561">'
                f'<span style="color:#9fa8da">{k}</span>'
                f'<span style="color:#fff;font-weight:600">{v}</span></div>',
                unsafe_allow_html=True,
            )

        metrics_data = _load_eval_metrics()
        summary = metrics_data.get("summary", {})
        bilstm_spell = summary.get("BiLSTM_plus_spellcheck", {})
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**📊 Best-Model Performance**")
        for lbl, val, color in [
            ("Accuracy",  f"{bilstm_spell.get('accuracy', 0.9732)*100:.2f}%", "#2e7d32"),
            ("Precision", f"{bilstm_spell.get('precision', 0.8259)*100:.2f}%", "#1565c0"),
            ("Recall",    f"{bilstm_spell.get('recall', 0.7897)*100:.2f}%",    "#e65100"),
            ("F1 Score",  f"{bilstm_spell.get('f1', 0.8003):.4f}",             "#6a1b9a"),
        ]:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:.83rem;padding:3px 0;border-bottom:1px solid #2d3561">'
                f'<span style="color:#9fa8da">{lbl}</span>'
                f'<span style="color:{color};font-weight:700">{val}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
            '<p style="font-size:.72rem;color:#666;text-align:center;margin:0">'
            'Telugu POS Tagging &amp; Intelligent Spellchecking · BiLSTM NLP Demo</p>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tab 1 — Home
# ---------------------------------------------------------------------------

def tab_home() -> None:
    st.markdown(
        '<div class="hero">'
        '  <div class="hero-title">🪷 Telugu POS Tagging &amp; Intelligent Spellchecking</div>'
        '  <div class="hero-sub">'
        '    AI-powered Telugu text processing · BiLSTM POS tagging · Probabilistic spellchecking'
        '  </div>'
        '  <div class="hero-pills">'
        '    <span class="hero-pill">BiLSTM</span>'
        '    <span class="hero-pill">Probabilistic Spellchecker</span>'
        '    <span class="hero-pill">PyTorch</span>'
        '    <span class="hero-pill">Telugu NLP</span>'
        '    <span class="hero-pill">Confidence Scores</span>'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )

    metrics_data = _load_eval_metrics()
    summary = metrics_data.get("summary", {})
    bilstm = summary.get("BiLSTM", {})
    bilstm_spell = summary.get("BiLSTM_plus_spellcheck", {})

    st.markdown('<div class="sec-hdr">🚀 Quick Demo</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        demo_text = st.text_input(
            "Try a sentence:",
            value=SAMPLE_SENTENCE,
            key="home_demo_input",
            label_visibility="collapsed",
        )
    with col2:
        run_demo = st.button("▶ Analyze", type="primary", use_container_width=True, key="home_run")

    if run_demo and demo_text.strip():
        _validate_script(demo_text)
        with st.spinner("Analyzing…"):
            try:
                res = analyze_sentence(demo_text.strip())
                st.success(f"✅ Corrected: **{res['corrected']}**")
                if res["tagged"]:
                    df = pd.DataFrame([
                        {"Word": it["word"], "POS Tag": it["tag"],
                         "Meaning": translate_tag(it["tag"]),
                         "Confidence": f"{it.get('confidence',1.0)*100:.1f}%"}
                        for it in res["tagged"]
                    ])
                    st.dataframe(df, hide_index=True, use_container_width=True)
            except MissingArtifactError as exc:
                st.error(f"🚨 Model not found: {exc}")
            except Exception as exc:
                st.error(f"Error: {exc}")

    st.markdown('<div class="sec-hdr">📊 Model Quick Stats</div>', unsafe_allow_html=True)
    cards_html = '<div class="met-grid">'
    cards_html += _metric_card("BiLSTM Accuracy",   f"{bilstm.get('accuracy',0.9643)*100:.2f}%",      "#1565c0", "BiLSTM only")
    cards_html += _metric_card("BiLSTM+SC Accuracy",f"{bilstm_spell.get('accuracy',0.9732)*100:.2f}%","#2e7d32", "BiLSTM + Spellcheck")
    cards_html += _metric_card("Best F1 Score",     f"{bilstm_spell.get('f1',0.8003):.4f}",           "#6a1b9a", "BiLSTM + Spellcheck")
    cards_html += _metric_card("Best Recall",       f"{bilstm_spell.get('recall',0.7897)*100:.2f}%",  "#e65100", "BiLSTM + Spellcheck")
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">⚙️ Pipeline Overview</div>', unsafe_allow_html=True)
    st.markdown(_pipeline_html(), unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">🛠️ Tech Stack</div>', unsafe_allow_html=True)
    st.markdown(_tech_badges_html(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 2 — Sentence Analysis
# ---------------------------------------------------------------------------

def tab_sentence_analysis() -> None:
    st.markdown('<div class="sec-hdr">✍️ Enter Telugu Text</div>', unsafe_allow_html=True)

    col_txt, col_btn = st.columns([5, 1])
    with col_txt:
        user_text = st.text_area(
            label="Telugu input",
            height=130,
            placeholder="Example: నేను పాఠశాలకు వెళ్తున్నాను.",
            label_visibility="collapsed",
            key="sa_text_input",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📋 Sample", use_container_width=True, key="sa_sample"):
            st.session_state["sa_loaded_sample"] = SAMPLE_SENTENCE
            st.rerun()

    if "sa_loaded_sample" in st.session_state and not user_text:
        user_text = st.session_state.pop("sa_loaded_sample")

    bc1, bc2, bc3, _pad = st.columns([3, 3, 3, 3])
    with bc1:
        spell_clicked = st.button("🔍 Spell Check", use_container_width=True, key="sa_spell")
    with bc2:
        pos_clicked = st.button("🏷️ POS Tagging", use_container_width=True, key="sa_pos")
    with bc3:
        full_clicked = st.button("🚀 Full Analysis", type="primary",
                                 use_container_width=True, key="sa_full")

    active_text = (user_text or "").strip()
    mode: str | None = None
    if spell_clicked:   mode = "spell"
    elif pos_clicked:   mode = "pos"
    elif full_clicked:  mode = "full"

    if mode is not None:
        if not active_text:
            st.warning("⚠️ Please enter a Telugu sentence before running analysis.")
            return
        _validate_script(active_text)
        with st.spinner("Running analysis…"):
            try:
                result = analyze_sentence(active_text)
                st.session_state["sa_last_result"] = result
            except MissingArtifactError as exc:
                st.error(f"🚨 Model artifact not found: {exc}")
                st.info("Train the model first: `python -m src.train`")
                return
            except ValueError as exc:
                st.warning(f"⚠️ {exc}")
                return
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
                return

    result = st.session_state.get("sa_last_result")
    if result is None:
        return

    st.markdown('<div class="sec-hdr">📊 Analysis Results</div>', unsafe_allow_html=True)

    # Before/After comparison
    orig_tokens = result.get("original_tokens", [])
    corr_tokens = result.get("corrected_tokens", [])
    if mode in (None, "spell", "full") or True:
        st.markdown(
            _result_card("📄 Original Text", result["original"], "#1565c0"),
            unsafe_allow_html=True,
        )
        if orig_tokens != corr_tokens:
            st.markdown(
                _result_card(
                    "✨ Before / After Comparison",
                    _spell_diff_html(orig_tokens, corr_tokens),
                    "#2e7d32",
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                _result_card("✨ Corrected Text", result["corrected"], "#2e7d32"),
                unsafe_allow_html=True,
            )

    # Misspellings
    st.markdown('<div class="sec-hdr">⚠️ Detected Misspellings</div>', unsafe_allow_html=True)
    if result.get("misspellings"):
        rows_html = ""
        for item in result["misspellings"]:
            rows_html += (
                f'<div class="spell-row">'
                f'  <span class="badge-bad">{item["original"]}</span>'
                f'  <span class="badge-arrow">→</span>'
                f'  <span class="badge-good">{item["corrected"]}</span>'
                f'</div>'
            )
        st.markdown(rows_html, unsafe_allow_html=True)
    else:
        st.success("✅ No misspellings detected — all words are in the vocabulary.")

    # POS table with confidence
    if result.get("tagged"):
        st.markdown('<div class="sec-hdr">🏷️ POS Tagging Results</div>', unsafe_allow_html=True)
        rows = [
            {
                "Word":           item["word"],
                "POS Tag":        item["tag"],
                "Telugu Meaning": translate_tag_telugu(item["tag"]),
                "English Meaning":translate_tag(item["tag"]),
                "Confidence %":   f"{item.get('confidence', 1.0)*100:.1f}%",
            }
            for item in result["tagged"]
        ]
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Text analytics cards
    analytics = get_text_analytics(result["original"], result)
    st.markdown('<div class="sec-hdr">📈 Text Analytics</div>', unsafe_allow_html=True)
    ac = '<div class="analytics-grid">'
    ac += _analytics_card("Characters",     str(analytics["char_count"]),     "#1565c0")
    ac += _analytics_card("Words",          str(analytics["word_count"]),     "#2e7d32")
    ac += _analytics_card("Unique Words",   str(analytics["unique_words"]),   "#e65100")
    ac += _analytics_card("Avg Word Length",f'{analytics["avg_word_length"]}', "#6a1b9a")
    ac += _analytics_card("Punctuation",    str(analytics["punctuation_count"]), "#0277bd")
    ac += _analytics_card("Misspellings",   str(analytics["misspelling_count"]), "#c62828")
    ac += "</div>"
    st.markdown(ac, unsafe_allow_html=True)

    # POS distribution bar chart
    if result.get("tagged"):
        dist = get_pos_distribution(result["tagged"])
        if dist:
            st.markdown('<div class="sec-hdr">📊 POS Distribution</div>', unsafe_allow_html=True)
            dist_df = pd.DataFrame(
                {"POS Tag": list(dist.keys()), "Count": list(dist.values())}
            ).sort_values("Count", ascending=False)
            st.bar_chart(dist_df.set_index("POS Tag"))

    # Export buttons
    st.markdown('<div class="sec-hdr">⬇️ Export Results</div>', unsafe_allow_html=True)
    exp_col1, exp_col2, _ = st.columns([2, 2, 4])
    if result.get("tagged"):
        export_rows = [
            {
                "word": item["word"],
                "tag": item["tag"],
                "tag_english": translate_tag(item["tag"]),
                "tag_telugu": translate_tag_telugu(item["tag"]),
                "confidence": round(item.get("confidence", 1.0), 4),
            }
            for item in result["tagged"]
        ]
        csv_buf = io.StringIO()
        pd.DataFrame(export_rows).to_csv(csv_buf, index=False, encoding="utf-8")
        with exp_col1:
            st.download_button(
                "⬇️ Download CSV",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name="telugu_pos_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        json_str = json.dumps(result, ensure_ascii=False, indent=2)
        with exp_col2:
            st.download_button(
                "⬇️ Download JSON",
                data=json_str.encode("utf-8"),
                file_name="telugu_pos_results.json",
                mime="application/json",
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Tab 3 — Spellchecker
# ---------------------------------------------------------------------------

def tab_spellchecker() -> None:
    st.markdown('<div class="sec-hdr">🔍 Telugu Spellchecker</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:.88rem;color:#555;margin-top:-10px">Enter Telugu text to detect and '
        'correct misspellings using probabilistic edit-distance ranking.</p>',
        unsafe_allow_html=True,
    )

    spell_text = st.text_area(
        "Input", height=120,
        placeholder="Enter Telugu text here…",
        label_visibility="collapsed",
        key="sc_input",
    )
    sc1, sc2, _ = st.columns([2, 2, 6])
    with sc1:
        run_sc = st.button("🔍 Check Spelling", type="primary",
                           use_container_width=True, key="sc_run")
    with sc2:
        if st.button("📋 Load Sample", use_container_width=True, key="sc_sample"):
            st.session_state["sc_loaded"] = SAMPLE_SENTENCE
            st.rerun()

    if "sc_loaded" in st.session_state and not spell_text:
        spell_text = st.session_state.pop("sc_loaded")

    if run_sc:
        active = (spell_text or "").strip()
        if not active:
            st.warning("⚠️ Please enter text first.")
            return
        _validate_script(active)
        with st.spinner("Checking…"):
            try:
                result = analyze_sentence(active)
                st.session_state["sc_last_result"] = result
            except MissingArtifactError as exc:
                st.error(f"🚨 {exc}")
                return
            except Exception as exc:
                st.error(f"Error: {exc}")
                return

    result = st.session_state.get("sc_last_result")
    if result is None:
        return

    st.markdown(
        _result_card("📄 Original", result["original"], "#1565c0"),
        unsafe_allow_html=True,
    )
    st.markdown(
        _result_card("✨ Corrected", result["corrected"], "#2e7d32"),
        unsafe_allow_html=True,
    )

    if result.get("misspellings"):
        st.markdown('<div class="sec-hdr">🔧 Corrections Detail</div>', unsafe_allow_html=True)
        for item in result["misspellings"]:
            with st.expander(f"❌ {item['original']}  →  ✅ {item['corrected']}"):
                st.write(f"**Original:** {item['original']}")
                st.write(f"**Best correction:** {item['corrected']}")
                if item.get("candidates"):
                    cand_df = pd.DataFrame(
                        [{"Candidate": c, "Log Score": f"{s:.4f}"} for c, s in item["candidates"][:8]]
                    )
                    st.dataframe(cand_df, hide_index=True, use_container_width=True)
    else:
        st.success("✅ No misspellings detected — all words are in the vocabulary.")


# ---------------------------------------------------------------------------
# Tab 4 — POS Analysis + Noisy Experiment
# ---------------------------------------------------------------------------

def tab_pos_analysis() -> None:
    st.markdown('<div class="sec-hdr">🏷️ POS Confidence Visualization</div>', unsafe_allow_html=True)

    pos_text = st.text_area(
        "Input", height=120,
        placeholder="Enter Telugu text for detailed POS analysis…",
        label_visibility="collapsed",
        key="pa_input",
    )
    pa1, pa2, _ = st.columns([2, 2, 6])
    with pa1:
        run_pa = st.button("▶ Analyze POS", type="primary", use_container_width=True, key="pa_run")
    with pa2:
        if st.button("📋 Sample", use_container_width=True, key="pa_sample"):
            st.session_state["pa_loaded"] = SAMPLE_SENTENCE
            st.rerun()

    if "pa_loaded" in st.session_state and not pos_text:
        pos_text = st.session_state.pop("pa_loaded")

    if run_pa:
        active = (pos_text or "").strip()
        if not active:
            st.warning("⚠️ Please enter text first.")
            return
        _validate_script(active)
        with st.spinner("Analyzing…"):
            try:
                result = analyze_sentence(active)
                st.session_state["pa_last_result"] = result
            except MissingArtifactError as exc:
                st.error(f"🚨 {exc}")
                return
            except Exception as exc:
                st.error(f"Error: {exc}")
                return

    result = st.session_state.get("pa_last_result")
    if result is None:
        return

    if result.get("tagged"):
        st.markdown('<div class="sec-hdr">📊 Per-Token Confidence</div>', unsafe_allow_html=True)
        st.markdown(_confidence_bars_html(result["tagged"]), unsafe_allow_html=True)

        st.markdown('<div class="sec-hdr">🔍 Per-Word Explainer</div>', unsafe_allow_html=True)
        for item in result["tagged"]:
            conf = item.get("confidence", 1.0)
            color = "#2e7d32" if conf >= 0.9 else "#e65100" if conf >= 0.6 else "#c62828"
            bar_w = int(conf * 100)
            with st.expander(f"{item['word']}  →  {item['tag']}  ({conf*100:.1f}%)"):
                st.markdown(
                    f"**Word:** {item['word']}  \n"
                    f"**POS Tag:** `{item['tag']}`  \n"
                    f"**English:** {translate_tag(item['tag'])}  \n"
                    f"**Telugu:** {translate_tag_telugu(item['tag'])}  \n"
                    f"**Confidence:** "
                    f"<span style='color:{color};font-weight:700'>{conf*100:.2f}%</span>",
                    unsafe_allow_html=True,
                )

    # ── Noisy Text Experiment ─────────────────────────────────────────────────
    st.markdown('<hr class="te-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">🧪 Noise Robustness Experiment</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:.85rem;color:#555;margin-top:-10px">'
        'Introduce controlled character-level noise and compare BiLSTM vs BiLSTM+Spellcheck.</p>',
        unsafe_allow_html=True,
    )

    noise_col1, noise_col2 = st.columns([4, 1])
    with noise_col1:
        noise_text = st.text_input(
            "Noise input",
            value=result["original"] if result else SAMPLE_SENTENCE,
            label_visibility="collapsed",
            key="noise_input",
        )
    with noise_col2:
        noise_level = st.slider("Noise %", 10, 80, 30, step=10, key="noise_level") / 100.0

    if st.button("🧪 Run Noise Experiment", type="secondary", key="noise_run"):
        active = (noise_text or "").strip()
        if not active:
            st.warning("Please enter text first.")
            return
        with st.spinner("Running noise experiment…"):
            try:
                model, word_vocab, tag_vocab, spellchecker, device = _load_artifacts()
                tokens = tokenize(active)
                noisy_tokens = introduce_noise(tokens, noise_level=noise_level)
                noisy_sentence = " ".join(noisy_tokens)

                # BiLSTM on noisy (no spellcheck)
                tags_noisy = predict_with_confidence(model, word_vocab, tag_vocab, noisy_tokens, device)

                # BiLSTM + Spellcheck on noisy
                sc_result = spellchecker.correct_sentence(noisy_tokens)
                corrected_tokens = sc_result["corrected_tokens"]
                tags_corrected = predict_with_confidence(
                    model, word_vocab, tag_vocab, corrected_tokens, device
                )

                st.markdown(
                    f'<div class="warn-banner">🔧 Noisy input: <em>{noisy_sentence}</em></div>',
                    unsafe_allow_html=True,
                )

                noisy_rows = [
                    {"Word (noisy)": w, "Tag": t, "Confidence": f"{c*100:.1f}%"}
                    for w, (t, c) in zip(noisy_tokens, tags_noisy)
                ]
                corr_rows = [
                    {"Word (corrected)": w, "Tag": t, "Confidence": f"{c*100:.1f}%"}
                    for w, (t, c) in zip(corrected_tokens, tags_corrected)
                ]

                cmp1, cmp2 = st.columns(2)
                with cmp1:
                    st.markdown("**BiLSTM only (noisy)**")
                    st.dataframe(pd.DataFrame(noisy_rows), hide_index=True, use_container_width=True)
                with cmp2:
                    st.markdown("**BiLSTM + Spellcheck**")
                    st.dataframe(pd.DataFrame(corr_rows), hide_index=True, use_container_width=True)

            except MissingArtifactError as exc:
                st.error(f"🚨 {exc}")
            except Exception as exc:
                st.error(f"Error: {exc}")


# ---------------------------------------------------------------------------
# Tab 5 — Model Comparison
# ---------------------------------------------------------------------------

def tab_model_comparison() -> None:
    st.markdown('<div class="sec-hdr">📊 Model Performance Comparison</div>', unsafe_allow_html=True)

    metrics_data = _load_eval_metrics()
    summary = metrics_data.get("summary", {})
    note = metrics_data.get("dataset_note", "")
    if note:
        st.info(f"ℹ️ {note}")

    # Build comparison table
    display_names = {
        "MostFrequentTag":       "Most Frequent Tag (Baseline)",
        "BiLSTM":                "BiLSTM Only",
        "BiLSTM_on_noisy":       "BiLSTM on Noisy Input",
        "BiLSTM_plus_spellcheck":"BiLSTM + Spellcheck",
    }
    rows = []
    for key, nice_name in display_names.items():
        m = summary.get(key)
        if m:
            rows.append({
                "Model": nice_name,
                "Accuracy": f"{m['accuracy']*100:.2f}%",
                "Precision": f"{m['precision']*100:.2f}%",
                "Recall": f"{m['recall']*100:.2f}%",
                "F1": f"{m['f1']:.4f}",
                "Support": int(m.get("support", 0)),
            })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)

    # Bar charts
    if summary:
        keys_ordered = [k for k in display_names if k in summary]
        short_names = {
            "MostFrequentTag":       "Baseline",
            "BiLSTM":                "BiLSTM",
            "BiLSTM_on_noisy":       "BiLSTM (noisy)",
            "BiLSTM_plus_spellcheck":"BiLSTM+SC",
        }
        acc_vals   = [summary[k]["accuracy"]  for k in keys_ordered]
        f1_vals    = [summary[k]["f1"]        for k in keys_ordered]
        prec_vals  = [summary[k]["precision"] for k in keys_ordered]
        rec_vals   = [summary[k]["recall"]    for k in keys_ordered]
        labels     = [short_names[k] for k in keys_ordered]

        chart_df = pd.DataFrame({
            "Model": labels,
            "Accuracy":  acc_vals,
            "F1":        f1_vals,
            "Precision": prec_vals,
            "Recall":    rec_vals,
        }).set_index("Model")

        st.markdown('<div class="sec-hdr">📈 Accuracy &amp; F1 Comparison</div>', unsafe_allow_html=True)
        st.bar_chart(chart_df[["Accuracy", "F1"]])

        st.markdown('<div class="sec-hdr">📈 Precision &amp; Recall Comparison</div>', unsafe_allow_html=True)
        st.bar_chart(chart_df[["Precision", "Recall"]])

    # Per-class metrics if available
    full_metrics = metrics_data.get("metrics", {})
    bilstm_report = full_metrics.get("BiLSTM", {}).get("report", {})
    if bilstm_report:
        st.markdown('<div class="sec-hdr">🏷️ Per-Class Metrics (BiLSTM)</div>', unsafe_allow_html=True)
        class_rows = []
        for tag, vals in bilstm_report.items():
            if isinstance(vals, dict) and "f1-score" in vals:
                class_rows.append({
                    "Tag": tag,
                    "Precision": f"{vals.get('precision',0)*100:.2f}%",
                    "Recall":    f"{vals.get('recall',0)*100:.2f}%",
                    "F1":        f"{vals.get('f1-score',0):.4f}",
                    "Support":   int(vals.get("support", 0)),
                })
        if class_rows:
            st.dataframe(pd.DataFrame(class_rows), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 6 — Batch Processing
# ---------------------------------------------------------------------------

def tab_batch_processing() -> None:
    st.markdown('<div class="sec-hdr">📦 Batch Processing</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:.88rem;color:#555;margin-top:-10px">'
        'Upload a TXT (one sentence per line) or CSV file and download results.</p>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload file",
        type=["txt", "csv"],
        key="batch_upload",
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.info("📂 Upload a TXT (one sentence per line) or CSV (first column = sentence).")
        return

    content = uploaded.read().decode("utf-8", errors="replace")
    if uploaded.name.endswith(".csv"):
        try:
            raw_df = pd.read_csv(io.StringIO(content), header=None)
            sentences = raw_df.iloc[:, 0].dropna().astype(str).tolist()
        except Exception:
            sentences = [line.strip() for line in content.splitlines() if line.strip()]
    else:
        sentences = [line.strip() for line in content.splitlines() if line.strip()]

    sentences = sentences[:200]  # Cap at 200 lines to avoid timeouts
    st.info(f"📄 Loaded **{len(sentences)}** sentence(s) from `{uploaded.name}`.")

    if st.button("🚀 Process All", type="primary", key="batch_run"):
        all_rows = []
        progress = st.progress(0)
        status_text = st.empty()

        try:
            model, word_vocab, tag_vocab, spellchecker, device = _load_artifacts()
        except MissingArtifactError as exc:
            st.error(f"🚨 {exc}")
            return

        for i, sent in enumerate(sentences):
            progress.progress((i + 1) / len(sentences))
            status_text.text(f"Processing sentence {i+1}/{len(sentences)}…")
            sent = sent.strip()
            if not sent:
                continue
            try:
                result = analyze_sentence(sent)
                for item in result.get("tagged", []):
                    all_rows.append({
                        "sentence_idx": i + 1,
                        "original_sentence": result["original"],
                        "corrected_sentence": result["corrected"],
                        "word": item["word"],
                        "tag": item["tag"],
                        "tag_english": translate_tag(item["tag"]),
                        "tag_telugu": translate_tag_telugu(item["tag"]),
                        "confidence": round(item.get("confidence", 1.0), 4),
                    })
            except Exception as exc:
                all_rows.append({
                    "sentence_idx": i + 1,
                    "original_sentence": sent,
                    "corrected_sentence": "",
                    "word": f"ERROR: {exc}",
                    "tag": "", "tag_english": "", "tag_telugu": "", "confidence": 0.0,
                })

        progress.empty()
        status_text.empty()
        st.success(f"✅ Processed {len(sentences)} sentence(s), {len(all_rows)} tokens total.")

        result_df = pd.DataFrame(all_rows)
        st.dataframe(result_df, hide_index=True, use_container_width=True)

        csv_buf = io.StringIO()
        result_df.to_csv(csv_buf, index=False, encoding="utf-8")
        exp1, exp2, _ = st.columns([2, 2, 4])
        with exp1:
            st.download_button(
                "⬇️ Download CSV",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name="batch_pos_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        json_rows = result_df.to_dict(orient="records")
        with exp2:
            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(json_rows, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="batch_pos_results.json",
                mime="application/json",
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Tab 7 — Model Info
# ---------------------------------------------------------------------------

def tab_model_info() -> None:
    st.markdown('<div class="sec-hdr">ℹ️ Model Architecture</div>', unsafe_allow_html=True)
    arch_rows = [
        ("Architecture",   "Bidirectional LSTM (BiLSTM)"),
        ("Embedding Dim",  "128"),
        ("Hidden Dim",     "128 (× 2 directions = 256 effective)"),
        ("Num Layers",     "1"),
        ("Dropout",        "0.3"),
        ("Pad Index",      "0"),
        ("Framework",      "PyTorch"),
        ("Device",         "CUDA if available, else CPU"),
        ("Max Seq Len",    "64 tokens"),
        ("Batch Size",     "8"),
        ("Epochs",         "25 (with early stopping, patience=6)"),
        ("Optimizer",      "Adam (lr=1e-3, weight_decay=1e-5)"),
        ("Gradient Clip",  "Norm ≤ 5.0"),
    ]
    for k, v in arch_rows:
        st.markdown(
            f'<div style="display:flex;gap:14px;padding:7px 0;border-bottom:1px solid #eee">'
            f'  <span style="font-weight:700;min-width:160px;color:#0d1b6e">{k}</span>'
            f'  <span style="color:#333">{v}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sec-hdr">📚 Dataset Information</div>', unsafe_allow_html=True)
    metrics_data = _load_eval_metrics()
    n_test = metrics_data.get("n_test_sentences", "—")
    n_tokens = metrics_data.get("n_test_tokens", "—")
    labels = metrics_data.get("labels", [])
    dataset_rows = [
        ("Language",         "Telugu (te) — Dravidian family"),
        ("Tag Set",          "Universal POS tags (UD v2)"),
        ("Train Split",      "70%"),
        ("Validation Split", "15%"),
        ("Test Split",       f"15% ({n_test} sentences, {n_tokens} tokens)"),
        ("Min Word Freq",    "1"),
        ("POS Classes",      str(len(labels)) + (f" — {', '.join(labels)}" if labels else "")),
        ("Spellchecker",     "Probabilistic (edit-distance + unigram log-prob)"),
        ("Max Edit Distance","2"),
    ]
    for k, v in dataset_rows:
        st.markdown(
            f'<div style="display:flex;gap:14px;padding:7px 0;border-bottom:1px solid #eee">'
            f'  <span style="font-weight:700;min-width:160px;color:#0d1b6e">{k}</span>'
            f'  <span style="color:#333">{v}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sec-hdr">🔬 Inference Pipeline</div>', unsafe_allow_html=True)
    st.markdown(_pipeline_html(), unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">🛠️ Tech Stack</div>', unsafe_allow_html=True)
    st.markdown(_tech_badges_html(), unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">📂 Model Artifact Paths</div>', unsafe_allow_html=True)
    from src.config import MODEL_PATH, VOCAB_PATH, SPELLCHECKER_PATH, EVAL_METRICS_PATH
    for name, path in [
        ("BiLSTM weights",   MODEL_PATH),
        ("Vocabularies",     VOCAB_PATH),
        ("Spellchecker",     SPELLCHECKER_PATH),
        ("Eval Metrics",     EVAL_METRICS_PATH),
    ]:
        exists = "✅" if path.exists() else "❌ (not found)"
        st.markdown(
            f'<div style="display:flex;gap:14px;padding:7px 0;border-bottom:1px solid #eee">'
            f'  <span style="font-weight:700;min-width:160px;color:#0d1b6e">{name}</span>'
            f'  <span style="color:#333;font-size:.88rem">{exists} &nbsp; <code>{path}</code></span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Telugu POS Tagging & Intelligent Spellchecking",
        page_icon="🪷",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_sidebar()

   
   
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Home",
        "Sentence Analysis",
        "Spellchecker",
        "POS Analysis",
        "Model Comparison",
        "Batch Processing",
        "Model Info",
        "Document Doctor",
    ])

    with tab1:
        tab_home()
    with tab2:
        tab_sentence_analysis()
    with tab3:
        tab_spellchecker()
    with tab4:
        tab_pos_analysis()
    with tab5:
        tab_model_comparison()
    with tab6:
        tab_batch_processing()
    with tab7:
        tab_model_info()
    with tab8:
        from app.document_doctor import render_document_doctor
        render_document_doctor()

    st.markdown(
        '<div class="footer">Telugu POS Tagging &amp; Intelligent Spellchecking · '
        'BiLSTM + Probabilistic Spellchecker · PyTorch · Streamlit</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
