"""Streamlit app for Lay Language AI.

AI assistance: OpenAI Codex helped generate this project code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
LOGO_PATH = PROJECT_ROOT / "lay_language.png"


SAMPLE_TEXT = (
    "MRI demonstrates mild bilateral neural foraminal stenosis at L4-L5 "
    "without evidence of acute cord compression."
)


@st.cache_resource(show_spinner=False)
def load_rewriter():
    from lay_language_ai.inference import Rewriter

    return Rewriter.from_default_paths()


def main() -> None:
    st.set_page_config(page_title="Lay Language AI", page_icon="LL", layout="wide")

    st.markdown(
        """
        <style>
        :root {
            --lay-navy: #062f5f;
            --lay-teal: #12aaa4;
            --lay-teal-dark: #078d89;
            --lay-ink: #172033;
            --lay-muted: #667085;
            --lay-surface: #f6fbfb;
            --lay-border: #d8e7e7;
        }

        .stApp {
            background:
                linear-gradient(180deg, #ffffff 0%, var(--lay-surface) 100%);
            color: var(--lay-ink);
        }

        .block-container {
            max-width: 1080px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3 {
            color: var(--lay-navy);
            letter-spacing: 0;
        }

        .brand-row {
            display: flex;
            align-items: center;
            gap: 1.1rem;
            margin-bottom: 1.4rem;
        }

        .brand-copy h1 {
            margin: 0;
            font-size: 2.3rem;
            line-height: 1.05;
        }

        .brand-copy p {
            margin: .35rem 0 0;
            color: var(--lay-muted);
            font-size: 1.02rem;
        }

        .stTextArea textarea {
            background: #ffffff;
            color: var(--lay-ink);
            border-color: var(--lay-border);
            border-radius: 8px;
            line-height: 1.5;
        }

        .stTextArea textarea::placeholder {
            color: #7b8794;
        }

        .stTextArea textarea:focus {
            background: #ffffff;
            color: var(--lay-ink);
            border-color: var(--lay-teal);
            box-shadow: 0 0 0 1px var(--lay-teal);
        }

        .stTextArea label {
            color: var(--lay-navy);
            font-weight: 700;
        }

        .stButton button {
            background: var(--lay-teal);
            border: 1px solid var(--lay-teal);
            border-radius: 8px;
            color: #ffffff;
            font-weight: 700;
            padding: .55rem 1.25rem;
        }

        .stButton button:hover {
            background: var(--lay-teal-dark);
            border-color: var(--lay-teal-dark);
            color: #ffffff;
        }

        .stAlert {
            border-radius: 8px;
        }

        .disclaimer {
            margin-top: 1.5rem;
            border: 1px solid #f1c36d;
            border-left: 5px solid #d98d00;
            border-radius: 8px;
            background: #fff8e8;
            color: #332400;
            padding: .85rem 1rem;
            font-size: .95rem;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    logo_col, title_col = st.columns([1, 3.5], vertical_alignment="center")
    with logo_col:
        st.image(str(LOGO_PATH), width="stretch")
    with title_col:
        st.markdown(
            """
            <div class="brand-copy">
              <h1>Complex medical terms. Plain English.</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

    medical_text = st.text_area(
        "Add your diagnosis, operative report, or other medical text here.",
        value=SAMPLE_TEXT,
        placeholder="Paste a short medical report sentence or paragraph.",
        height=180,
    )

    generate = st.button("Rewrite", type="primary", use_container_width=False)

    if generate:
        if not medical_text.strip():
            st.warning("Paste medical text before rewriting.")
            return

        rewriter = load_rewriter()
        with st.spinner("Rewriting..."):
            comparison = rewriter.compare(medical_text)

        left, right = st.columns(2, gap="large")
        with left:
            with st.container(border=True):
                st.subheader("Base Model")
                st.write(comparison.base_output)
        with right:
            with st.container(border=True):
                st.subheader("Fine-Tuned Model")
                st.write(comparison.adapted_output)

    st.markdown(
        """
        <div class="disclaimer">
          Educational prototype only. This tool does not provide medical advice and should not replace a clinician.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
