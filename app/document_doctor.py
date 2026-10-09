
"""Document Doctor: review Telugu documents using the existing NLP pipeline."""

import json
import re

import streamlit as st
from src.inference import analyze_sentence


def split_sentences(text: str) -> list[str]:
    """Split a document into non-empty sentence-like units."""
    if not text or not text.strip():
        return []

    parts = re.split(r"(?<=[.!?।])\s+|\r?\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def render_document_doctor() -> None:
    st.header("📝 Telugu Document Doctor")
    st.write(
        "Upload a Telugu text document to review spelling corrections "
        "and POS predictions sentence by sentence."
    )

    uploaded = st.file_uploader(
        "Upload a UTF-8 .txt document",
        type=["txt"],
        key="document_doctor_upload",
    )

    if uploaded is None:
        st.info("Upload a .txt file to begin.")
        return

    try:
        original_text = uploaded.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError:
        st.error("Could not read the file as UTF-8. Save it with UTF-8 encoding.")
        return

    if not original_text.strip():
        st.warning("The uploaded document is empty.")
        return

    st.text_area(
        "Original document",
        value=original_text,
        height=180,
        disabled=True,
        key="document_doctor_original",
    )

    sentences = split_sentences(original_text)
    st.caption(f"Detected text units: {len(sentences)}")

    if not sentences:
        st.warning("No text sentences were found.")
        return

    if st.button("🔍 Analyze Document", key="document_doctor_analyze"):
        results = []
        errors = []

        with st.spinner("Analyzing document..."):
            for index, sentence in enumerate(sentences, start=1):
                try:
                    results.append({
                        "sentence_number": index,
                        "original": sentence,
                        "analysis": analyze_sentence(sentence),
                    })
                except Exception:
                    # Keep the document usable if one sentence fails.
                    errors.append({
                        "sentence_number": index,
                        "original": sentence,
                        "error": "This sentence could not be analyzed.",
                    })

        st.session_state["document_doctor_results"] = results
        st.session_state["document_doctor_errors"] = errors
        st.session_state["document_doctor_source"] = original_text

    if st.session_state.get("document_doctor_source") != original_text:
        return

    results = st.session_state.get("document_doctor_results", [])
    errors = st.session_state.get("document_doctor_errors", [])

    if results:
        st.subheader("Analysis Results")

        for item in results:
            analysis = item["analysis"]

            with st.expander(
                f"Sentence {item['sentence_number']}: {item['original']}"
            ):
                st.write("**Original:**", analysis["original"])
                st.write("**Suggested corrected text:**", analysis["corrected"])

                if analysis["misspellings"]:
                    st.write("**Detected misspellings:**")
                    st.json(analysis["misspellings"])
                else:
                    st.write("No misspellings reported by the spellchecker.")

                st.write("**POS predictions:**")
                st.dataframe(analysis["tagged"], use_container_width=True)

        # Start from the original sentences, then apply only explicitly
        # selected corrections.
        st.subheader("Review Corrections")

        accepted = {}
        for item in results:
            analysis = item["analysis"]
            if analysis["original"] != analysis["corrected"]:
                accepted[item["sentence_number"]] = st.checkbox(
                    f"Accept suggestion for sentence {item['sentence_number']}: "
                    f"{analysis['corrected']}",
                    value=False,
                    key=f"document_doctor_accept_{item['sentence_number']}",
                )

        corrected_sentences = []
        for item in results:
            number = item["sentence_number"]
            analysis = item["analysis"]
            if accepted.get(number, False):
                corrected_sentences.append(analysis["corrected"])
            else:
                corrected_sentences.append(item["original"])

        corrected_document = "\n".join(corrected_sentences)
        st.text_area(
            "Reviewed document",
            value=corrected_document,
            height=180,
            key="document_doctor_reviewed",
        )

        st.download_button(
            "⬇️ Download reviewed document",
            data=corrected_document,
            file_name="telugu_document_reviewed.txt",
            mime="text/plain; charset=utf-8",
        )

        report = {
            "results": results,
            "errors": errors,
        }
        st.download_button(
            "⬇️ Download JSON analysis report",
            data=json.dumps(report, ensure_ascii=False, indent=2, default=str),
            file_name="telugu_document_analysis.json",
            mime="application/json",
        )

    if errors:
        st.warning(f"{len(errors)} sentence(s) could not be analyzed.")
        st.json(errors)
