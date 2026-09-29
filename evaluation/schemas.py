"""
schemas.py — Layer 0 output schema (layer0_plan.md, Step 2).

One PDFFeatures row per PDF. This is the `pdf_features` block that the
Layer 2 golden dataset schema (plan.md, Section 4) embeds, and what the
target model will receive at eval time.
"""

from pydantic import BaseModel
from typing import Literal


class PDFFeatures(BaseModel):
    pdf_id: str
    page_count: int
    doc_type: Literal["financial", "contract", "academic", "manual", "slide_deck", "scanned", "unknown"]
    doc_type_confidence: float
    table_density: Literal["high", "medium", "low"]
    table_density_score: float  # detected tables per page
    column_layout: Literal["single", "multi"]
    ocr_quality: Literal["n/a", "good", "poor"]
    avg_paragraph_length: float  # mean word count of paragraph blocks
    flagged_for_review: bool = False


class Layer1QA(BaseModel):
    """One surviving Layer 1 Q&A pair (evaluation/plan.md Section 4).

    Generated from a single strategy-agnostic structural passage; the
    generator answers using only that passage and a second pass verifies
    groundedness + single-passage sufficiency. Pairs failing either are
    discarded, so every row here is retrieval ground truth for Layer 2.
    """
    pdf_id: str
    question: str
    answer: str
    source_passage_id: str  # "<pdf_stem>:<page:04d>:<seq>"
    passage_page: int
    passage_section: str  # heading text of the passage's section ("" if none)
    passage_text: str  # exact passage text the Q&A was generated from
    doc_type: str  # from the Layer 0 row for this pdf_id
    generation_model: str
    faithfulness_passed: bool = True  # always True — failed pairs are discarded
