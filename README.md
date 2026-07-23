# pm-playbook
# PM Playbook

A RAG application for the [DataTalksClub LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp).

## Problem

Lenny's Podcast features world-class product leaders, but actionable advice is buried in 300+ long-form transcripts. PM Playbook lets users ask natural-language questions and get synthesized, cited answers.

## Quick Start

```bash
uv sync
export OPENAI_API_KEY="your-key"
python -m pm_playbook.ingest
streamlit run streamlit_app/Home.py