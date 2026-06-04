# Psychiatric-Medication-Review-RAG-Assistant
RAG system for psychiatric medication Q&amp;A — FAISS vector search,  MiniLM embeddings, hybrid retrieval, MRR 1.0 on evaluation set
[https://colab.research.google.com/github.com/dacamas/Psychiatric-Medication-Review-RAG-Assistant/blob/main/Psychiatric_Medication_Review_RAG_Assistant.ipynb
](https://colab.reserach.google.com/github/dacamas/Psychiatric-Medication-Review-RAG-Assistant/blob/main/Psychiatric_Medication_Review_RAG_Assistant.ipynb)
🧠 Psychiatric Medication Review RAG Assistant
A production-style Retrieval-Augmented Generation (RAG) system that answers
questions about psychiatric medications using real patient reviews — not hallucinated
model knowledge.
Built end-to-end in Python, structured as a modular package, and fully runnable in Google Colab.
📊 Evaluation Results (Actual Run)
These are real numbers from the evaluation framework, not estimates.
Metric Score Notes
MRR 1.000 First result is always relevant
Precision@3 0.917 9/10 top-3 results are on-topic
Precision@5 0.900 Stays strong at top-5
Recall@5 1.000 Never misses a relevant review
Groundedness 0.692 69% of answer tokens sourced
from retrieved reviews
Citation Compliance 1.000 Every answer includes a source
citation
Context Relevance 0.646 Retrieved reviews genuinely
relate to the query
Semantic Similarity 0.485 Answers stay in correct
semantic neighbourhood
Note on generation quality: The LLM layer uses google/flan-t5-large (780M
params) for Colab free-tier compatibility. Answer fluency improves significantly with a
larger model (Mistral-7B, Llama-3-8B). The retrieval architecture is fully modelagnostic — swap llm_name in ImprovedRAGPipeline to upgrade.
🏗️ Architecture
User Query (natural language)
 │
 ▼
┌─────────────────────────────┐
│ EmbeddingPipeline │ all-MiniLM-L6-v2 (384-dim, L2-normalised)
└──────────────┬──────────────┘
 │ query vector
 ▼
┌─────────────────────────────┐
│ VectorStore (FAISS) │ IndexFlatIP — exact cosine similarity
│ 31,397 review vectors │ + metadata: drug, condition, rating, text
└──────────────┬──────────────┘
 │ top-K reviews + similarity scores
 ▼
┌─────────────────────────────┐
│ Prompt Builder │ [REVIEW N] structured context blocks
│ (citation-enforced) │ + grounding instruction
└──────────────┬──────────────┘
 │
 ▼
┌─────────────────────────────┐
│ FLAN-T5 Generator │ Deterministic decoding (do_sample=False)
│ (flan-t5-large) │ repetition_penalty=1.3
└──────────────┬──────────────┘
 │
 ▼
 Answer + [REVIEW N] citations + source breakdown
💬 Example Outputs
Sertraline Side Effects
Q: What side effects do patients commonly report with sertraline?
ANSWER:
[REVIEW 1] Sertraline (Rating 10.0/10): "i was prescribed 50mg for the
first week and 100 every day afterwards. after the sertraline took effect
i went from having usually daily anxiety attacks to having one approximately
every 2 months..."
Sources:
 [REVIEW 1] Sertraline | Generalized Anxiety Disorder | Score 0.7649
 [REVIEW 2] Sertraline | Anxiety and Stress | Score 0.7170
 [REVIEW 3] Sertraline | Depression | Score 0.7062
 [REVIEW 4] Sertraline | Anxiety and Stress | Score 0.7026
 [REVIEW 5] Sertraline | Depression | Score 0.7480
Lithium Complaints
Q: What are common complaints about lithium?
ANSWER:
[REVIEW 1] Lithium (Rating 9.0/10): "my only two complaints about lithium
are the potentially detrimental effects on the kidneys and the nausea."
Sources:
 [REVIEW 1] Lithium | Bipolar Disorder | Score 0.6541
 [REVIEW 2] Lithium | Cluster Headaches | Score 0.6490
 [REVIEW 3] Lithium | Bipolar Disorder | Score 0.6905
Hybrid vs Pure Semantic Search
Query: "anxiety medication weight gain problems"
[Pure Semantic] → Bupropion, Quetiapine, Propranolol (broad anxiolytics)
[Hybrid Search] → Sertraline, Zoloft, Paxil (SSRIs, more specific)
The keyword component boosted SSRIs where patients specifically mentioned weight alongside
anxiety — exactly what the query intended.
🔍 Why Hybrid Search Matters
Pure semantic search finds topically related reviews but can miss specific drug names or symptoms
that appear verbatim in the query. Hybrid search combines semantic similarity (α=0.7) with
keyword overlap (α=0.3):
hybrid_score = 0.7 × semantic_score + 0.3 × keyword_overlap
In testing this surfaced more clinically specific results — e.g. for a fatigue query, semantic-only
returned stimulants (associated with fatigue treatment) while hybrid correctly returned
antidepressants with fatigue as a reported side effect.
📦 Tech Stack
Component Technology Why
Embeddings all-MiniLM-L6-v2 Fast, 384-dim, BEIR
benchmark top performer at
22M params
Vector store FAISS IndexFlatIP Exact cosine search, no
approximation error for N <
500k
LLM google/flan-t5-large Colab free-tier compatible,
instruction-tuned
Dataset UCI Drug Review (215k
reviews)
Real patient reviews,
Drugs.com
Retrieval Dense + hybrid (semantic +
BM25-style)
Better specificity than semantic
alone
Evaluation ROUGE, MRR, Precision@K,
groundedness
Multi-layer quality assessment
🗂️ Project Structure
psychiatric_rag/
├── data_processing.py # Load, clean, filter, EDA
├── embedding_pipeline.py # SentenceTransformer encoding + disk caching
├── vector_store.py # FAISS index build / save / load / search
├── retriever.py # Query interface, hybrid search, drug comparison
├── improved_rag_pipeline.py # Citation-enforced RAG with FLAN-T5
├── evaluation.py # Precision@K, Recall@K, MRR, groundedness, ROUGE
├── demo.py # End-to-end demo (all example queries)
├── psychiatric_rag.ipynb # Complete runnable Colab notebook
├── requirements.txt
└── README.md
🚀 Quickstart
Option A — Google Colab (recommended)
1. Open psychiatric_rag.ipynb in Google Colab
2. Run Cell 1 (installs packages) → Runtime → Restart session
3. Run all remaining cells top to bottom
4. Everything is self-contained — dataset downloads automatically
Option B — Local
git clone https://github.com/YOUR_USERNAME/psychiatric-rag
cd psychiatric-rag
pip install numpy==1.26.4
pip install faiss-cpu==1.7.4
pip install -r requirements.txt
python demo.py
Python version note: Requires Python 3.10–3.11. NumPy must be pinned to 1.26.4
because faiss-cpu is compiled against NumPy 1.x.
Quick API usage
from embedding_pipeline import EmbeddingPipeline
from vector_store import VectorStore, build_metadata_from_df
from retriever import Retriever
from improved_rag_pipeline import ImprovedRAGPipeline
from data_processing import load_dataset, preprocess_dataframe,
filter_psychiatric
# Build index (first time only — cached on subsequent runs)
emb = EmbeddingPipeline()
df = filter_psychiatric(preprocess_dataframe(load_dataset()))
embs = emb.encode_with_cache(df["review"].tolist())
vs = VectorStore(embedding_dim=384)
vs.build_index(embs, build_metadata_from_df(df))
vs.save()
# Query
ret = Retriever(emb, vs)
rag = ImprovedRAGPipeline(ret)
result = rag.answer("What side effects does sertraline cause?")
print(result["answer"])
print(result["citations_found"])
📊 Evaluation Framework
Three evaluation layers run automatically in the notebook:
A — Retrieval Quality
• Precision@K — what fraction of top-K results are relevant?
• Recall@K — what fraction of all relevant reviews appear in top-K?
• MRR — Mean Reciprocal Rank of the first relevant result
B — Generation Grounding
• Groundedness — token overlap between answer and retrieved reviews
• Context Relevance — do retrieved reviews actually relate to the query?
• Citation Compliance — does the answer include [REVIEW N] references?
• Answer Relevance — does the answer address the question's key terms?
C — Text Overlap
• ROUGE-1 / ROUGE-L — n-gram overlap with reference answers
• Semantic Similarity — cosine similarity between answer and reference embeddings
⚠️ Known Limitations
Semantic drift on intent-opposite queries A query like "antidepressants that cause fatigue" can
retrieve stimulants that treat fatigue because the embedding model captures topic similarity but not
causal direction. A reranker or query expansion step would address this.
Generation quality ceiling flan-t5-large (780M params) follows simple summarisation instructions
but struggles with multi-review synthesis and consistent citation formatting. Swap to Mistral-7B or
Llama-3-8B for production-quality generation — no other code changes required.
Colab session persistence The FAISS index is saved to /tmp which is wiped between Colab
sessions. The notebook detects this automatically and rebuilds the index on next run.
🔮 Future Improvements
Improvement Impact
Replace flan-t5 with Mistral-7B via HF
Inference API
High — much better answer quality, free
Add a cross-encoder reranker (e.g. crossencoder/ms-marco-MiniLM-L-6-v2)
High — fixes semantic drift failures
Fine-tune MiniLM on drug review sentence
pairs
Medium — domain-specific embeddings
IndexIVFFlat for corpora > 500k reviews Medium — sub-linear search at scale
FastAPI REST endpoint Medium — production deployment
Streamlit / Gradio UI Low — interactive demo
🎓 Skills Demonstrated
Skill Area Where
Dense retrieval & vector search EmbeddingPipeline + FAISS
Hybrid search design VectorStore.hybrid_search()
LLM prompt engineering Grounding + citation enforcement
RAG architecture Full retriever → reader → grounding pipeline
Retrieval evaluation MRR, Precision@K, Recall@K
Generation evaluation Groundedness, ROUGE, semantic similarity
NLP preprocessing Text cleaning, filtering, EDA
Modular ML package design 6 importable modules with clean APIs
Reproducibility Cached embeddings, pinned deps, auto-rebuild
📄 Dataset
UCI Drug Review Dataset
• 215,063 patient reviews from Drugs.com
• Fields: drug name, condition, review text, rating (1–10), useful count
• Source: UCI ML Repository
• License: Creative Commons Attribution 4.0
After filtering to psychiatric medications and conditions: 31,397 reviews across depression, anxiety,
ADHD, bipolar disorder, schizophrenia, PTSD, OCD, and panic disorder.
⚠️ Disclaimer
This project is for educational and portfolio demonstration purposes only. It does not constitute
medical advice. Always consult a qualified healthcare professional before making any medication
decisions.
📬 Contact
Built by Dylan Camas
