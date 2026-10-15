"""
RAG/QA Chunking Evaluation (Rate-Limited & OpenAI-Free)
--------------------------
Compares two chunking strategies (Structured_RAG vs Docling_Chunks) in Qdrant
using a RAG/QA evaluation set with RAGAS metrics without OpenAI dependency.
"""

import os
import json
import time
import tiktoken
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq as LlamaIndexGroq
from langchain_groq import ChatGroq
from llama_index.core import PromptTemplate
from llama_index.core.response_synthesizers import get_response_synthesizer

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_precision,
    context_recall,
    answer_relevancy,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import OllamaEmbeddings
from ragas.run_config import RunConfig  # <-- CRITICAL FOR RATE LIMIT PROTECTION

# ============================================================
# 1. CONFIG
# ============================================================

COLLECTION_A = "Structured_RAG"
COLLECTION_B = "Docling_Chunks"

# Retrieval Models
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
GEN_MODEL = "openai/gpt-oss-20b"         

# RAGAS Judge Models
JUDGE_MODEL = "openai/gpt-oss-120b"       
JUDGE_EMBED_MODEL = "mxbai-embed-large" 

# Rate Limiting Guardrails
DELAY_BETWEEN_PIPELINE_QUERIES = 2.0  
TOKEN_BUDGET = 2000          
FETCH_K_A = 20               
FETCH_K_B = 40               
EVAL_SET_PATH = "RAG/src/eval_set.jsonl"
OUTPUT_CSV = "RAG/data/chunking_eval_results.csv"

QA_PROMPT = """Answer the question using ONLY the context below.
If the context is insufficient, say "Insufficient context."

Context:
{context_str}

Question: {query_str}
Answer:"""

# ============================================================
# 2. SETUP MODELS + INDEXES
# ============================================================

print("[setup] initializing local models...")
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

# Verify API key presence early to avoid failing later
if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("CRITICAL: GROQ_API_KEY environment variable is not set!")

Settings.llm = LlamaIndexGroq(model=GEN_MODEL, temperature=0.0)

print("[setup] connecting to Qdrant...")
qdrant_client = QdrantClient()  

vs_a = QdrantVectorStore(collection_name=COLLECTION_A, client=qdrant_client)
vs_b = QdrantVectorStore(collection_name=COLLECTION_B, client=qdrant_client)

index_a = VectorStoreIndex.from_vector_store(vector_store=vs_a)
index_b = VectorStoreIndex.from_vector_store(vector_store=vs_b)

retriever_a = index_a.as_retriever(similarity_top_k=FETCH_K_A)
retriever_b = index_b.as_retriever(similarity_top_k=FETCH_K_B)

wrapped_prompt = PromptTemplate(QA_PROMPT)

synth_a = get_response_synthesizer(text_qa_template=wrapped_prompt, response_mode="compact")
synth_b = get_response_synthesizer(text_qa_template=wrapped_prompt, response_mode="compact")

enc = tiktoken.get_encoding("cl100k_base")

# ============================================================
# 3. TOKEN-BUDGET RETRIEVAL
# ============================================================

def retrieve_by_token_budget(retriever, query, token_budget=TOKEN_BUDGET):
    try:
        nodes = retriever.retrieve(query)
    except Exception as e:
        print(f"  [warn] retrieval failed: {e}")
        return []

    kept, used = [], 0
    for n in nodes:
        text = n.get_content() or ""
        t = len(enc.encode(text))
        if used + t > token_budget:
            break
        kept.append(n)
        used += t
    return kept

def answer_with(retriever, synthesizer, question):
    nodes = retrieve_by_token_budget(retriever, question)
    contexts = [n.get_content() for n in nodes]
    if not nodes:
        return "Insufficient context.", [], []
    try:
        response = synthesizer.synthesize(question, nodes=nodes)
        return str(response), contexts, nodes
    except Exception as e:
        print(f"  [warn] synthesis failed: {e}")
        return "Error generating answer.", contexts, nodes

# ============================================================
# 4. LOAD EVAL SET
# ============================================================

def load_eval_set(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[eval] loaded {len(rows)} questions from {path}")
    return rows

# ============================================================
# 5. RUN BOTH PIPELINES WITH RATE LIMIT PADDING
# ============================================================

def run_pipeline(retriever, synthesizer, rows, label):
    samples = []
    logs = []
    print(f"[run] {label}: {len(rows)} questions")
    for i, r in enumerate(rows, 1):
        q = r["question"]
        ans, ctxs, nodes = answer_with(retriever, synthesizer, q)
        tokens = sum(len(enc.encode(c)) for c in ctxs)
        samples.append(SingleTurnSample(
            user_input=q,
            response=ans,
            retrieved_contexts=ctxs,
            reference=r["ground_truth"],
        ))
        logs.append({
            "question": q,
            "answer": ans,
            "n_chunks": len(nodes),
            "n_tokens": tokens,
            "n_contexts": len(ctxs),
        })
        print(f"  [{label}] {i}/{len(rows)} | chunks={len(nodes)} tokens={tokens}")
        
        if DELAY_BETWEEN_PIPELINE_QUERIES > 0:
            time.sleep(DELAY_BETWEEN_PIPELINE_QUERIES)
            
    return samples, pd.DataFrame(logs)
# ============================================================
# 6. RAGAS EVALUATION WITH GROQ 'N=1' BUG FIX
# ============================================================
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
    AnswerRelevancy,
)

def score_with_ragas(samples, label):
    print(f"[ragas] scoring {label} via Groq...")

    # 1. Initialize ChatGroq with strict parameter boundaries
    groq_judge = ChatGroq(
        model=JUDGE_MODEL, 
        temperature=0,
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        # N=1 overrides are sometimes ignored downstream by wrappers; 
        # we configure metric overrides below to be absolutely safe
    )

    judge_llm = LangchainLLMWrapper(groq_judge)
    judge_emb = LangchainEmbeddingsWrapper(Settings.embed_model)

    # 2. FIX: Initialize the metric objects and explicitly override their internal LLM wrappers to use n=1
    m_faithfulness = Faithfulness(llm=judge_llm)
    m_context_precision = ContextPrecision(llm=judge_llm)
    m_context_recall = ContextRecall(llm=judge_llm)
    
    # Answer Correctness and Relevancy default to n > 1. We force n=1 on their internal structures.
    m_answer_correctness = AnswerCorrectness(llm=judge_llm)
    m_answer_relevancy = AnswerRelevancy(llm=judge_llm)
    
    # Ragas metric subcomponents use specific configs depending on version:
    if hasattr(m_answer_correctness, 'llm'):
        # Force the metric to compute generation loops sequentially rather than natively requesting n > 1
        m_answer_correctness.llm.n = 1
    if hasattr(m_answer_relevancy, 'llm'):
        m_answer_relevancy.llm.n = 1

    # Force strict serial thread execution to completely avoid Groq free tier 429 locks
    custom_run_config = RunConfig(max_workers=1, timeout=120)

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[
            m_faithfulness,
            m_answer_correctness,
            m_context_precision,
            m_context_recall,
            m_answer_relevancy,
        ],
        llm=judge_llm,
        embeddings=judge_emb,
        raise_exceptions=False,
        run_config=custom_run_config
    )
    
    df = result.to_pandas()
    
    # Clean data mapping logic
    column_mapping = {col: f"{col}_{label}" for col in df.columns}
    df = df.rename(columns=column_mapping)
    return df

# ============================================================
# 7. PAIRED COMPARISON
# ============================================================

METRICS = [
    "answer_correctness",
    "faithfulness",
    "context_recall",
    "context_precision",
    "answer_relevancy",
]

WEIGHTS = {
    "answer_correctness": 0.35,
    "faithfulness": 0.25,
    "context_recall": 0.20,
    "context_precision": 0.10,
    "answer_relevancy": 0.10,
}

def paired_report(df_a, df_b, logs_a, logs_b):
    print("\n" + "=" * 70)
    print("PAIRED COMPARISON  (A = " + COLLECTION_A + " | B = " + COLLECTION_B + ")")
    print("=" * 70)

    n = min(len(df_a), len(df_b))
    df_a, df_b = df_a.iloc[:n].reset_index(drop=True), df_b.iloc[:n].reset_index(drop=True)

    summary_rows = []
    for m in METRICS:
        a = df_a[f"{m}_A"].astype(float).values
        b = df_b[f"{m}_B"].astype(float).values
        diff = a - b

        wins = int((diff > 0.05).sum())
        losses = int((diff < -0.05).sum())
        ties = int(n - wins - losses)

        try:
            _, p = wilcoxon(a, b)
        except Exception:
            p = float("nan")

        summary_rows.append({
            "metric": m,
            "A_mean": np.nanmean(a),
            "B_mean": np.nanmean(b),
            "mean_diff (A-B)": np.nanmean(diff),
            "A_wins": wins,
            "B_wins": losses,
            "ties": ties,
            "wilcoxon_p": p,
        })

    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    score_a = sum(WEIGHTS[m] * df_a[f"{m}_A"].astype(float) for m in METRICS)
    score_b = sum(WEIGHTS[m] * df_b[f"{m}_B"].astype(float) for m in METRICS)
    print(f"\nComposite score A ({COLLECTION_A}): {np.nanmean(score_a):.3f}")
    print(f"Composite score B ({COLLECTION_B}): {np.nanmean(score_b):.3f}")

    winner = COLLECTION_A if np.nanmean(score_a) > np.nanmean(score_b) else COLLECTION_B
    print(f"\n>>> WINNER: {winner}")

    print("\n--- Retrieval footprint ---")
    print(f"A: mean chunks/query = {logs_a['n_chunks'].mean():.2f}, "
          f"mean tokens/query = {logs_a['n_tokens'].mean():.0f}")
    print(f"B: mean chunks/query = {logs_b['n_chunks'].mean():.2f}, "
          f"mean tokens/query = {logs_b['n_tokens'].mean():.0f}")

    out = pd.concat([df_a, df_b], axis=1)
    out["question"] = logs_a["question"].values
    out["score_A"] = score_a.values
    out["score_B"] = score_b.values
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[save] per-question results written to {OUTPUT_CSV}")
    return summary, score_a, score_b

def main():
    rows = load_eval_set(EVAL_SET_PATH)
    samples_a, logs_a = run_pipeline(retriever_a, synth_a, rows, "A")
    samples_b, logs_b = run_pipeline(retriever_b, synth_b, rows, "B")
    df_a = score_with_ragas(samples_a, "A")
    df_b = score_with_ragas(samples_b, "B")
    paired_report(df_a, df_b, logs_a, logs_b)

if __name__ == "__main__":
    main()