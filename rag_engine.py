import os
import re
import json
import pickle
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Generator
import pypdf
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from groq import Groq

import config

class RAGEngine:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.groq_client = Groq(api_key=self.api_key)
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_corpus_tokens: List[List[str]] = []
        
        # Dense Semantic Search Components (TF-IDF + LSA)
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None
        self.dense_embeddings: Optional[np.ndarray] = None
        
        # Optional SentenceTransformer support
        self.st_model = None
        self.st_embeddings = None
        
        self.is_indexed = False
        self.indexing_stats = {
            "total_documents": 0,
            "total_pages": 0,
            "total_chunks": 0,
            "reports": []
        }

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric words."""
        return [w.lower() for w in re.findall(r'\b[A-Za-z0-9\$\₹\%\.\,\-]+\b', text) if len(w) > 1]

    def load_or_build_index(self, force_rebuild: bool = False):
        """Loads cached chunks and indexes, or builds them from PDFs."""
        if not force_rebuild and config.CHUNKS_FILE.exists() and config.INDEX_CACHE_FILE.exists():
            print("[RAG] Loading cached chunks and indexes from disk...")
            try:
                with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                with open(config.INDEX_CACHE_FILE, "rb") as f:
                    cache_data = pickle.load(f)
                    self.bm25 = cache_data.get("bm25")
                    self.bm25_corpus_tokens = cache_data.get("bm25_tokens", [])
                    self.tfidf_vectorizer = cache_data.get("tfidf_vectorizer")
                    self.svd = cache_data.get("svd")
                    self.dense_embeddings = cache_data.get("dense_embeddings")
                    self.indexing_stats = cache_data.get("indexing_stats", self.indexing_stats)
                
                self.is_indexed = True
                print(f"[RAG] Successfully loaded {len(self.chunks)} chunks from cache.")
                return
            except Exception as e:
                print(f"[RAG] Error loading cache ({e}), rebuilding index...")

        self.build_index_from_pdfs()

    def build_index_from_pdfs(self):
        """Extracts text from PDF documents, chunks them, and builds BM25 and dense indices."""
        print("[RAG] Parsing PDF documents and building index...")
        self.chunks = []
        total_pages = 0
        reports_meta = []

        for doc_cfg in config.PDF_DOCS:
            pdf_path = doc_cfg["path"]
            if not pdf_path.exists():
                print(f"[RAG] Warning: {pdf_path} not found.")
                continue

            print(f"[RAG] Processing {doc_cfg['title']} ({doc_cfg['filename']})...")
            reader = pypdf.PdfReader(str(pdf_path))
            doc_pages = len(reader.pages)
            total_pages += doc_pages
            doc_chunks_count = 0

            reports_meta.append({
                "filename": doc_cfg["filename"],
                "title": doc_cfg["title"],
                "fiscal_year": doc_cfg["fiscal_year"],
                "theme": doc_cfg["theme"],
                "pages": doc_pages
            })

            for page_idx, page in enumerate(reader.pages):
                page_num = page_idx + 1
                try:
                    raw_text = page.extract_text() or ""
                except Exception as err:
                    print(f"Error extracting page {page_num} in {doc_cfg['filename']}: {err}")
                    raw_text = ""

                cleaned = self._clean_page_text(raw_text)
                if not cleaned or len(cleaned) < 50:
                    continue

                page_chunks = self._chunk_page(cleaned, doc_cfg, page_num)
                self.chunks.extend(page_chunks)
                doc_chunks_count += len(page_chunks)

            print(f"[RAG] Indexed {doc_cfg['filename']}: {doc_pages} pages -> {doc_chunks_count} chunks.")

        self.indexing_stats = {
            "total_documents": len(reports_meta),
            "total_pages": total_pages,
            "total_chunks": len(self.chunks),
            "reports": reports_meta
        }

        print(f"[RAG] Total chunks created: {len(self.chunks)}. Building indices...")

        # 1. Build BM25 Index
        self.bm25_corpus_tokens = [self.tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.bm25_corpus_tokens)

        # 2. Build TF-IDF + TruncatedSVD Dense Semantic Index
        texts = [c["text"] for c in self.chunks]
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=2,
            sublinear_tf=True
        )
        tfidf_mat = self.tfidf_vectorizer.fit_transform(texts)
        n_components = min(256, tfidf_mat.shape[1] - 1) if tfidf_mat.shape[1] > 10 else 10
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        dense_embs = self.svd.fit_transform(tfidf_mat)
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(dense_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.dense_embeddings = dense_embs / norms

        self.is_indexed = True

        # Cache to disk for instant restarts
        try:
            with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False)
            with open(config.INDEX_CACHE_FILE, "wb") as f:
                pickle.dump({
                    "bm25": self.bm25,
                    "bm25_tokens": self.bm25_corpus_tokens,
                    "tfidf_vectorizer": self.tfidf_vectorizer,
                    "svd": self.svd,
                    "dense_embeddings": self.dense_embeddings,
                    "indexing_stats": self.indexing_stats
                }, f)
            print("[RAG] Cached chunks and vector indexes to disk.")
        except Exception as e:
            print(f"[RAG] Error writing cache: {e}")

    def _clean_page_text(self, text: str) -> str:
        """Cleans and standardizes extracted page text."""
        # Normalize recurring footer/header lines
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            line_str = line.strip()
            # Skip lone page numbers or blank lines
            if not line_str:
                continue
            cleaned_lines.append(line_str)
        return "\n".join(cleaned_lines)

    def _detect_section(self, lines: List[str]) -> str:
        """Detects likely section title from the first non-empty lines."""
        for line in lines[:4]:
            if 3 < len(line) < 80 and not line.isdigit():
                return line
        return "General"

    def _chunk_page(self, text: str, doc_cfg: Dict[str, Any], page_num: int) -> List[Dict[str, Any]]:
        """Splits page text into coherent chunks with overlap."""
        lines = text.split("\n")
        section = self._detect_section(lines)
        
        # Rejoin text with spaces or newlines
        full_page_text = "\n".join(lines)
        
        chunks = []
        chunk_size = config.CHUNK_SIZE
        overlap = config.CHUNK_OVERLAP

        # If page is short enough, keep as single chunk
        if len(full_page_text) <= chunk_size + 100:
            chunks.append({
                "id": f"{doc_cfg['filename']}_p{page_num}_c1",
                "filename": doc_cfg["filename"],
                "report_title": doc_cfg["title"],
                "fiscal_year": doc_cfg["fiscal_year"],
                "page": page_num,
                "section": section,
                "text": full_page_text
            })
            return chunks

        # Sliding window chunking with sentence/paragraph boundary preference
        start = 0
        chunk_idx = 1
        text_len = len(full_page_text)

        while start < text_len:
            end = start + chunk_size
            if end >= text_len:
                chunk_slice = full_page_text[start:]
                chunks.append({
                    "id": f"{doc_cfg['filename']}_p{page_num}_c{chunk_idx}",
                    "filename": doc_cfg["filename"],
                    "report_title": doc_cfg["title"],
                    "fiscal_year": doc_cfg["fiscal_year"],
                    "page": page_num,
                    "section": section,
                    "text": chunk_slice.strip()
                })
                break
            
            # Find closest sentence or newline break
            break_idx = full_page_text.rfind("\n", start, end)
            if break_idx == -1 or break_idx < start + (chunk_size // 2):
                break_idx = full_page_text.rfind(". ", start, end)
                if break_idx != -1 and break_idx >= start + (chunk_size // 2):
                    break_idx += 1
                else:
                    break_idx = end
            
            chunk_slice = full_page_text[start:break_idx]
            if len(chunk_slice.strip()) > 60:
                chunks.append({
                    "id": f"{doc_cfg['filename']}_p{page_num}_c{chunk_idx}",
                    "filename": doc_cfg["filename"],
                    "report_title": doc_cfg["title"],
                    "fiscal_year": doc_cfg["fiscal_year"],
                    "page": page_num,
                    "section": section,
                    "text": chunk_slice.strip()
                })
                chunk_idx += 1
            
            start = max(start + 1, break_idx - overlap)

        return chunks

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        fiscal_year: Optional[str] = "all"
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k chunks for a query using hybrid, semantic, or bm25 search.
        Reciprocal Rank Fusion (RRF) is used for hybrid combination.
        """
        if not self.is_indexed or not self.chunks:
            return []

        # Filter chunk indices by fiscal year if requested
        valid_indices = list(range(len(self.chunks)))
        if fiscal_year and fiscal_year != "all":
            valid_indices = [
                i for i in valid_indices
                if self.chunks[i]["fiscal_year"] == fiscal_year
            ]
            if not valid_indices:
                valid_indices = list(range(len(self.chunks)))

        q_tokens = self.tokenize(query)
        bm25_scores = {}
        if self.bm25:
            all_bm25 = self.bm25.get_scores(q_tokens)
            for i in valid_indices:
                bm25_scores[i] = float(all_bm25[i])

        # Semantic scores via TF-IDF + SVD
        semantic_scores = {}
        if self.tfidf_vectorizer and self.svd and self.dense_embeddings is not None:
            q_vec = self.tfidf_vectorizer.transform([query])
            q_dense = self.svd.transform(q_vec)
            q_norm = np.linalg.norm(q_dense)
            if q_norm > 0:
                q_dense = q_dense / q_norm
                sub_embs = self.dense_embeddings[valid_indices]
                sims = np.dot(sub_embs, q_dense.T).flatten()
                for idx, orig_i in enumerate(valid_indices):
                    semantic_scores[orig_i] = float(sims[idx])
            else:
                for orig_i in valid_indices:
                    semantic_scores[orig_i] = 0.0

        scored_results = []

        if mode == "bm25":
            sorted_indices = sorted(valid_indices, key=lambda i: bm25_scores.get(i, 0.0), reverse=True)
            max_s = max(bm25_scores.values()) if bm25_scores else 1.0
            for i in sorted_indices[:top_k]:
                score = bm25_scores.get(i, 0.0)
                norm_score = (score / max_s) if max_s > 0 else 0.0
                chunk_copy = dict(self.chunks[i])
                chunk_copy["relevance_score"] = round(float(norm_score), 4)
                chunk_copy["score_percent"] = min(99, max(30, int(norm_score * 100)))
                scored_results.append(chunk_copy)

        elif mode == "semantic":
            sorted_indices = sorted(valid_indices, key=lambda i: semantic_scores.get(i, 0.0), reverse=True)
            for i in sorted_indices[:top_k]:
                score = semantic_scores.get(i, 0.0)
                norm_score = max(0.0, score)
                chunk_copy = dict(self.chunks[i])
                chunk_copy["relevance_score"] = round(float(norm_score), 4)
                chunk_copy["score_percent"] = min(99, max(30, int(norm_score * 100)))
                scored_results.append(chunk_copy)

        else: # Hybrid RRF (Reciprocal Rank Fusion)
            # Rank BM25
            bm25_ranked = sorted(valid_indices, key=lambda i: bm25_scores.get(i, 0.0), reverse=True)
            bm25_rank_map = {idx: rank + 1 for rank, idx in enumerate(bm25_ranked)}

            # Rank Semantic
            sem_ranked = sorted(valid_indices, key=lambda i: semantic_scores.get(i, 0.0), reverse=True)
            sem_rank_map = {idx: rank + 1 for rank, idx in enumerate(sem_ranked)}

            k_rrf = 60.0
            rrf_scores = {}
            for i in valid_indices:
                r_bm25 = bm25_rank_map.get(i, len(valid_indices))
                r_sem = sem_rank_map.get(i, len(valid_indices))
                rrf_scores[i] = (1.0 / (k_rrf + r_bm25)) + (1.0 / (k_rrf + r_sem))

            sorted_indices = sorted(valid_indices, key=lambda i: rrf_scores.get(i, 0.0), reverse=True)
            max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0

            for i in sorted_indices[:top_k]:
                score = rrf_scores.get(i, 0.0)
                norm_score = score / max_rrf if max_rrf > 0 else 0.0
                chunk_copy = dict(self.chunks[i])
                chunk_copy["relevance_score"] = round(float(norm_score), 4)
                chunk_copy["score_percent"] = min(99, max(45, int(norm_score * 100)))
                chunk_copy["bm25_score"] = round(bm25_scores.get(i, 0.0), 3)
                chunk_copy["semantic_score"] = round(semantic_scores.get(i, 0.0), 3)
                scored_results.append(chunk_copy)

        return scored_results

    def generate_prompt(self, query: str, context_chunks: List[Dict[str, Any]], conversation_history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """Constructs prompt messages for Groq LLM with context grounding."""
        context_blocks = []
        for idx, chunk in enumerate(context_chunks, 1):
            source_citation = f"Source [{idx}]: {chunk['report_title']} (Page {chunk['page']})"
            block = f"{source_citation}\n{chunk['text'].strip()}"
            context_blocks.append(block)

        context_text = "\n\n---\n\n".join(context_blocks)

        system_instruction = (
            "You are the official Infosys Enterprise AI Assistant, an authoritative, intelligent, and highly articulate "
            "knowledge assistant powered by Groq LPU and grounded in the Infosys Integrated Annual Reports (FY 2024-25 and FY 2025-26).\n\n"
            "Core Guidelines:\n"
            "1. Grounding & Accuracy: Answer the user's inquiry accurately using the provided Annual Report excerpts. "
            "Cite exact numbers, metrics, initiatives, leadership remarks, and strategic pillars whenever available.\n"
            "2. Citation Format: Explicitly reference your sources using format like [Report Title, Page X] or [Source 1, Page X] "
            "so the reader can trace facts directly back to the report.\n"
            "3. Formatting & Clarity: Structure your responses with clear markdown: use bullet points, bold key highlights, "
            "create markdown comparison tables when discussing financial figures or multi-year trends, and provide executive summaries.\n"
            "4. Tone: Professional, forward-looking, analytical, and objective.\n"
            "5. Coverage: Highlight Infosys key platforms such as Infosys Topaz (AI & generative AI), Infosys Cobalt (Cloud), "
            "Finacle (Core banking), EdgeVerve, ESG milestones, global client partnerships (Hertz, Liberty Global, Mondelēz, etc.), "
            "and executive vision from Nandan Nilekani and Salil Parekh when relevant.\n"
            "6. If the context does not contain the exact answer, transparently state what is confirmed in the reports "
            "and provide the most relevant grounded information available."
        )

        messages = [{"role": "system", "content": system_instruction}]

        # Include prior conversation history if provided (last 4 turns)
        if conversation_history:
            for turn in conversation_history[-4:]:
                if turn.get("role") in ("user", "assistant") and turn.get("content"):
                    messages.append({"role": turn["role"], "content": turn["content"]})

        user_content = (
            f"Here are relevant excerpts retrieved from the Infosys Annual Reports:\n\n"
            f"<ANNUAL_REPORT_CONTEXT>\n{context_text}\n</ANNUAL_REPORT_CONTEXT>\n\n"
            f"User Question: {query}\n\n"
            f"Please provide a comprehensive, well-structured, and cited answer based on the reports."
        )

        messages.append({"role": "user", "content": user_content})
        return messages

    def answer_query(
        self,
        query: str,
        model: Optional[str] = None,
        top_k: int = 5,
        mode: str = "hybrid",
        fiscal_year: Optional[str] = "all",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """Answers a query using RAG with non-streaming Groq completion."""
        model_name = model or config.DEFAULT_MODEL
        chunks = self.search(query, top_k=top_k, mode=mode, fiscal_year=fiscal_year)
        messages = self.generate_prompt(query, chunks, conversation_history)

        try:
            response = self.groq_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=1500
            )
            answer_content = response.choices[0].message.content or ""
            reasoning_content = getattr(response.choices[0].message, "reasoning", None)

            return {
                "answer": answer_content,
                "reasoning": reasoning_content,
                "model": model_name,
                "sources": chunks,
                "query": query,
                "tokens": {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0)
                }
            }
        except Exception as e:
            return {
                "answer": f"Error generating answer via Groq API: {str(e)}",
                "sources": chunks,
                "model": model_name,
                "query": query,
                "error": str(e)
            }

    def answer_query_stream(
        self,
        query: str,
        model: Optional[str] = None,
        top_k: int = 5,
        mode: str = "hybrid",
        fiscal_year: Optional[str] = "all",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.2
    ) -> Generator[Dict[str, Any], None, None]:
        """Generator yielding SSE-compatible chunks for real-time streaming."""
        model_name = model or config.DEFAULT_MODEL
        chunks = self.search(query, top_k=top_k, mode=mode, fiscal_year=fiscal_year)
        
        # First yield the sources metadata so the frontend can immediately show citations
        yield {
            "type": "sources",
            "sources": chunks,
            "model": model_name,
            "query": query
        }

        messages = self.generate_prompt(query, chunks, conversation_history)

        try:
            stream = self.groq_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=2048,
                stream=True
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                token = delta.content or ""
                reasoning = getattr(delta, "reasoning", "")
                if token or reasoning:
                    yield {
                        "type": "content",
                        "text": token,
                        "reasoning": reasoning
                    }

            yield {"type": "done"}
        except Exception as e:
            yield {
                "type": "error",
                "error": str(e)
            }

if __name__ == "__main__":
    import sys
    print("=" * 64)
    print("  Infosys Enterprise RAG AI Assistant (Terminal Mode)")
    print("  Powered by Groq LPU & Official Annual Reports")
    print("=" * 64)
    
    engine = RAGEngine()
    engine.load_or_build_index()
    
    stats = engine.indexing_stats
    print(f"\n[Ready] Grounded in {stats['total_pages']} pages ({stats['total_chunks']} chunks).")
    print("Type your question below (or type 'exit' / 'quit' to quit):\n")
    
    while True:
        try:
            query = input("\n\033[1;36mInfosys Query > \033[0m").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("\nGoodbye!")
                break
                
            print(f"\n\033[1;32m[Searching & Generating with {config.DEFAULT_MODEL} via Groq LPU...]\033[0m\n")
            
            stream_gen = engine.answer_query_stream(query)
            sources = []
            
            for item in stream_gen:
                if item.get("type") == "sources":
                    sources = item.get("sources", [])
                elif item.get("type") == "content":
                    sys.stdout.write(item.get("text", ""))
                    sys.stdout.flush()
                elif item.get("type") == "error":
                    print(f"\n[Error]: {item.get('error')}")
                    
            print("\n\n" + "-" * 50)
            print("\033[1;33m[Grounded Sources Cited]:\033[0m")
            for idx, s in enumerate(sources, 1):
                pct = s.get("score_percent", 90)
                print(f"  [{idx}] {s['report_title']} - Page {s['page']} (Relevance: {pct}%)")
            print("-" * 50)
            
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

