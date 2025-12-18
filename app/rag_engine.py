import os
import chromadb
import google.generativeai as genai
from firecrawl import FirecrawlApp
from sentence_transformers import SentenceTransformer, CrossEncoder
try:
    from app.scraper_catalog import scrape_catalog
except ImportError:
    from scraper_catalog import scrape_catalog
import json
import re
from dotenv import load_dotenv
from typing import List, Dict
import numpy as np

load_dotenv()

GENAI_KEY = os.getenv("GOOGLE_API_KEY")
FIRECRAWL_KEY = os.getenv("FIRECRAWL_API_KEY")

if GENAI_KEY: genai.configure(api_key=GENAI_KEY)
firecrawl = FirecrawlApp(api_key=FIRECRAWL_KEY) if FIRECRAWL_KEY else None


class RAGEngine:
    def __init__(self):
        scrape_catalog()
        
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="shl_tests_v2")
        
        # UPGRADE 1: Better embedding model (768 dimensions vs 384)
        print("🧠 Loading Embedding Model (all-mpnet-base-v2)...")
        self.embedder = SentenceTransformer('all-mpnet-base-v2')
        
        # UPGRADE 2: Cross-encoder for reranking (significantly improves precision)
        print("🎯 Loading Reranker (ms-marco-MiniLM)...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        if self.collection.count() == 0:
            self._index_data()

    def _create_rich_document(self, item: Dict) -> str:
        """
        UPGRADE 3: Rich text representation with structured fields
        This helps the model understand context better
        """
        parts = []
        
        # Title with emphasis
        if item.get('name'):
            parts.append(f"Assessment: {item['name']}")
        
        # Test types (critical for matching)
        if item.get('test_type'):
            types = item['test_type'] if isinstance(item['test_type'], list) else [item['test_type']]
            parts.append(f"Categories: {', '.join(types)}")
        
        # Description
        if item.get('description'):
            parts.append(f"Description: {item['description']}")
        
        # Skills (key for technical matching)
        if item.get('skills'):
            skills = item['skills'] if isinstance(item['skills'], list) else [item['skills']]
            parts.append(f"Skills tested: {', '.join(skills)}")
        
        # Competencies (key for behavioral matching)
        if item.get('competencies'):
            comps = item['competencies'] if isinstance(item['competencies'], list) else [item['competencies']]
            parts.append(f"Competencies: {', '.join(comps)}")
        
        return "\n".join(parts)

    def _index_data(self):
        print("📚 Indexing data with rich representations...")
        with open("data/assessments.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        docs = []
        metadatas = []
        
        for d in data:
            # Create rich document
            text = self._create_rich_document(d)
            docs.append(text)

            # Sanitize metadata for ChromaDB (only accepts str, int, float, bool)
            meta_obj = d.copy()
            for key, value in meta_obj.items():
                if value is None:
                    # ChromaDB doesn't accept None values
                    meta_obj[key] = ""
                elif isinstance(value, list):
                    # Convert lists to comma-separated strings
                    meta_obj[key] = ", ".join(str(v) for v in value if v is not None)
            metadatas.append(meta_obj)

        ids = [str(i) for i in range(len(data))]
        embeddings = self.embedder.encode(docs, show_progress_bar=True, batch_size=32).tolist()
        
        self.collection.add(
            documents=docs,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ Indexed {len(docs)} assessments")

    def _extract_structured_query(self, text: str) -> Dict:
        """
        UPGRADE 4: Better query understanding with Gemini
        Extract structured requirements instead of vague enhancement
        """
        if not GENAI_KEY:
            return {"full_text": text, "technical_skills": [], "soft_skills": []}
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""Analyze this job requirement and extract structured information.

Job Requirement:
{text[:2000]}

Return a JSON object with:
1. "technical_skills": List of specific technical skills (e.g., ["Python", "SQL", "React"])
2. "soft_skills": List of behavioral traits/competencies (e.g., ["Leadership", "Communication"])
3. "job_level": Entry/Mid/Senior level
4. "key_requirements": 2-3 most important requirements as short phrases

Return ONLY valid JSON, no markdown formatting."""

        try:
            response = model.generate_content(prompt).text.strip()
            # Remove markdown code blocks if present
            response = re.sub(r'```json\s*|\s*```', '', response)
            parsed = json.loads(response)
            parsed["full_text"] = text
            return parsed
        except Exception as e:
            print(f"⚠️ Gemini extraction failed: {e}")
            return {"full_text": text, "technical_skills": [], "soft_skills": []}

    def _build_hybrid_query(self, structured: Dict) -> str:
        """
        UPGRADE 5: Build optimized query from structured data
        """
        parts = []
        
        if structured.get("technical_skills"):
            parts.append("Technical skills: " + ", ".join(structured["technical_skills"]))
        
        if structured.get("soft_skills"):
            parts.append("Behavioral competencies: " + ", ".join(structured["soft_skills"]))
        
        if structured.get("key_requirements"):
            parts.append("Requirements: " + ", ".join(structured["key_requirements"]))
        
        # Always include original text for context
        parts.append(structured.get("full_text", ""))
        
        return " | ".join(parts)

    def _keyword_boost(self, candidates: List[Dict], structured_query: Dict) -> List[Dict]:
        """
        UPGRADE 6: Keyword-based boosting for exact matches
        This catches cases where embeddings miss obvious keywords
        """
        all_keywords = set()
        
        # Collect all keywords to search for
        for skill in structured_query.get("technical_skills", []):
            all_keywords.add(skill.lower())
        for skill in structured_query.get("soft_skills", []):
            all_keywords.add(skill.lower())
        
        # Score each candidate based on keyword matches
        for cand in candidates:
            match_score = 0
            doc_text = self._create_rich_document(cand).lower()
            
            for keyword in all_keywords:
                if keyword in doc_text:
                    match_score += 1
            
            cand['_keyword_score'] = match_score
        
        # Sort by keyword matches (high to low)
        return sorted(candidates, key=lambda x: x.get('_keyword_score', 0), reverse=True)

    def _rerank_with_cross_encoder(self, query: str, candidates: List[Dict], top_k: int = 20) -> List[Dict]:
        """
        UPGRADE 7: Use cross-encoder to rerank top candidates
        This is much more accurate than pure embedding similarity
        """
        if not candidates:
            return []
        
        # Prepare pairs for reranking
        pairs = []
        for cand in candidates:
            doc_text = self._create_rich_document(cand)
            pairs.append([query, doc_text])
        
        # Get relevance scores
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        
        # Attach scores and sort
        for i, cand in enumerate(candidates):
            cand['_rerank_score'] = float(scores[i])
        
        reranked = sorted(candidates, key=lambda x: x['_rerank_score'], reverse=True)
        return reranked[:top_k]

    def process_query(self, user_input: str):
        """
        Main query processing pipeline with all upgrades
        """
        search_text = user_input
        
        # 1. URL Handling
        if user_input.startswith("http"):
            try:
                if firecrawl:
                    res = firecrawl.scrape_url(user_input, params={'formats': ['markdown']})
                    search_text = res['markdown'][:3000]
            except Exception as e:
                print(f"⚠️ Firecrawl Error: {e}")

        # 2. Extract structured information
        structured_query = self._extract_structured_query(search_text)
        
        # 3. Build optimized query
        enhanced_query = self._build_hybrid_query(structured_query)
        
        # 4. Vector Search (cast wider net)
        query_vec = self.embedder.encode([enhanced_query]).tolist()
        results = self.collection.query(query_embeddings=query_vec, n_results=100)
        
        # 5. Deduplicate and collect candidates
        raw_candidates = []
        seen_urls = set()
        
        if results['metadatas']:
            for meta in results['metadatas'][0]:
                if meta['url'] not in seen_urls:
                    # Convert strings back to lists
                    for key in ['test_type', 'skills', 'competencies']:
                        if key in meta and isinstance(meta[key], str):
                            meta[key] = [x.strip() for x in meta[key].split(",")]
                    
                    raw_candidates.append(meta)
                    seen_urls.add(meta['url'])
        
        # 6. Keyword boosting
        boosted = self._keyword_boost(raw_candidates, structured_query)
        
        # 7. Take top 50 for reranking
        top_candidates = boosted[:50]
        
        # 8. Cross-encoder reranking
        reranked = self._rerank_with_cross_encoder(search_text, top_candidates, top_k=30)
        
        # 9. Final balancing
        final = self._balance_results(reranked, target=10)
        
        # Clean up temporary scoring fields
        for item in final:
            item.pop('_keyword_score', None)
            item.pop('_rerank_score', None)
        
        return final

    def _balance_results(self, candidates, target=10):
        """
        Balance between Knowledge/Skills and Personality/Behavior tests
        """
        def check_type(cand, type_name):
            t = cand.get('test_type', [])
            if isinstance(t, list): return type_name in t
            return type_name in str(t)

        k_tests = [c for c in candidates if check_type(c, "Knowledge & Skills")]
        p_tests = [c for c in candidates if check_type(c, "Personality & Behavior")]
        others = [c for c in candidates if c not in k_tests and c not in p_tests]

        balanced = []
        max_len = max(len(k_tests), len(p_tests), len(others))
        
        for i in range(max_len):
            if i < len(k_tests): balanced.append(k_tests[i])
            if i < len(p_tests): balanced.append(p_tests[i])
            if i < len(others): balanced.append(others[i])
            if len(balanced) >= target: break
            
        return balanced[:target]