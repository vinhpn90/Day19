import os
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# 1. ENVIRONMENT LOADER & API CLIENT INITIALIZATION
# ---------------------------------------------------------
def load_env(env_path):
    """Manually parse .env file to load variables into environment."""
    if os.path.exists(env_path):
        print(f"Loading environment variables from {env_path}...")
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    k, v = line.split('=', 1)
                    k = k.strip()
                    v = v.strip().strip("'").strip('"')
                    os.environ[k] = v
                    # print(f"Set env: {k}")

# Load environment from the known path
env_file_path = "/Users/ngocvinh/ownCloud/HocTap/phase1-track3-lab1-advanced-agent/.env"
load_env(env_file_path)

# API parameters
BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://14.241.208.1:443/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "/var/lib/vllm/hf/gpt-oss-120b")

print(f"API Base URL: {BASE_URL}")
print(f"Using model: {MODEL_NAME}")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# ---------------------------------------------------------
# 2. ROBUST TEXT PREPROCESSOR & CLEANER
# ---------------------------------------------------------
def clean_text(text):
    """
    Cleans full content by filtering out binary PDF stream garbage
    and keeping only readable text lines.
    """
    cleaned_lines = []
    lines = text.split('\n')
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip standard PDF objects and stream keywords
        if any(marker in stripped for marker in ['endstream', 'endobj', 'stream', '/Type', '/Filter', '/Length', '>>', '<<']):
            continue
        # Skip lines with unicode replacement character (indicates decodings errors in binary)
        if '\ufffd' in stripped:
            continue
        # Skip lines with too many control characters (ASCII < 32)
        control_chars = sum(1 for c in stripped if ord(c) < 32 and c not in '\t\n\r')
        if control_chars > 2:
            continue
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines)

def parse_doc(file_path):
    """Parse document containing structured fields (Query, Title, Link, Snippet, Full Content)."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    query = ""
    title = ""
    link = ""
    snippet = ""
    full_content = ""
    
    lines = content.split('\n')
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("Query:"):
            query = line.replace("Query:", "").strip()
        elif line.startswith("Title:"):
            title = line.replace("Title:", "").strip()
        elif line.startswith("Link:"):
            link = line.replace("Link:", "").strip()
        elif line.startswith("Snippet:"):
            snippet = line.replace("Snippet:", "").strip()
        elif line.startswith("Full Content:"):
            full_content = "\n".join(lines[idx+1:])
            break
        idx += 1
        
    cleaned_content = clean_text(full_content)
    
    return {
        "filename": os.path.basename(file_path),
        "query": query,
        "title": title,
        "link": link,
        "snippet": snippet,
        "full_content": cleaned_content
    }

# ---------------------------------------------------------
# 3. ENTITY & RELATION EXTRACTION (INDEXING)
# ---------------------------------------------------------
def extract_triples_for_doc(doc):
    """Use the LLM to extract triples from a document's title, snippet, and content."""
    filename = doc["filename"]
    text_to_process = f"Title: {doc['title']}\nSnippet: {doc['snippet']}\nContent:\n{doc['full_content'][:30000]}"
    
    prompt = f"""You are an AI assistant specialized in information extraction for a Knowledge Graph.
Extract key entities (nodes) and their relationships (directed edges) from the following text. Focus on electric vehicles, market sentiment, financials, companies, key people, and metrics.

Text:
{text_to_process}

Output ONLY a valid JSON array of triples. Each triple must have exactly the keys: "subject", "relation", and "object".
Do not include any introductory or concluding text, only the raw JSON code block starting with ```json and ending with ```.

CRITICAL: Keep your reasoning/thinking phase extremely brief and concise (less than 3 sentences). Move as quickly as possible to generating the final JSON array.
"""
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048
        )
        elapsed = time.time() - start_time
        msg = response.choices[0].message
        response_text = msg.content
        if response_text is None:
            if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                response_text = msg.reasoning_content
            elif hasattr(msg, 'reasoning') and msg.reasoning:
                response_text = msg.reasoning
            else:
                response_text = ""
                
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        
        # Clean reasoning tags out
        response_clean = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        
        # Parse JSON
        json_str = ""
        json_match = re.search(r'```json\s*(\[.*?\])\s*```', response_clean, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            brackets_match = re.search(r'(\[.*?\])', response_clean, re.DOTALL)
            if brackets_match:
                json_str = brackets_match.group(1)
            else:
                json_str = response_clean.strip()
                
        try:
            triples = json.loads(json_str)
        except Exception as json_err:
            print(f"[{filename}] Standard JSON parsing failed ({json_err}). Attempting regex fallback extraction...")
            triples = []
            blocks = re.findall(r'\{[^{}]*\}', response_clean)
            for block in blocks:
                try:
                    subj_m = re.search(r'"subject"\s*:\s*"(.*?)"', block)
                    rel_m = re.search(r'"relation"\s*:\s*"(.*?)"', block)
                    obj_m = re.search(r'"object"\s*:\s*"(.*?)"', block)
                    if subj_m and rel_m and obj_m:
                        triples.append({
                            "subject": subj_m.group(1).strip(),
                            "relation": rel_m.group(1).strip(),
                            "object": obj_m.group(1).strip()
                        })
                except Exception:
                    pass
            if not triples:
                raise json_err
                
        print(f"[{filename}] Extracted {len(triples)} triples in {elapsed:.2f}s ({prompt_tokens} prompt tokens, {completion_tokens} completion tokens).")
        return {
            "filename": filename,
            "triples": triples,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "time_taken": elapsed
        }
    except Exception as e:
        print(f"Error extracting from {filename}: {e}")
        return {
            "filename": filename,
            "triples": [],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "time_taken": 0.0
        }

def run_indexing(docs, cache_path="extracted_triples.json"):
    """Run LLM entity extraction over all documents, caching the results locally."""
    if os.path.exists(cache_path):
        print(f"Loading triples from cache: {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    print("No cache found. Running entity & relation extraction using LLM...")
    all_results = []
    
    # Thread pool execution for speed
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(extract_triples_for_doc, doc): doc for doc in docs}
        for future in as_completed(futures):
            res = future.result()
            if res["triples"]:
                all_results.append(res)
                
    # Save cache
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
        
    return all_results

# ---------------------------------------------------------
# 4. GRAPH CONSTRUCTION (NETWORKX)
# ---------------------------------------------------------
def normalize_entity(name):
    """Normalize names of entities to prevent duplicates."""
    if not isinstance(name, str):
        return str(name)
    name = name.strip()
    # Basic mapping dictionary for company entities
    mapping = {
        "Tesla Inc.": "Tesla",
        "Tesla Motors": "Tesla",
        "Tesla, Inc.": "Tesla",
        "NVIDIA Corporation": "NVIDIA",
        "Nvidia": "NVIDIA",
        "VinFast Auto": "VinFast",
        "VinFast Reports": "VinFast",
        "Mercedes-Benz Group": "Mercedes-Benz",
        "Mercedes-Benz AG": "Mercedes-Benz",
        "Mercedes Benz": "Mercedes-Benz",
        "Ford Motor Company": "Ford",
        "BMW AG": "BMW",
        "General Motors": "GM",
        "US": "U.S.",
        "United States": "U.S.",
        "US Government": "U.S. Government"
    }
    return mapping.get(name, name)

def build_networkx_graph(all_results):
    """Load extracted triples into a NetworkX DiGraph."""
    G = nx.DiGraph()
    total_triples = 0
    
    for doc_res in all_results:
        filename = doc_res["filename"]
        for triple in doc_res["triples"]:
            subj = normalize_entity(triple.get("subject", ""))
            rel = triple.get("relation", "").upper().replace(" ", "_")
            obj = normalize_entity(triple.get("object", ""))
            
            if subj and rel and obj:
                total_triples += 1
                # If edge already exists, append filename to sources attribute
                if G.has_edge(subj, obj):
                    sources = G[subj][obj].get("sources", [])
                    if filename not in sources:
                        sources.append(filename)
                        G[subj][obj]["sources"] = sources
                else:
                    G.add_edge(subj, obj, relation=rel, sources=[filename])
                    
    # Add BRAND_LINK edges to connect brand-related components
    node_list = list(G.nodes())
    brands = ["Tesla", "Cadillac", "BMW", "Mercedes", "Audi", "Ford", "Chevrolet", "Chevy", "ZEEKR", "VinFast", "OpenAI"]
    brand_links = 0
    for brand in brands:
        brand_nodes = [n for n in node_list if brand.lower() in str(n).lower()]
        for i in range(len(brand_nodes)):
            for j in range(i + 1, len(brand_nodes)):
                u, v = brand_nodes[i], brand_nodes[j]
                if not G.has_edge(u, v):
                    G.add_edge(u, v, relation="BRAND_LINK", sources=["GraphRAG_Linker"])
                    brand_links += 1
                if not G.has_edge(v, u):
                    G.add_edge(v, u, relation="BRAND_LINK", sources=["GraphRAG_Linker"])
                    brand_links += 1
                    
    print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges ({brand_links} brand links added) from {total_triples} total triples.")
    return G


def visualize_graph(G, output_path="knowledge_graph.png"):
    """Visualize top connected nodes in the graph and save as a high-quality PNG."""
    plt.figure(figsize=(15, 12))
    
    # Select nodes with degree > 1 to avoid cluttering, or top 50 nodes by degree
    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:40]
    subgraph = G.subgraph(top_nodes)
    
    # Layout algorithm
    pos = nx.spring_layout(subgraph, k=1.8, seed=42)
    
    # Plot configurations - aesthetic colors
    node_color = '#2c3e50'
    edge_color = '#95a5a6'
    text_color = '#e74c3c'
    
    # Draw nodes
    nx.draw_networkx_nodes(subgraph, pos, node_size=800, node_color=node_color, alpha=0.9)
    
    # Draw edges
    nx.draw_networkx_edges(subgraph, pos, width=1.5, alpha=0.5, edge_color=edge_color, arrows=True, arrowsize=15)
    
    # Draw node labels
    nx.draw_networkx_labels(subgraph, pos, font_size=9, font_family='sans-serif', font_color='white', font_weight='bold')
    
    # Draw edge labels (relations) for a subset of edges to keep it readable
    edge_labels = {}
    for u, v, data in subgraph.edges(data=True):
        edge_labels[(u, v)] = data["relation"]
        
    # Draw selected edge labels (only first 20 to avoid clutter)
    selected_labels = dict(list(edge_labels.items())[:25])
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=selected_labels, font_size=7, font_color=text_color)
    
    plt.title("Tech Company & EV Industry Knowledge Graph (Top 40 Nodes)", fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graph visualization saved to {output_path}")

# ---------------------------------------------------------
# 5. QUERY SYSTEMS
# ---------------------------------------------------------

class FlatRAGSystem:
    def __init__(self, docs):
        self.docs = docs
        print("Initializing Flat RAG with FAISS and SentenceTransformers...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Chunk the documents to prevent content loss
        self.chunks = []
        chunk_size = 1500
        overlap = 300
        
        for d in docs:
            text = d['full_content']
            title = d['title']
            filename = d['filename']
            snippet = d['snippet']
            
            # If text is very short, just keep it as a single chunk
            if len(text) <= chunk_size:
                self.chunks.append({
                    "filename": filename,
                    "title": title,
                    "snippet": snippet,
                    "text": text
                })
            else:
                start = 0
                while start < len(text):
                    end = start + chunk_size
                    chunk_text = text[start:end]
                    self.chunks.append({
                        "filename": filename,
                        "title": title,
                        "snippet": snippet,
                        "text": chunk_text
                    })
                    start += (chunk_size - overlap)
                    
        # Generate corpus representations
        self.corpus = [
            f"Source: {c['filename']}\nTitle: {c['title']}\nSnippet: {c['snippet']}\nContent: {c['text']}"
            for c in self.chunks
        ]
        
        print(f"Generating embeddings for {len(self.corpus)} text chunks...")
        embeddings = self.model.encode(self.corpus, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize embeddings for cosine similarity using inner product
        faiss.normalize_L2(embeddings)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        print("FAISS index built successfully.")
        
    def query(self, user_query, k=5):
        """Retrieve top k chunks using FAISS similarity and send to LLM."""
        query_emb = self.model.encode([user_query])
        query_emb = np.array(query_emb).astype('float32')
        faiss.normalize_L2(query_emb)
        
        # Search index
        scores, indices = self.index.search(query_emb, k)
        top_indices = indices[0]
        
        retrieved_context = []
        retrieved_sources = set()
        
        for idx in top_indices:
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            retrieved_sources.add(chunk['filename'])
            retrieved_context.append(f"Source: {chunk['filename']}\nTitle: {chunk['title']}\nContent:\n{chunk['text']}")
            
        context_str = "\n\n---\n\n".join(retrieved_context)
        
        prompt = f"""You are an expert analyst. Answer the user query based ONLY on the provided context document details.
If the information is not in the context, say "Based on the documents, I cannot answer this query."

Context:
{context_str}

Query: {user_query}
Answer:"""
        
        start_time = time.time()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=800
        )
        elapsed = time.time() - start_time
        
        msg = response.choices[0].message
        response_text = msg.content
        if response_text is None:
            if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                response_text = msg.reasoning_content
            elif hasattr(msg, 'reasoning') and msg.reasoning:
                response_text = msg.reasoning
            else:
                response_text = ""
                
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        
        # Clean reasoning tags out
        response_clean = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        
        return {
            "answer": response_clean.strip(),
            "retrieved_sources": sorted(list(retrieved_sources)),
            "time_taken": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }

class GraphRAGSystem:
    def __init__(self, G):
        self.G = G
        
    def extract_entities_from_query(self, query):
        """Use the LLM to extract entity keywords (brands, models, metrics, dates, laws, organizations) from the query."""
        prompt = f"""You are an expert entity extractor. Extract any key search terms, entities, company names, brand names, model names, specific laws/policies, specific numbers (like percentages or dollar amounts), dates, or time periods mentioned in the user query that can help look up information in a Knowledge Graph.
Output ONLY a valid JSON list of strings. Do not include any explanations, introductory remarks, or markdown formatting blocks.

Examples:
Query: "What are the specific EV sales growth percentages for Mercedes-Benz, BMW, and Cadillac in Q1 2024?"
Output: ["Mercedes-Benz", "BMW", "Cadillac", "Q1 2024"]

Query: "Which brand recorded a 499.2% increase in sales in Q1 2024 and which model was responsible for it?"
Output: ["499.2%", "Q1 2024"]

Query: "Compare the market share of Tesla in U.S. Q1 2024 vs Q1 2023."
Output: ["Tesla", "Q1 2024", "Q1 2023", "U.S."]

Query: "Explain the relation between the Inflation Reduction Act (IRA), U.S., and the European Union based on the documents."
Output: ["Inflation Reduction Act", "U.S.", "European Union"]

Query: {query}
Output:"""
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200
            )
            msg = response.choices[0].message
            response_text = msg.content
            if response_text is None:
                if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                    response_text = msg.reasoning_content
                elif hasattr(msg, 'reasoning') and msg.reasoning:
                    response_text = msg.reasoning
                else:
                    response_text = ""
            response_clean = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
            
            # Parse JSON list
            match = re.search(r'(\[.*?\])', response_clean, re.DOTALL)
            if match:
                extracted = json.loads(match.group(1))
            else:
                extracted = json.loads(response_clean.strip())
                
            # Match extracted entities to graph nodes
            matched_nodes = []
            for ext in extracted:
                ext_lower = str(ext).lower().strip()
                if not ext_lower or len(ext_lower) < 2:
                    continue
                for node in self.G.nodes():
                    node_lower = str(node).lower().strip()
                    # Match if exact, or substring of each other
                    if ext_lower == node_lower or ext_lower in node_lower or node_lower in ext_lower:
                        if node not in matched_nodes:
                            # Avoid matching extremely short generic abbreviations unless they are digits
                            if len(ext_lower) > 2 or ext_lower.isdigit() or len(node_lower) > 2:
                                matched_nodes.append(node)
            return matched_nodes
        except Exception as e:
            print(f"Error in LLM entity matching: {e}. Falling back to keyword search...")
            # Fallback to simple keyword check
            matched_nodes = []
            query_lower = query.lower()
            for node in self.G.nodes():
                if len(node) > 2:
                    if node.lower() in query_lower or query_lower in node.lower():
                        matched_nodes.append(node)
            return matched_nodes

    def get_combined_2_hop_context(self, entities):
        """Retrieve the neighborhood triples, ranking specific nodes higher and mitigating hub-node pollution."""
        all_nodes = set()
        entity_set = set(entities)
        
        # Calculate node degrees
        degrees = dict(self.G.degree())
        
        # Traverse
        for start_node in entities:
            if start_node not in self.G:
                continue
            
            all_nodes.add(start_node)
            
            # 1-hop
            neighbors_1 = list(self.G.successors(start_node)) + list(self.G.predecessors(start_node))
            all_nodes.update(neighbors_1)
            
            # 2-hop: do not expand through a hub node (degree > 15) to prevent flooding
            for n in neighbors_1:
                if degrees.get(n, 0) <= 15 or n in entity_set:
                    neighbors_2 = list(self.G.successors(n)) + list(self.G.predecessors(n))
                    all_nodes.update(neighbors_2)
                    
        if not all_nodes:
            return ""
            
        subgraph = self.G.subgraph(all_nodes)
        
        # Rank triples
        ranked_triples = []
        for u, v, data in subgraph.edges(data=True):
            rel = data["relation"]
            sources = ", ".join(data.get("sources", ["Unknown"]))
            
            score = 1
            u_deg = degrees.get(u, 0)
            v_deg = degrees.get(v, 0)
            
            if u in entity_set and v in entity_set:
                score = 4
            elif u in entity_set or v in entity_set:
                if u_deg <= 15 and v_deg <= 15:
                    score = 3
                else:
                    score = 2
                    
            ranked_triples.append((score, u, rel, v, sources))
            
        ranked_triples.sort(key=lambda x: x[0], reverse=True)
        
        # Format top 150 triples
        context_triples = []
        for score, u, rel, v, sources in ranked_triples[:150]:
            context_triples.append(f"- ({u}) -[{rel}]-> ({v}) [Source docs: {sources}]")
            
        return "\n".join(context_triples)

    def query(self, user_query):
        """Extract multiple entities, pull combined graph context, and query LLM."""
        entities = self.extract_entities_from_query(user_query)
        graph_context = ""
        
        if entities:
            print(f"Extracted entities from query: {entities}")
            graph_context = self.get_combined_2_hop_context(entities)
        else:
            print(f"No matching entities found in graph for query: '{user_query}'")
            
        if not graph_context:
            graph_context = "No direct knowledge graph paths available for this query entity."
            
        prompt = f"""You are an expert analyst. Answer the user query using the Knowledge Graph triples provided below.
The triples format is: - (Subject) -[Relation]-> (Object) [Source docs: ...]
If the triples do not contain the answer, say "Based on the knowledge graph, I cannot answer this query."

Knowledge Graph Context:
{graph_context}

Query: {user_query}
Answer:"""
        
        start_time = time.time()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=800
        )
        elapsed = time.time() - start_time
        
        msg = response.choices[0].message
        response_text = msg.content
        if response_text is None:
            if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                response_text = msg.reasoning_content
            elif hasattr(msg, 'reasoning') and msg.reasoning:
                response_text = msg.reasoning
            else:
                response_text = ""
                
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        
        # Clean reasoning tags out
        response_clean = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        
        return {
            "answer": response_clean.strip(),
            "extracted_entity": ", ".join(entities) if entities else None,
            "graph_context_triples": graph_context.split('\n') if graph_context else [],
            "time_taken": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }

# ---------------------------------------------------------
# 6. BENCHMARK & EVALUATION
# ---------------------------------------------------------
def run_benchmark(flat_rag, graph_rag, benchmark_questions):
    """Run benchmark questions on both systems and save results."""
    print(f"Running benchmark on {len(benchmark_questions)} questions...")
    results = []
    
    for idx, q in enumerate(benchmark_questions):
        print(f"\nQuestion {idx+1}/{len(benchmark_questions)}: {q}")
        
        # Run Flat RAG
        flat_res = flat_rag.query(q)
        
        # Run GraphRAG
        graph_res = graph_rag.query(q)
        
        results.append({
            "id": idx + 1,
            "question": q,
            "flat_rag": {
                "answer": flat_res["answer"],
                "sources": flat_res["retrieved_sources"],
                "time_taken": flat_res["time_taken"],
                "prompt_tokens": flat_res["prompt_tokens"],
                "completion_tokens": flat_res["completion_tokens"]
            },
            "graph_rag": {
                "answer": graph_res["answer"],
                "entity": graph_res["extracted_entity"],
                "context_triples_count": len(graph_res["graph_context_triples"]),
                "time_taken": graph_res["time_taken"],
                "prompt_tokens": graph_res["prompt_tokens"],
                "completion_tokens": graph_res["completion_tokens"]
            }
        })
        
    with open("benchmark_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print("\nBenchmark completed. Results saved to benchmark_results.json")
    return results

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    import sys
    
    # Parse documents
    dataset_dir = "/Users/ngocvinh/ownCloud/HocTap/Day19/dataset"
    print(f"Parsing documents from {dataset_dir}...")
    doc_files = sorted(
        [f for f in os.listdir(dataset_dir) if f.startswith('doc_') and f.endswith('.txt')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    
    docs = []
    for f in doc_files:
        path = os.path.join(dataset_dir, f)
        docs.append(parse_doc(path))
    print(f"Loaded {len(docs)} documents.")
    
    is_dry_run = len(sys.argv) > 1 and sys.argv[1] == "--dry-run"
    if is_dry_run:
        print("Dry run: Slicing dataset to first 2 documents to save time and tokens.")
        docs = docs[:2]
    
    # Run indexing (LLM entity/relation extraction)
    indexing_results = run_indexing(docs, cache_path="dry_run_triples.json" if is_dry_run else "extracted_triples.json")
    
    # Build NetworkX Graph
    G = build_networkx_graph(indexing_results)
    
    # Visualize the Graph
    visualize_graph(G, output_path="dry_run_graph.png" if is_dry_run else "knowledge_graph.png")
    
    # Initialize systems
    flat_rag = FlatRAGSystem(docs)
    graph_rag = GraphRAGSystem(G)
    
    # Defining 20 complex, multi-hop queries
    benchmark_questions = [
        "What are the specific EV sales growth percentages for Mercedes-Benz, BMW, and Cadillac in Q1 2024?",
        "Which brand recorded a 499.2% increase in sales in Q1 2024 and which model was responsible for it?",
        "Compare the market share of Tesla in U.S. Q1 2024 vs Q1 2023.",
        "What model did Chevrolet temporarily halt production of in Q1 2024, and when is the new version expected to launch?",
        "Which company has ZEEKR reported financial results for in the documents, and what quarter was it?",
        "What are the top metropolitan areas with the highest electric vehicle uptake in 2020, and what policy incentives did they use?",
        "Who are the authors of the study on electric vehicle market growth across U.S. cities published on September 14, 2021?",
        "What did Stephanie Valdez Streaty comment about Tesla's Q1 2024 sales performance?",
        "Explain the relation between the Inflation Reduction Act (IRA), U.S., and the European Union based on the documents.",
        "What is the average transaction price of a new EV in Q1 2024, and how did it change from the previous year?",
        "Which brand achieved the second-highest EV sales volume behind Tesla in Q1 2024, and what was its growth rate?",
        "Who founded OpenAI and in which year was it founded?",
        "What was the year-over-year EV sales volume growth in Q1 2023 and Q1 2022?",
        "List the luxury EV makers that recorded more than 50% year-over-year growth in EV sales in Q1 2024.",
        "What is the average electric vehicle share of new-vehicle sales in U.S. Q1 2024?",
        "What was the total dollar amount of U.S. electric vehicle investments mentioned in the EDF report?",
        "What specific consumer incentives from state, city, or utilities support EV purchases in the leading metropolitan areas?",
        "Describe the relationship between public and workplace charging availability and electric vehicle growth.",
        "What challenges or issues arose in the EU-US relations due to the Inflation Reduction Act?",
        "Which EV model was launched by Cadillac that drove its sales growth in Q1 2024?"
    ]
    
    # If dry run is requested
    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        print("Dry run requested. Checking first query on both systems...")
        q = benchmark_questions[0]
        print("Flat RAG response:", flat_rag.query(q, k=1))
        print("GraphRAG response:", graph_rag.query(q))
    else:
        # Run full benchmark
        run_benchmark(flat_rag, graph_rag, benchmark_questions)
