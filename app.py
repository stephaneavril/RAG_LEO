from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
import os, sqlite3
import numpy as np

# ===== Config =====
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))  # dimensión del embedding
DB_PATH = os.getenv("DB_PATH", "medical_rag.db")
ALLOW_INGEST = os.getenv("ALLOW_INGEST", "true").lower() == "true"  # protege /ingest en prod si lo deseas

app = Flask(__name__, static_folder="static", static_url_path="/static")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_schema():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS docs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT, source_url TEXT, published_at TEXT, added_at TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS chunks(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id INTEGER, chunk_index INTEGER, content TEXT, tokens INTEGER
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS embeddings(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chunk_id INTEGER, embedding BLOB
    )""")
    conn.commit(); conn.close()

ensure_schema()

# ======== Embeddings (placeholder) ========
# En producción reemplaza por API real de embeddings (p. ej., OpenAI) y guarda np.float32.tobytes()
def embed_text(text: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.normal(size=EMBED_DIM).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-9)
    return v

def cosine_sim(a, b): 
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))

def search_similar(qvec: np.ndarray, topk=5):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
      SELECT e.id, e.chunk_id, e.embedding, c.content, c.doc_id
      FROM embeddings e 
      JOIN chunks c ON c.id = e.chunk_id
    """)
    rows = cur.fetchall()
    scored = []
    for r in rows:
        emb = np.frombuffer(r["embedding"], dtype=np.float32)
        scored.append((cosine_sim(qvec, emb), r))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, r in scored[:topk]:
        d = get_db().execute("SELECT title, source_url, published_at FROM docs WHERE id=?",(r["doc_id"],)).fetchone()
        results.append({
            "score": score,
            "content": r["content"],
            "title": d["title"] if d else None,
            "source_url": d["source_url"] if d else None,
            "published_at": d["published_at"] if d else None
        })
    conn.close()
    return results

# ====== LLM answer (placeholder) ======
SYSTEM_PROMPT = """Eres un asistente médico informativo.
- No diagnostiques ni prescribas.
- Ofrece información general basada en el contexto proporcionado.
- Menciona límites y recomienda consultar a un profesional.
- Incluye una sección 'Fuentes' con título/URL/fecha cuando existan.
- Si falta evidencia, dilo explícitamente."""

def llm_answer(prompt: str) -> str:
    # Placeholder: en producción llama a tu proveedor LLM (OpenAI, etc.)
    # Aquí solo recortamos el texto para demostración.
    return "Respuesta (demo): " + prompt[:600]

# ====== Routes ======
@app.route("/")
def index():
    # Sirve el chat del avatar sin labios
    return send_from_directory(app.static_folder, "avatar_chat.html")

@app.route("/health")
def health():
    return jsonify({"ok": True, "ts": datetime.utcnow().isoformat()})

@app.route("/ingest", methods=["POST"])
def ingest():
    if not ALLOW_INGEST:
        return jsonify({"ok": False, "error": "Ingesta deshabilitada en producción"}), 403
    data = request.get_json(force=True)
    title = data.get("title") or "Fuente interna"
    source_url = data.get("source_url")
    published_at = data.get("published_at")
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"ok": False, "error": "Falta 'text'"}), 400

    # chunking simple
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    conn = get_db(); cur = conn.cursor()
    cur.execute("INSERT INTO docs(title, source_url, published_at, added_at) VALUES(?,?,?,?)",
                (title, source_url, published_at, datetime.utcnow().isoformat()))
    doc_id = cur.lastrowid

    for idx, ch in enumerate(chunks):
        cur.execute("INSERT INTO chunks(doc_id, chunk_index, content, tokens) VALUES(?,?,?,?)",
                    (doc_id, idx, ch, len(ch.split())))
        chunk_id = cur.lastrowid
        v = embed_text(ch).tobytes()
        cur.execute("INSERT INTO embeddings(chunk_id, embedding) VALUES(?,?)", (chunk_id, v))
    conn.commit(); conn.close()
    return jsonify({"ok": True, "doc_id": doc_id, "chunks": len(chunks)})

@app.route("/ask", methods=["POST"])
def ask():
    j = request.get_json(force=True)
    q = (j.get("question") or "").strip()
    if not q:
        return jsonify({"answer": "Haz tu pregunta médica con claridad.", "sources": []})
    qvec = embed_text(q)
    ctx = search_similar(qvec, topk=int(os.getenv("TOPK", "5")))

    # Construcción del prompt con contexto
    context_blocks = []
    for c in ctx:
        meta = f"[{c.get('title') or 'Fuente interna'} | {c.get('published_at') or 's/f'} | {c.get('source_url') or 's/u'}]"
        context_blocks.append(meta + "\n" + (c.get("content") or ""))
    context_text = "\n\n---\n\n".join(context_blocks)

    user_prompt = f"""[PREGUNTA]
{q}

[CONTEXTOS RELEVANTES]
{context_text}

[INSTRUCCIONES]
- Contesta en español claro para público general.
- Indica límites (no es diagnóstico).
- Devuelve una sección 'Fuentes' enumerando las 3 más relevantes.
"""

    raw = llm_answer(SYSTEM_PROMPT + "\n\n" + user_prompt)

    sources = []
    for c in ctx[:3]:
        sources.append({
            "title": c.get("title") or "Fuente interna",
            "url": c.get("source_url"),
            "date": c.get("published_at")
        })

    return jsonify({
        "answer": raw,
        "sources": sources,
        "timestamp": datetime.utcnow().isoformat()
    })

if __name__ == "__main__":
    # Para desarrollo local
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)
