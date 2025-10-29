from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
import os, sqlite3
import numpy as np
from PyPDF2 import PdfReader
from openai import OpenAI

# ===== Config =====
EMBED_DIM = 1536
DB_PATH = os.getenv("DB_PATH", "medical_rag.db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # ← Coloca tu clave aquí en Render
ALLOW_INGEST = os.getenv("ALLOW_INGEST", "true").lower() == "true"

client = OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__, static_folder="static", static_url_path="/static")

# ======== Base de datos ========
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

# ======== Embeddings reales ========
def embed_text(text: str) -> np.ndarray:
    """Usa OpenAI embeddings reales."""
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text[:8000]
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

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
        d = get_db().execute(
            "SELECT title, source_url, published_at FROM docs WHERE id=?",
            (r["doc_id"],)
        ).fetchone()
        results.append({
            "score": score,
            "content": r["content"],
            "title": d["title"] if d else None,
            "source_url": d["source_url"] if d else None,
            "published_at": d["published_at"] if d else None
        })
    conn.close()
    return results

# ======== IA médica (GPT-4o) ========
def llm_answer(question: str, context: str):
    """Genera respuesta médica basada en contexto científico."""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": (
                "Eres un asistente médico informativo y profesional. "
                "Usa exclusivamente la evidencia proporcionada, "
                "no inventes datos. Sé claro y prudente. "
                "Termina con un aviso: 'Esta información no sustituye una consulta médica.'"
            )},
            {"role": "user", "content": f"Pregunta: {question}\n\nEvidencia encontrada:\n{context}"}
        ]
    )
    return completion.choices[0].message.content.strip()

# ====== Rutas ======
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "avatar_chat.html")

@app.route("/health")
def health():
    return jsonify({"ok": True, "ts": datetime.utcnow().isoformat()})

# ====== Subida de PDF ======
@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No se envió archivo PDF"}), 400
    file = request.files["file"]
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Solo se aceptan archivos PDF"}), 400

    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    if not text.strip():
        return jsonify({"error": "No se pudo extraer texto del PDF"}), 400

    title = os.path.splitext(file.filename)[0]
    conn = get_db(); cur = conn.cursor()
    cur.execute("INSERT INTO docs(title, added_at) VALUES(?,?)", (title, datetime.utcnow().isoformat()))
    doc_id = cur.lastrowid

    CHUNK_SIZE = 1000
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    for idx, ch in enumerate(chunks):
        cur.execute("INSERT INTO chunks(doc_id, chunk_index, content, tokens) VALUES(?,?,?,?)",
                    (doc_id, idx, ch, len(ch.split())))
        chunk_id = cur.lastrowid
        v = embed_text(ch).tobytes()
        cur.execute("INSERT INTO embeddings(chunk_id, embedding) VALUES(?,?)", (chunk_id, v))
    conn.commit(); conn.close()
    return jsonify({"ok": True, "title": title, "chunks": len(chunks)})

# ====== Pregunta médica ======
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"answer": "Por favor formula una pregunta médica.", "sources": []})

    qvec = embed_text(question)
    context_data = search_similar(qvec, topk=5)
    context_text = "\n\n---\n\n".join([
        f"{c['title'] or 'Documento interno'} ({c['published_at'] or 's/f'}):\n{c['content']}"
        for c in context_data
    ])

    answer = llm_answer(question, context_text)

    sources = [
        {"title": c["title"] or "Fuente interna", "url": c["source_url"], "date": c["published_at"]}
        for c in context_data[:3]
    ]

    return jsonify({"answer": answer, "sources": sources})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)
