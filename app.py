from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
import os
import sqlite3
import numpy as np
from PyPDF2 import PdfReader

# =========================
# Configuración
# =========================
DB_PATH = os.getenv("DB_PATH", "medical_rag.db")
ALLOW_INGEST = os.getenv("ALLOW_INGEST", "true").lower() == "true"

# Modelos por defecto:
# - text-embedding-3-small (dim 1536) es más barato y suficiente para RAG
# - gpt-4o-mini rápido/costo-eficiente para síntesis
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Otros parámetros
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
TOPK = int(os.getenv("TOPK", "5"))

app = Flask(__name__, static_folder="static", static_url_path="/static")


# =========================
# DB helpers
# =========================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_schema():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS docs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            source_url TEXT,
            published_at TEXT,
            added_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER,
            chunk_index INTEGER,
            content TEXT,
            tokens INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id INTEGER,
            embedding BLOB
        )
    """)
    conn.commit()
    conn.close()

ensure_schema()


# =========================
# OpenAI (cliente "perezoso")
# =========================
def get_openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        # Respuesta clara si falta clave
        raise RuntimeError(
            "Falta OPENAI_API_KEY. Configúrala en Render > Environment."
        )
    from openai import OpenAI  # import diferido para evitar fallos al importar app
    return OpenAI(api_key=key)


# =========================
# Embeddings + búsqueda
# =========================
def embed_text(text: str) -> np.ndarray:
    """
    Genera embedding real con OpenAI.
    Normaliza el vector para una similitud de coseno estable.
    """
    client = get_openai_client()
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text[:8000]  # seguridad por tamaño
    )
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    v /= (np.linalg.norm(v) + 1e-9)
    return v

def cosine_sim(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def search_similar(qvec: np.ndarray, topk: int = TOPK):
    """
    Recupera los chunks más similares (coseno) desde SQLite.
    """
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
      SELECT e.embedding, c.content, c.doc_id
      FROM embeddings e
      JOIN chunks c ON c.id = e.chunk_id
    """)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return []

    scored = []
    for r in rows:
        emb = np.frombuffer(r["embedding"], dtype=np.float32)
        scored.append((cosine_sim(qvec, emb), r["content"], r["doc_id"]))
    scored.sort(key=lambda x: x[0], reverse=True)

    out = []
    for score, content, doc_id in scored[:topk]:
        d = get_db().execute(
            "SELECT title, source_url, published_at FROM docs WHERE id=?",
            (doc_id,)
        ).fetchone()
        out.append({
            "score": score,
            "content": content,
            "title": d["title"] if d else None,
            "source_url": d["source_url"] if d else None,
            "published_at": d["published_at"] if d else None
        })
    return out


# =========================
# LLM (respuesta basada en evidencia)
# =========================
def llm_answer(question: str, context: str) -> str:
    """
    Usa GPT para sintetizar SOLO lo del contexto.
    """
    client = get_openai_client()
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un asistente médico informativo basado en evidencia cargada por el usuario. "
                    "Responde en español claro para público general. "
                    "Usa EXCLUSIVAMENTE la evidencia del contexto; si falta, dilo. "
                    "No diagnostiques ni prescribas. "
                    "Termina con: 'Esta información no sustituye una consulta médica.'"
                )
            },
            {
                "role": "user",
                "content": f"Pregunta: {question}\n\nContexto de evidencia (citas textuales):\n{context}"
            }
        ]
    )
    return completion.choices[0].message.content.strip()


# =========================
# Rutas
# =========================
@app.route("/")
def index():
    # Sirve tu interfaz: avatar_chat.html, upload.html, ingest.html están en /static
    return send_from_directory(app.static_folder, "avatar_chat.html")

@app.route("/health")
def health():
    return jsonify({"ok": True, "ts": datetime.utcnow().isoformat()}), 200


@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    """
    Sube un PDF, extrae texto, trocea, guarda embeddings.
    Front sugerido en /static/upload.html
    """
    if "file" not in request.files:
        return jsonify({"error": "No se envió archivo PDF"}), 400
    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Solo se aceptan archivos PDF"}), 400

    # Extraer texto
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    if not text.strip():
        return jsonify({"error": "No se pudo extraer texto del PDF"}), 400

    title = os.path.splitext(file.filename)[0]

    conn = get_db(); cur = conn.cursor()
    cur.execute(
        "INSERT INTO docs(title, source_url, published_at, added_at) VALUES (?,?,?,?)",
        (title, None, None, datetime.utcnow().isoformat())
    )
    doc_id = cur.lastrowid

    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    for idx, ch in enumerate(chunks):
        cur.execute(
            "INSERT INTO chunks(doc_id, chunk_index, content, tokens) VALUES (?,?,?,?)",
            (doc_id, idx, ch, len(ch.split()))
        )
        chunk_id = cur.lastrowid
        v = embed_text(ch).tobytes()
        cur.execute("INSERT INTO embeddings(chunk_id, embedding) VALUES (?,?)", (chunk_id, v))

    conn.commit(); conn.close()

    return jsonify({"ok": True, "title": title, "chunks": len(chunks)}), 200


@app.route("/ingest", methods=["POST"])
def ingest():
    """
    Ingesta de texto plano (para copiar/pegar guías, resúmenes, etc.).
    Protege con ALLOW_INGEST en prod.
    """
    if not ALLOW_INGEST:
        return jsonify({"ok": False, "error": "Ingesta deshabilitada (ALLOW_INGEST=false)"}), 403

    data = request.get_json(force=True)
    title = data.get("title") or "Fuente interna"
    source_url = data.get("source_url")
    published_at = data.get("published_at")
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"ok": False, "error": "Falta 'text'"}), 400

    conn = get_db(); cur = conn.cursor()
    cur.execute(
        "INSERT INTO docs(title, source_url, published_at, added_at) VALUES (?,?,?,?)",
        (title, source_url, published_at, datetime.utcnow().isoformat())
    )
    doc_id = cur.lastrowid

    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    for idx, ch in enumerate(chunks):
        cur.execute(
            "INSERT INTO chunks(doc_id, chunk_index, content, tokens) VALUES (?,?,?,?)",
            (doc_id, idx, ch, len(ch.split()))
        )
        chunk_id = cur.lastrowid
        v = embed_text(ch).tobytes()
        cur.execute("INSERT INTO embeddings(chunk_id, embedding) VALUES (?,?)", (chunk_id, v))

    conn.commit(); conn.close()

    return jsonify({"ok": True, "doc_id": doc_id, "chunks": len(chunks)}), 200


@app.route("/ask", methods=["POST"])
def ask():
    """
    Respuesta basada SOLO en la evidencia guardada.
    """
    try:
        j = request.get_json(force=True)
    except Exception:
        return jsonify({"answer": "Solicitud inválida (JSON).", "sources": []}), 400

    q = (j.get("question") or "").strip()
    if not q:
        return jsonify({"answer": "Por favor escribe tu pregunta médica.", "sources": []}), 200

    try:
        qvec = embed_text(q)
    except Exception as e:
        return jsonify({"answer": f"Error generando embeddings: {e}", "sources": []}), 500

    ctx = search_similar(qvec, topk=TOPK)

    if not ctx:
        return jsonify({
            "answer": (
                "No encontré evidencia en los documentos cargados para esa pregunta. "
                "Puedes subir más PDFs o textos para ampliar la base. "
                "Esta información no sustituye una consulta médica."
            ),
            "sources": [],
            "timestamp": datetime.utcnow().isoformat()
        }), 200

    # Armar contexto para el LLM
    context_text = "\n\n---\n\n".join([
        f"[{c['title'] or 'Fuente interna'} | {c['published_at'] or 's/f'} | {c['source_url'] or 's/u'}]\n{c['content']}"
        for c in ctx
    ])

    try:
        answer = llm_answer(q, context_text)
    except Exception as e:
        return jsonify({"answer": f"Error generando respuesta: {e}", "sources": []}), 500

    sources = [
        {
            "title": c["title"] or "Fuente interna",
            "url": c["source_url"],
            "date": c["published_at"]
        }
        for c in ctx[:3]
    ]

    return jsonify({
        "answer": answer,
        "sources": sources,
        "timestamp": datetime.utcnow().isoformat()
    }), 200


# =========================
# Run local
# =========================
if __name__ == "__main__":
    # Para desarrollo local
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)
