# Avatar Médico (sin labios) + Base RAG mínima

Este paquete monta un **agente médico informativo** (no diagnóstica) con:
- **Frontend** estático (`/static/avatar_chat.html`): chat con imagen fija del “avatar”.
- **Backend Flask** (`app.py`): endpoints `/ask`, `/ingest`, `/health`.
- **RAG mínimo** con SQLite: ingesta de texto -> chunking -> "embeddings" placeholder -> búsqueda coseno.
- **Render**: `render.yaml` listo para crear el servicio web.

> ⚠️ **Importante**: Los embeddings y el LLM son *placeholders de demo*. En producción debes conectar:
> - Embeddings reales (p. ej., OpenAI text-embedding-3-large) y
> - Un LLM real para respuestas.
> El resto de la arquitectura y los endpoints se mantienen.

## Estructura
```
app.py
requirements.txt
render.yaml
Procfile
static/
  └─ avatar_chat.html
```

## Despliegue en Render
1. Crea un **nuevo Web Service** desde este repositorio/zip.
2. Render reconocerá `render.yaml` (o configura manualmente):
   - Build: `pip install -r requirements.txt`
   - Start: `gunicorn app:app -w 2 -k gthread -b 0.0.0.0:$PORT --threads 8 --timeout 120`
3. Variables de entorno (ya en `render.yaml`, puedes ajustarlas en el panel):
   - `DB_PATH=medical_rag.db`
   - `ALLOW_INGEST=true` (desactiva a `false` en producción si bloqueas /ingest)
   - `EMBED_DIM=1536`

## Probar
- Abre `https://TU_DOMINIO/` → verás el chat (imagen fija).
- Endpoint de salud: `GET /health`

## Ingesta de contenido
Envía texto (guías, FAQs) para que el agente tenga contexto.

```bash
curl -X POST https://TU_DOMINIO/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "title":"Guía de hipertensión (resumen)",
    "source_url":"https://ejemplo.org/guias/hta",
    "published_at":"2024-10-01",
    "text":"La hipertensión arterial es... (pega aquí el texto largo)"
  }'
```

> En producción pon `ALLOW_INGEST=false` y expón ingesta solo por backoffice (o exige API key).

## Consultar (desde el chat o via API)
```bash
curl -X POST https://TU_DOMINIO/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"¿Qué es la hipertensión y cómo se clasifica?"}'
```

## Reemplazos para producción
1. **Embeddings reales**  
   - Guarda `np.array(dtype=float32).tobytes()` en la tabla `embeddings`.  
   - Ejemplo pseudo-código:
   ```python
   import numpy as np
   from openai import OpenAI
   client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
   def embed_text(text):
       out = client.embeddings.create(model="text-embedding-3-large", input=text)
       v = np.array(out.data[0].embedding, dtype=np.float32)
       return v / (np.linalg.norm(v) + 1e-9)
   ```
   - **Sugerencia**: migrar a **Postgres + pgvector** si crecerá (>50k chunks).

2. **LLM real**  
   Cambia `llm_answer()` por llamada a tu proveedor (OpenAI, etc.) con **prompt con contexto** (RAG).

3. **Seguridad y cumplimiento**  
   - Añadir disclaimers en UI y respuesta.
   - Nunca dar diagnósticos ni prescribir.
   - Log de auditoría (pregunta, pasajes, similitud, timestamp, IP opcional).
   - Rate-limiting / API-keys en `/ask` y `/ingest`.

4. **Branding**  
   Cambia la imagen del avatar y colores en `static/avatar_chat.html`.

## Desarrollo local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py  # http://localhost:8000
```

## Notas
- Esta demo no usa CORS porque frontend y backend son el mismo servicio. Si separas frontend, agrega CORS.
- Para WhatsApp/IVR, simplemente llama a `/ask` y entrega la respuesta por el canal correspondiente.
