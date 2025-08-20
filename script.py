import asyncio
import discord
import json
from pathlib import Path
import numpy as np
import ollama
import os

# =====================
# CONFIGURACIÓN GENERAL
# =====================
EMBEDDING_MODEL = 'hf.co/nomic-ai/nomic-embed-text-v2-moe-GGUF'
LANGUAGE_MODEL = 'hf.co/tensorblock/scb10x_llama3.1-typhoon2-8b-instruct-GGUF'
DATA_DIR = Path("data")
INDEX_FILE = Path("vector_db.json")
TOP_N = 5
MAX_WORDS = 1000
OVERLAP = 200

# ======================
# ESTRUCTURAS DE MEMORIA
# ======================
VECTOR_DB = {}       # {materia: [(chunk, embedding)]}
CHAT_HISTORY = {}    # {materia: [(usuario, asistente)]}

# ======================
# FUNCIONES DE UTILIDAD
# ======================
def chunk_by_words(text, max_words=1000, overlap=200):
    words = text.split()
    step = max(1, max_words - overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i:i+max_words]
        if not chunk_words:
            break
        yield ' '.join(chunk_words)

def l2_normalize(vec):
    v = np.array(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    return v if n == 0 or not np.isfinite(n) else v / n

def similaridad_coseno(a, b):
    return float(np.dot(a, b))

# =====================
# CARGA DE DATASETS
# =====================
def load_datasets():
    """Carga los embeddings desde cache o los genera."""
    if INDEX_FILE.exists():
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        for materia, data in cache.items():
            VECTOR_DB[materia] = [(c, np.array(e, dtype=np.float32)) for c, e in data]
            CHAT_HISTORY[materia] = []
        print(f"Embeddings cargados desde {INDEX_FILE}")
        return

    for subject_dir in DATA_DIR.iterdir():
        if subject_dir.is_dir():
            materia = subject_dir.name.lower()
            VECTOR_DB[materia] = []
            CHAT_HISTORY[materia] = []
            files = sorted(subject_dir.glob("*.txt"))

            for txt in files:
                with txt.open("r", encoding="utf-8") as f:
                    text = f.read().strip()
                    for chunk in chunk_by_words(text):
                        emb = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
                        VECTOR_DB[materia].append((f"[{txt.name}] {chunk}", l2_normalize(emb)))

            print(f"Materia '{materia}': {len(VECTOR_DB[materia])} chunks procesados.")

    # Guardar embeddings en cache
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump({m: [(c, e.tolist()) for c, e in chunks] for m, chunks in VECTOR_DB.items()}, f)
    print(f"Embeddings guardados en {INDEX_FILE}")

# =========================
# DETECCIÓN DE MATERIA
# =========================
def detect_subject(query):
    materias = list(VECTOR_DB.keys())
    prompt = f"""Eres un asistente que clasifica preguntas por materia.
Materias disponibles: {', '.join(materias)}

Pregunta: "{query}"

Responde solo con el nombre exacto de la materia más relacionada."""
    response = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[{'role': 'system', 'content': prompt}],
    )
    subject = response['message']['content'].strip().lower()
    return subject if subject in materias else None

# =========================
# RECUPERACIÓN DE CONTEXTO
# =========================
def retrieve(query, subject):
    q_emb = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    q = l2_normalize(q_emb)
    sims = [(c, similaridad_coseno(q, emb)) for c, emb in VECTOR_DB[subject]]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:TOP_N]

def build_instruction_prompt(retrieved_knowledge, chat_history):
    history_str = "\n".join([f"Usuario: {u}\nAsistente: {a}" for u, a in chat_history])
    context_str = "\n".join([f" - {chunk}" for chunk, _ in retrieved_knowledge])
    return f"""[ROL]
    Eres un profesor universitario experto en ciencias de la computación.

    [OBJETIVO]
    Explica conceptos de forma clara, progresiva y precisa, basándote en el contexto provisto.
    1. Primero explica sencillo, luego más técnico.
    2. Siempre da un ejemplo inventado.
    3. Usa analogías si el concepto es abstracto.
    4. Mantén un tono académico, pero accesible.

    [REGLAS]
    - Usa SOLO el contexto si es suficiente.
    - Si no está en el contexto, usa tu conocimiento general y acláralo.
    - No inventes definiciones incorrectas.

    Contexto relevante:
    {context_str}

    Historial:
    {history_str}
    """

# ======================
# GENERACIÓN DE RESPUESTA
# ======================
def generate_answer(query, retrieved, subject):
    messages = [
        {'role': 'system', 'content': build_instruction_prompt(retrieved, CHAT_HISTORY[subject])},
        {'role': 'user', 'content': query},
    ]
    resp = ollama.chat(model=LANGUAGE_MODEL, messages=messages, stream=False)
    return resp["message"]["content"]

# =====================
# BOT DE DISCORD
# =====================
intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)

@bot.event
async def on_ready():
    print(f"✅ Bot conectado como {bot.user}")

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    content = message.content.strip()

    if content.startswith("!ask "):
        query = content[5:].strip()
        if not query:
            await message.channel.send("⚠️ Tenés que poner una pregunta.")
            return

        subject = detect_subject(query)
        if not subject:
            await message.channel.send("⚠️ No pude determinar la materia. Probá ser más específico.")
            return

        retrieved = retrieve(query, subject)
        answer = generate_answer(query, retrieved, subject)
        CHAT_HISTORY[subject].append((query, answer))

        # Discord tiene límite de 2000 caracteres
        for i in range(0, len(answer), 1900):
            await message.channel.send(answer[i:i+1900])

# =====================
# INICIO DEL BOT
# =====================
if __name__ == "__main__":
    print("Cargando datasets y generando embeddings...")
    load_datasets()
    bot.run(os.getenv("DISCORD_TOKEN"))
