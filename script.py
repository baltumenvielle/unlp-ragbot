import asyncio
import discord
from discord.ext import commands
import json
import aiohttp
from pathlib import Path
import numpy as np
import ollama
import os
import faiss
import pickle
import datetime
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Descargar recursos necesarios de NLTK (solo la primera vez)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# =====================
# CONFIGURACIÓN GENERAL
# =====================
EMBEDDING_MODEL = 'hf.co/nomic-ai/nomic-embed-text-v2-moe-GGUF'
LANGUAGE_MODEL = 'hf.co/tensorblock/scb10x_llama3.1-typhoon2-8b-instruct-GGUF'
DATA_DIR = Path("data")
INDEX_FILE = Path("vector_db.json")
MEMORY_INDEX_FILE = Path("memory_index.pkl")
MEMORY_DATA_FILE = Path("memory_data.json")
TF_IDF_FILE = Path("tfidf_data.pkl")
TOP_N = 5
TOP_MEMORY = 3
EMBEDDING_DIM = 768
CONTEXT_WINDOW = 3  # Últimas N interacciones para contexto semántico
SEMANTIC_THRESHOLD = 0.4  # Umbral para continuidad semántica
CHUNK_MIN_SENTENCES = 2  # Mínimo de oraciones por chunk
CHUNK_MAX_SENTENCES = 8  # Máximo de oraciones por chunk
RECENCY_WEIGHT = 0.3  # Peso para relevancia temporal en memoria

# ======================
# ESTRUCTURAS DE MEMORIA
# ======================
VECTOR_DB = {}       # {materia: [(chunk, embedding, metadata)]}
CHAT_HISTORY = {}    # {materia: [(usuario, asistente)]}
TF_IDF_VECTORIZERS = {}  # {materia: TfidfVectorizer}
TF_IDF_MATRICES = {}     # {materia: matriz TF-IDF}

# Memoria semántica con FAISS y metadatos
MEMORY_INDEX = None      # Índice FAISS para búsqueda rápida
MEMORY_CONVERSATIONS = [] # Lista: [(query, answer, subject, embedding, metadata)]

# Contexto semántico por usuario (embeddings de conversaciones)
USER_CONTEXTS = {}  # {user_id: {'interactions': [{'query', 'answer', 'subject', 'timestamp', 'embedding'}]}}

# ======================
# FUNCIONES DE UTILIDAD
# ======================
def l2_normalize(vec):
    v = np.array(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    return v if n == 0 or not np.isfinite(n) else v / n

def similaridad_coseno(a, b):
    return float(np.dot(l2_normalize(a), l2_normalize(b)))

def get_timestamp():
    return datetime.datetime.now().isoformat()

def calculate_recency_score(timestamp):
    """Calcula un score de recencia (más reciente = mayor score)."""
    try:
        conv_time = datetime.datetime.fromisoformat(timestamp)
        now = datetime.datetime.now()
        hours_ago = (now - conv_time).total_seconds() / 3600
        # Score decrece exponencialmente: 1.0 para conversaciones recientes, ~0.1 para >48h
        return max(0.1, np.exp(-hours_ago / 24))
    except:
        return 0.5  # Score neutral si hay error

# ======================
# CHUNKING SEMÁNTICO INTELIGENTE
# ======================
def semantic_chunking(text, min_sentences=CHUNK_MIN_SENTENCES, max_sentences=CHUNK_MAX_SENTENCES):
    """
    Chunking semántico basado en oraciones completas y densidad de contenido.
    """
    try:
        sentences = sent_tokenize(text)
    except:
        # Fallback si NLTK falla
        sentences = text.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
    
    if not sentences:
        return [text]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        words = len(sentence.split())
        
        # Si agregar esta oración excede el máximo Y ya tenemos el mínimo
        if (len(current_chunk) >= min_sentences and 
            len(current_chunk) >= max_sentences):
            # Guardar chunk actual
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
        else:
            current_chunk.append(sentence)
    
    # Agregar último chunk si no está vacío
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Si algún chunk es demasiado corto, combinarlo con el anterior
    final_chunks = []
    for chunk in chunks:
        if len(chunk.split()) < 20 and final_chunks:  # Muy corto
            final_chunks[-1] += ' ' + chunk
        else:
            final_chunks.append(chunk)
    
    return final_chunks if final_chunks else [text]

# ======================
# CONTEXTO SEMÁNTICO POR USUARIO
# ======================
def update_user_semantic_context(user_id, query, answer, subject):
    """Actualiza el contexto semántico del usuario con embeddings."""
    if user_id not in USER_CONTEXTS:
        USER_CONTEXTS[user_id] = {'interactions': []}
    
    # Generar embedding de la interacción completa
    interaction_text = f"Q: {query} A: {answer}"
    try:
        embedding = ollama.embed(model=EMBEDDING_MODEL, input=interaction_text)['embeddings'][0]
        embedding = l2_normalize(embedding)
    except:
        embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    
    interaction = {
        'query': query,
        'answer': answer,
        'subject': subject,
        'timestamp': get_timestamp(),
        'embedding': embedding
    }
    
    # Mantener solo las últimas CONTEXT_WINDOW interacciones
    USER_CONTEXTS[user_id]['interactions'].append(interaction)
    if len(USER_CONTEXTS[user_id]['interactions']) > CONTEXT_WINDOW:
        USER_CONTEXTS[user_id]['interactions'].pop(0)

def detect_semantic_continuity(user_id, query):
    """
    Detecta continuidad semántica comparando embeddings de la conversación.
    """
    if user_id not in USER_CONTEXTS or not USER_CONTEXTS[user_id]['interactions']:
        return False, None, 0.0
    
    # Generar embedding de la nueva consulta
    try:
        query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
        query_embedding = l2_normalize(query_embedding)
    except:
        return False, None, 0.0
    
    interactions = USER_CONTEXTS[user_id]['interactions']
    
    # Calcular similitud con la última interacción
    last_interaction = interactions[-1]
    similarity = similaridad_coseno(query_embedding, last_interaction['embedding'])
    
    # Si la similitud es alta, es continuación
    if similarity >= SEMANTIC_THRESHOLD:
        return True, last_interaction['subject'], similarity
    
    # Revisar las últimas 2-3 interacciones para detectar continuidad indirecta
    if len(interactions) >= 2:
        for i in range(min(3, len(interactions))):
            interaction = interactions[-(i+1)]
            sim = similaridad_coseno(query_embedding, interaction['embedding'])
            if sim >= SEMANTIC_THRESHOLD * 0.8:  # Umbral ligeramente menor
                return True, interaction['subject'], sim
    
    return False, None, 0.0

# ======================
# MEMORIA SEMÁNTICA MEJORADA
# ======================
def init_memory_index():
    """Inicializa el índice FAISS para memoria semántica."""
    global MEMORY_INDEX
    MEMORY_INDEX = faiss.IndexFlatIP(EMBEDDING_DIM)

def load_memory():
    """Carga conversaciones previas desde disco."""
    global MEMORY_CONVERSATIONS, MEMORY_INDEX
    
    if MEMORY_DATA_FILE.exists() and MEMORY_INDEX_FILE.exists():
        # Cargar datos de conversaciones con metadatos
        with open(MEMORY_DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            MEMORY_CONVERSATIONS = []
            for item in data:
                # Compatibilidad con formato anterior y nuevo
                metadata = item.get('metadata', {
                    'timestamp': item.get('timestamp', get_timestamp()),
                    'user_id': item.get('user_id', 'unknown'),
                    'interaction_count': 1
                })
                MEMORY_CONVERSATIONS.append((
                    item['query'], 
                    item['answer'], 
                    item['subject'],
                    np.array(item['embedding'], dtype=np.float32),
                    metadata
                ))
        
        # Cargar índice FAISS
        MEMORY_INDEX = faiss.read_index(str(MEMORY_INDEX_FILE))
        print(f"Memoria cargada: {len(MEMORY_CONVERSATIONS)} conversaciones")
    else:
        init_memory_index()
        print("Memoria inicializada vacía")

def save_memory():
    """Guarda la memoria semántica en disco."""
    # Guardar datos de conversaciones con metadatos
    data = []
    for q, a, s, e, metadata in MEMORY_CONVERSATIONS:
        data.append({
            'query': q, 
            'answer': a, 
            'subject': s, 
            'embedding': e.tolist(),
            'metadata': metadata
        })
    
    with open(MEMORY_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Guardar índice FAISS
    faiss.write_index(MEMORY_INDEX, str(MEMORY_INDEX_FILE))

def add_to_memory(query, answer, subject, user_id):
    """Añade una nueva conversación a la memoria semántica con metadatos."""
    # Generar embedding para la pregunta
    query_emb = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    query_emb = l2_normalize(query_emb)
    
    # Crear metadatos
    metadata = {
        'timestamp': get_timestamp(),
        'user_id': user_id,
        'interaction_count': len([c for c in MEMORY_CONVERSATIONS if c[4].get('user_id') == user_id]) + 1
    }
    
    # Añadir a la lista de conversaciones
    MEMORY_CONVERSATIONS.append((query, answer, subject, query_emb, metadata))
    
    # Añadir al índice FAISS
    MEMORY_INDEX.add(query_emb.reshape(1, -1))
    
    # Guardar cada 3 conversaciones nuevas
    if len(MEMORY_CONVERSATIONS) % 3 == 0:
        save_memory()

def retrieve_from_memory_weighted(query, user_id, top_k=TOP_MEMORY):
    """
    Recupera conversaciones relevantes con ponderación por similitud, recencia y usuario.
    """
    if len(MEMORY_CONVERSATIONS) == 0:
        return []
    
    # Generar embedding de la consulta
    query_emb = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    query_emb = l2_normalize(query_emb)
    
    # Buscar en el índice FAISS (más candidatos para re-ranking)
    search_k = min(top_k * 3, len(MEMORY_CONVERSATIONS))
    similarities, indices = MEMORY_INDEX.search(query_emb.reshape(1, -1), search_k)
    
    # Calcular scores ponderados
    weighted_results = []
    for i, idx in enumerate(indices[0]):
        if idx >= 0:
            conv = MEMORY_CONVERSATIONS[idx]
            query_conv, answer_conv, subject_conv, emb_conv, metadata = conv
            
            # Score base de similitud semántica
            semantic_score = similarities[0][i]
            
            # Score de recencia (conversaciones más recientes tienen mayor peso)
            recency_score = calculate_recency_score(metadata['timestamp'])
            
            # Score de relevancia del usuario (conversaciones del mismo usuario pesan más)
            user_score = 1.5 if metadata.get('user_id') == user_id else 1.0
            
            # Score final ponderado
            final_score = (
                semantic_score * 0.5 +      # 50% similitud semántica
                recency_score * RECENCY_WEIGHT +  # 30% recencia
                (user_score - 1.0) * 0.2    # 20% relevancia de usuario
            )
            
            if semantic_score > 0.4:  # Umbral mínimo de similitud
                weighted_results.append({
                    'query': query_conv,
                    'answer': answer_conv,
                    'subject': subject_conv,
                    'similarity': semantic_score,
                    'recency_score': recency_score,
                    'user_score': user_score,
                    'final_score': final_score,
                    'timestamp': metadata['timestamp']
                })
    
    # Ordenar por score final y retornar top_k
    weighted_results.sort(key=lambda x: x['final_score'], reverse=True)
    return weighted_results[:top_k]

# ======================
# CLASIFICACIÓN HÍBRIDA DE MATERIAS
# ======================
def build_tfidf_models():
    """Construye modelos TF-IDF para cada materia."""
    global TF_IDF_VECTORIZERS, TF_IDF_MATRICES
    
    for subject, chunks in VECTOR_DB.items():
        if not chunks:
            continue
        
        # Extraer textos de chunks
        texts = [chunk[0] for chunk in chunks]  # chunk[0] es el texto
        
        # Crear vectorizador TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=2
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            TF_IDF_VECTORIZERS[subject] = vectorizer
            TF_IDF_MATRICES[subject] = tfidf_matrix
        except ValueError:
            # Si no hay suficiente texto para TF-IDF
            TF_IDF_VECTORIZERS[subject] = None
            TF_IDF_MATRICES[subject] = None

def classify_with_tfidf(query):
    """Clasifica usando TF-IDF."""
    if not TF_IDF_VECTORIZERS:
        return {}
    
    scores = {}
    for subject, vectorizer in TF_IDF_VECTORIZERS.items():
        if vectorizer is None:
            continue
        
        try:
            # Transformar query a vector TF-IDF
            query_vec = vectorizer.transform([query])
            
            # Calcular similitud con el centroide de la materia
            tfidf_matrix = TF_IDF_MATRICES[subject]
            centroid = np.mean(tfidf_matrix.toarray(), axis=0).reshape(1, -1)
            similarity = cosine_similarity(query_vec, centroid)[0][0]
            scores[subject] = similarity
        except:
            scores[subject] = 0.0
    
    return scores

def detect_subject_semantic(query):
    """Clasificación semántica mejorada."""
    if not VECTOR_DB:
        return {}
    
    # Generar embedding de la consulta
    try:
        query_emb = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
        query_emb = l2_normalize(query_emb)
    except:
        return {}
    
    subject_scores = {}
    for subject, chunks in VECTOR_DB.items():
        if not chunks:
            continue
        
        # Calcular similitud con top 5 chunks más relevantes
        similarities = []
        for chunk_text, emb, metadata in chunks:
            sim = similaridad_coseno(query_emb, emb)
            similarities.append(sim)
        
        # Promedio ponderado de los top 5
        similarities.sort(reverse=True)
        top_sims = similarities[:5]
        if top_sims:
            # Dar más peso a los chunks más similares
            weights = [0.4, 0.3, 0.15, 0.1, 0.05][:len(top_sims)]
            weighted_score = sum(sim * weight for sim, weight in zip(top_sims, weights))
            subject_scores[subject] = weighted_score
        else:
            subject_scores[subject] = 0.0
    
    return subject_scores

def detect_subject_llm(query):
    """Clasificación con LLM usando ejemplos representativos."""
    materias = list(VECTOR_DB.keys())
    if not materias:
        return {}
    
    # Crear ejemplos más representativos
    examples = {}
    for subject, chunks in VECTOR_DB.items():
        if chunks:
            # Seleccionar chunks más diversos
            examples[subject] = []
            chunk_texts = [chunk[0] for chunk in chunks]
            
            # Tomar chunks de diferentes archivos si es posible
            seen_files = set()
            for chunk_text in chunk_texts[:10]:  # Revisar los primeros 10
                file_marker = chunk_text.split(']')[0] + ']' if ']' in chunk_text else ''
                if file_marker not in seen_files:
                    clean_text = chunk_text.split('] ', 1)[1] if '] ' in chunk_text else chunk_text
                    examples[subject].append(clean_text[:80] + '...')
                    seen_files.add(file_marker)
                    if len(examples[subject]) >= 3:  # Máximo 3 ejemplos por materia
                        break
    
    examples_str = ""
    for subject, sample_chunks in examples.items():
        examples_str += f"\n{subject.upper()}: {' | '.join(sample_chunks)}"
    
    prompt = f"""Analiza esta pregunta y clasifícala según las materias disponibles:

CONTENIDOS POR MATERIA:{examples_str}

PREGUNTA: "{query}"

Responde SOLO con el nombre exacto de la materia (en minúsculas) que mejor corresponda.
Si no estás seguro, responde "general"."""
    
    try:
        response = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1}
        )
        result = response['message']['content'].strip().lower()
        return {result: 0.8} if result in materias else {'general': 0.5}
    except:
        return {}

def smart_subject_classification(query, user_id):
    """
    Clasificación híbrida que combina contexto semántico, TF-IDF, embeddings y LLM.
    """
    # 1. Verificar continuidad semántica primero
    is_continuation, prev_subject, sem_similarity = detect_semantic_continuity(user_id, query)
    
    if is_continuation and sem_similarity > SEMANTIC_THRESHOLD:
        print(f"🔄 Continuidad semántica detectada → {prev_subject} (sim: {sem_similarity:.3f})")
        return prev_subject
    
    # 2. Si no hay continuidad, clasificar con métodos híbridos
    semantic_scores = detect_subject_semantic(query)
    tfidf_scores = classify_with_tfidf(query)
    llm_scores = detect_subject_llm(query)
    
    # 3. Combinar scores con pesos dinámicos
    combined_scores = {}
    all_subjects = set(semantic_scores.keys()) | set(tfidf_scores.keys()) | set(llm_scores.keys())
    
    for subject in all_subjects:
        # Obtener scores (0 si no existe)
        sem_score = semantic_scores.get(subject, 0.0)
        tfidf_score = tfidf_scores.get(subject, 0.0)
        llm_score = llm_scores.get(subject, 0.0)
        
        # Calcular confianza de cada método
        llm_confidence = 0.1  # Confianza fija para LLM
        
        # Score final combinado
        combined_scores[subject] = (
            sem_score * 0.7 + 
            tfidf_score * 0.2 + 
            llm_confidence * 0.1
        )
    
    # 4. Seleccionar la mejor materia
    if combined_scores:
        best_subject = max(combined_scores.items(), key=lambda x: x[1])
        if best_subject[1] > 0.3:  # Umbral mínimo
            print(f"🎯 Clasificación híbrida → {best_subject[0]} (score: {best_subject[1]:.3f})")
            return best_subject[0]
    
    print(f"❓ Clasificación incierta → usando 'general'")
    return "general"

# =====================
# CARGA DE DATASETS CON CHUNKING SEMÁNTICO
# =====================
def load_datasets():
    """Carga los embeddings desde cache o los genera con chunking semántico."""
    global TF_IDF_VECTORIZERS, TF_IDF_MATRICES
    
    if INDEX_FILE.exists():
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        
        for materia, data in cache.items():
            VECTOR_DB[materia] = []
            CHAT_HISTORY[materia] = []
            
            for item in data:
                # Compatibilidad con formato anterior y nuevo
                if len(item) == 2:  # Formato viejo: [chunk, embedding]
                    chunk_text, embedding = item
                    metadata = {'file': 'unknown', 'chunk_id': len(VECTOR_DB[materia])}
                else:  # Formato nuevo: [chunk, embedding, metadata]
                    chunk_text, embedding, metadata = item
                
                VECTOR_DB[materia].append((
                    chunk_text, 
                    np.array(embedding, dtype=np.float32),
                    metadata
                ))
        
        print(f"Embeddings cargados desde {INDEX_FILE}")
        
        # Construir modelos TF-IDF
        print("Construyendo modelos TF-IDF...")
        build_tfidf_models()
        return

    # Generar embeddings con chunking semántico
    print("Generando embeddings con chunking semántico...")
    for subject_dir in DATA_DIR.iterdir():
        if subject_dir.is_dir():
            materia = subject_dir.name.lower()
            VECTOR_DB[materia] = []
            CHAT_HISTORY[materia] = []
            files = sorted(subject_dir.glob("*.txt"))

            for txt in files:
                print(f"Procesando {txt.name}...")
                with txt.open("r", encoding="utf-8") as f:
                    text = f.read().strip()
                    
                    # Usar chunking semántico en lugar de por palabras
                    chunks = semantic_chunking(text)
                    
                    for i, chunk in enumerate(chunks):
                        # Agregar identificador del archivo al chunk
                        chunk_with_id = f"[{txt.name}] {chunk}"
                        
                        # Generar embedding
                        emb = ollama.embed(model=EMBEDDING_MODEL, input=chunk_with_id)['embeddings'][0]
                        
                        # Crear metadatos
                        metadata = {
                            'file': txt.name,
                            'chunk_id': i,
                            'original_length': len(chunk),
                            'sentences': len(chunk.split('. '))
                        }
                        
                        VECTOR_DB[materia].append((
                            chunk_with_id, 
                            l2_normalize(emb),
                            metadata
                        ))

            print(f"Materia '{materia}': {len(VECTOR_DB[materia])} chunks procesados.")

    # Guardar embeddings en cache con metadatos
    cache_data = {}
    for materia, chunks in VECTOR_DB.items():
        cache_data[materia] = [
            [chunk_text, emb.tolist(), metadata] 
            for chunk_text, emb, metadata in chunks
        ]
    
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
    print(f"Embeddings guardados en {INDEX_FILE}")
    
    # Construir modelos TF-IDF
    print("Construyendo modelos TF-IDF...")
    build_tfidf_models()

# =========================
# FUNCIONES DE COMPATIBILIDAD Y UTILIDAD
# =========================
def clear_user_context(user_id):
    """Limpia el contexto del usuario."""
    if user_id in USER_CONTEXTS:
        USER_CONTEXTS[user_id] = {'interactions': []}

def retrieve_from_memory(query, top_k=TOP_MEMORY):
    """Wrapper para compatibilidad - usa la versión mejorada."""
    return retrieve_from_memory_weighted(query, 'unknown', top_k)

def is_academic_question(query):
    """Detecta si una pregunta es académica (versión simplificada)."""
    academic_indicators = [
        'protocolo', 'red', 'redes', 'algoritmo', 'estructura', 'datos',
        'programación', 'código', 'función', 'clase', 'objeto', 'método',
        'variable', 'array', 'lista', 'árbol', 'grafo', 'nodo', 'recursión',
        'compilador', 'intérprete', 'sintaxis', 'base de datos', 'sql',
        'sistema operativo', 'proceso', 'memoria', 'cpu', 'cache',
        'búsqueda', 'ordenamiento', 'complejidad'
    ]
    
    query_lower = query.lower()
    return any(indicator in query_lower for indicator in academic_indicators)

def is_follow_up_question(query):
    """Detecta preguntas de seguimiento (simplificado)."""
    follow_up_indicators = [
        'explicalo', 'explica', 'más', 'profundidad', 'detalle', 'ejemplo',
        'cómo', 'por qué', 'cuál', 'también', 'además', 'pero', 'entonces'
    ]
    
    query_lower = query.lower()
    return any(indicator in query_lower for indicator in follow_up_indicators)

# =========================
# RECUPERACIÓN DE CONTEXTO MEJORADA
# =========================
def retrieve(query, subject):
    """Recupera chunks relevantes de una materia específica."""
    if subject not in VECTOR_DB or not VECTOR_DB[subject]:
        return []
    
    q_emb = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    q = l2_normalize(q_emb)
    
    # Calcular similitudes con todos los chunks
    sims = []
    for chunk_text, emb, metadata in VECTOR_DB[subject]:
        sim = similaridad_coseno(q, emb)
        sims.append((chunk_text, sim, metadata))
    
    # Ordenar por similitud y retornar top N
    sims.sort(key=lambda x: x[1], reverse=True)
    return [(chunk, sim) for chunk, sim, metadata in sims[:TOP_N]]

def update_user_context(user_id, query, subject):
    """Wrapper para compatibilidad - usa el contexto semántico."""
    # Esta función ahora se maneja automáticamente en update_user_semantic_context
    pass

def smart_subject_detection(query, user_id):
    """Wrapper que usa la nueva clasificación híbrida."""
    return smart_subject_classification(query, user_id)

def build_instruction_prompt(retrieved_knowledge, chat_history, memory_context=None, user_context=None):
    """Construye el prompt mejorado para el modelo."""
    history_str = "\n".join([f"Usuario: {u}\nAsistente: {a}" for u, a in chat_history[-3:]])
    context_str = "\n".join([f"- {chunk}" for chunk, _ in retrieved_knowledge])
    
    # Simplificar prompt para evitar confusión
    base_prompt = """Eres RAGBot, un asistente de estudio especializado en ciencias de la computación.

OBJETIVO: Responde de forma clara y precisa usando el contexto proporcionado.

REGLAS:
1. Si es una pregunta de seguimiento, conecta con la conversación anterior
2. Usa ejemplos prácticos cuando sea útil
3. Sé conciso pero completo
4. Mantén coherencia con respuestas previas"""

    if context_str:
        base_prompt += f"\n\nCONTEXTO DE DOCUMENTOS:\n{context_str}"
    
    if history_str:
        base_prompt += f"\n\nHISTORIAL RECIENTE:\n{history_str}"
    
    if memory_context:
        memory_str = "\n".join([f"- {conv['query']}: {conv['answer'][:100]}..." for conv in memory_context[:2]])
        base_prompt += f"\n\nCONVERSACIONES RELACIONADAS:\n{memory_str}"
    
    return base_prompt

def generate_answer(query, retrieved, subject, user_context=None, user_id='unknown'):
    """Genera respuesta usando el contexto completo mejorado."""
    # Recuperar contexto de memoria semántica ponderada
    memory_context = retrieve_from_memory_weighted(query, user_id, top_k=3)
    
    # Si no hay contexto relevante de documentos, permitir respuesta general
    if not any(sim > 0.3 for _, sim in retrieved):
        retrieved = []
    
    messages = [
        {'role': 'system', 'content': build_instruction_prompt(retrieved, CHAT_HISTORY[subject], memory_context, user_context)},
        {'role': 'user', 'content': query},
    ]
    
    resp = ollama.chat(
        model=LANGUAGE_MODEL, 
        messages=messages, 
        stream=False,
        options={'temperature': 0.1, 'top_p': 0.9}
    )
    
    # Limpiar respuesta si contiene partes del prompt
    response_content = resp["message"]["content"]
    
    # Filtrar líneas que parecen ser del prompt
    lines = response_content.split('\n')
    clean_lines = []
    for line in lines:
        # Omitir líneas que parecen ser del sistema
        if not any(keyword in line.upper() for keyword in ['OBJETIVO:', 'REGLAS:', 'CONTEXTO DE', 'HISTORIAL', 'CONVERSACIONES RELACIONADAS']):
            clean_lines.append(line)
    
    cleaned_response = '\n'.join(clean_lines).strip()
    return cleaned_response if cleaned_response else response_content

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
    user_id = str(message.author.id)

    if content.startswith("!ask "):
        query = content[5:].strip()
        if not query:
            await message.channel.send("⚠️ Tenés que poner una pregunta.")
            return

        # Usar clasificación híbrida inteligente
        subject = smart_subject_classification(query, user_id)
        
        # Crear materia si no existe
        if subject not in VECTOR_DB:
            VECTOR_DB[subject] = []
            CHAT_HISTORY[subject] = []

        # Recuperar documentos relevantes
        retrieved = retrieve(query, subject) if VECTOR_DB[subject] else []
        
        # Generar respuesta con contexto completo
        answer = generate_answer(query, retrieved, subject, None, user_id)
        
        # Actualizar contexto semántico del usuario
        update_user_semantic_context(user_id, query, answer, subject)
        
        # Guardar en memoria semántica e historial
        add_to_memory(query, answer, subject, user_id)
        CHAT_HISTORY[subject].append((query, answer))

        # Determinar tipo de información para mostrar
        is_continuation, prev_subject, similarity = detect_semantic_continuity(user_id, query)
        
        if is_continuation:
            debug_info = f"*[🔄 Continuando: {subject.upper()} (sim: {similarity:.2f})]*\n\n"
        elif is_academic_question(query):
            debug_info = f"*[🎯 Nueva consulta: {subject.upper()}]*\n\n"
        else:
            debug_info = f"*[💬 Conversación: {subject.upper()}]*\n\n"
        
        final_msg = f"{debug_info}{answer}"

        # Discord tiene límite de 2000 caracteres
        for i in range(0, len(final_msg), 1900):
            await message.channel.send(final_msg[i:i+1900])

    elif content == "!nuevo":
        """Comando para iniciar una nueva conversación."""
        clear_user_context(user_id)
        await message.channel.send("🆕 **Nueva conversación iniciada.** El contexto anterior se ha limpiado.")

    elif content == "!contexto":
        """Muestra el contexto semántico actual del usuario."""
        if user_id in USER_CONTEXTS and USER_CONTEXTS[user_id]['interactions']:
            interactions = USER_CONTEXTS[user_id]['interactions']
            ctx_msg = f"📋 **Tu contexto semántico ({len(interactions)} interacciones):**\n\n"
            
            for i, interaction in enumerate(interactions[-3:], 1):  # Mostrar últimas 3
                ctx_msg += f"**{i}.** [{interaction['subject']}] {interaction['query'][:60]}...\n"
            
            ctx_msg += f"\nUsa `!nuevo` para iniciar una conversación fresca."
        else:
            ctx_msg = "📋 No tenés ningún contexto semántico activo. Hacé una pregunta con `!ask` para empezar."
        
        await message.channel.send(ctx_msg)

    elif content.startswith("!memoria "):
        query = content[9:].strip()
        if not query:
            await message.channel.send("⚠️ Escribí una consulta para buscar en la memoria.")
            return
        
        memory_results = retrieve_from_memory_weighted(query, user_id, top_k=5)
        if not memory_results:
            await message.channel.send("🔍 No encontré conversaciones relacionadas en la memoria.")
            return
        
        response = "🧠 **Conversaciones relacionadas encontradas:**\n\n"
        for i, conv in enumerate(memory_results, 1):
            response += f"**{i}.** *[{conv['subject']}]* "
            response += f"Sem: {conv['similarity']:.2f} | Rec: {conv['recency_score']:.2f} | "
            response += f"Final: {conv['final_score']:.2f}\n"
            response += f"**P:** {conv['query'][:80]}...\n"
            response += f"**R:** {conv['answer'][:120]}...\n\n"
        
        for i in range(0, len(response), 1900):
            await message.channel.send(response[i:i+1900])
    
    elif content.startswith("!test "):
        query = content[6:].strip()
        if not query:
            await message.channel.send("⚠️ Escribí una pregunta para probar la clasificación.")
            return
        
        # Probar clasificación híbrida
        semantic_scores = detect_subject_semantic(query)
        tfidf_scores = classify_with_tfidf(query)
        llm_scores = detect_subject_llm(query)
        final_subject = smart_subject_classification(query, user_id)
        
        test_msg = f"🧪 **Test de clasificación híbrida:**\n"
        test_msg += f"Pregunta: `{query}`\n\n"
        test_msg += f"**Semántico:** {dict(list(semantic_scores.items())[:3])}\n"
        test_msg += f"**TF-IDF:** {dict(list(tfidf_scores.items())[:3])}\n"
        test_msg += f"**LLM:** {llm_scores}\n"
        test_msg += f"**Final:** {final_subject}\n"
        
        await message.channel.send(test_msg)
    
    elif content == "!help" or content == "!ayuda":
        """Muestra ayuda completa del bot."""
        help_msg = """🤖 **RAGBot - Asistente de Estudio Mejorado**

**💬 Conversación:**
• `!ask <pregunta>` - Hacer una pregunta (con contexto semántico)
• `!nuevo` - Iniciar nueva conversación (limpia contexto)
• `!contexto` - Ver tu contexto semántico actual

**🧠 Memoria Avanzada:**
• `!memoria <consulta>` - Buscar con ponderación temporal y de usuario

**🔧 Gestión:**
• `!stats` - Estadísticas avanzadas del bot
• `!reset_memoria` - ⚠️ Borrar toda la memoria

**🧪 Pruebas:**
• `!test <pregunta>` - Probar clasificación híbrida

**✨ Mejoras Implementadas:**
• **Contexto semántico:** Detecta continuidad por embeddings, no palabras clave
• **Clasificación híbrida:** Combina embeddings + TF-IDF + LLM con pesos dinámicos
• **Chunking semántico:** División inteligente por oraciones y densidad
• **Memoria ponderada:** Prioriza por similitud, recencia y relevancia de usuario

**Ejemplo avanzado:**
```
!ask qué es un protocolo de red?
!ask dame más detalles técnicos    # ← Detecta continuidad semántica
!ask y ejemplos de implementación   # ← Mantiene contexto automáticamente
```"""
        
        await message.channel.send(help_msg)

    elif content == "!stats":
        total_conversations = len(MEMORY_CONVERSATIONS)
        active_users = len([u for u in USER_CONTEXTS.values() if u.get('interactions')])
        materias_stats = {subject: len(chunks) for subject, chunks in VECTOR_DB.items()}
        tfidf_materias = len([v for v in TF_IDF_VECTORIZERS.values() if v is not None])
        
        # Calcular estadísticas de contexto semántico
        total_interactions = sum(len(u.get('interactions', [])) for u in USER_CONTEXTS.values())
        
        stats_msg = f"📊 **Estadísticas del Bot Mejorado**\n\n"
        stats_msg += f"🧠 Conversaciones en memoria: {total_conversations}\n"
        stats_msg += f"👥 Usuarios con contexto semántico: {active_users}\n"
        stats_msg += f"🔄 Total interacciones en contexto: {total_interactions}\n"
        stats_msg += f"📚 Materias con TF-IDF: {tfidf_materias}/{len(materias_stats)}\n\n"
        stats_msg += f"**Chunks por materia:**\n"
        for materia, chunks in materias_stats.items():
            stats_msg += f"  • {materia}: {chunks} chunks\n"
        
        await message.channel.send(stats_msg)
    
    elif content == "!reset_memoria":
        """Borra completamente la memoria semántica."""
        # Confirmar acción peligrosa
        confirm_msg = await message.channel.send(
            "⚠️ **¿Estás seguro de que querés borrar TODA la memoria?**\n"
            "Esto eliminará todas las conversaciones guardadas.\n"
            "Reaccioná con ✅ para confirmar o ❌ para cancelar."
        )
        await confirm_msg.add_reaction("✅")
        await confirm_msg.add_reaction("❌")
        
        def check(reaction, user):
            return user == message.author and str(reaction.emoji) in ["✅", "❌"] and reaction.message.id == confirm_msg.id
        
        try:
            reaction, user = await bot.wait_for('reaction_add', timeout=30.0, check=check)
            
            if str(reaction.emoji) == "✅":
                # Resetear memoria - usar las variables globales directamente
                MEMORY_CONVERSATIONS.clear()
                init_memory_index()
                
                # Eliminar archivos de memoria
                if MEMORY_DATA_FILE.exists():
                    MEMORY_DATA_FILE.unlink()
                if MEMORY_INDEX_FILE.exists():
                    MEMORY_INDEX_FILE.unlink()
                
                await message.channel.send("🧠 **Memoria completamente borrada.** El bot empezará desde cero.")
                
            else:
                await message.channel.send("❌ Operación cancelada. La memoria se mantiene intacta.")
                
        except asyncio.TimeoutError:
            await message.channel.send("⏰ Tiempo agotado. Operación cancelada.")

# =====================
# INICIO DEL BOT
# =====================
if __name__ == "__main__":
    print("🚀 Iniciando RAGBot Mejorado...")
    print("Cargando datasets con chunking semántico...")
    load_datasets()
    print("Inicializando memoria semántica avanzada...")
    load_memory()
    print("✅ Bot listo para funcionar!")
    
    # ⚠️ USAR VARIABLE DE ENTORNO PARA SEGURIDAD
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        print("❌ ERROR: Configure la variable DISCORD_TOKEN")
        print("En PowerShell: $env:DISCORD_TOKEN='TU_TOKEN'")
        exit(1)
    
    bot.run(token)
