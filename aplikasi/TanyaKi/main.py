"""
Flask API - Chatbot FMIPA UHO
RAG dengan ChromaDB + Groq + Memory/History + Filter Topik + Filter Format Teknis
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load variabel dari file .env
load_dotenv()

app = Flask(__name__)
CORS(app)

# ==========================================
# GLOBAL VARIABLES
# ==========================================

rag_chain = None
llm_only_chain = None
retriever = None
chat_histories = {}  # { session_id: [HumanMessage, AIMessage, ...] }


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def is_followup_formatting(question: str, history: list) -> bool:
    """Deteksi apakah pertanyaan hanya minta format ulang jawaban sebelumnya."""
    formatting_keywords = [
        "tabel", "table", "poin", "bullet", "list", "nomor", "numbering",
        "format", "rapikan", "ulangi", "tampilkan", "sajikan", "ringkas",
        "singkat", "lebih jelas", "bisa diperinci", "perinci"
    ]
    q = question.lower()
    return (
        len(history) > 0 and
        len(question) < 80 and
        any(kw in q for kw in formatting_keywords)
    )


def is_format_request(question: str) -> bool:
    """
    Deteksi permintaan format output teknis (HTML, CSS, JSON, dll).
    Dicek terpisah agar tidak bentrok dengan allowed_keywords di is_out_of_topic.
    """
    format_patterns = [
        # HTML
        "tampilkan dengan html", "tampilkan menggunakan html",
        "tampilkan pakai html", "buat dalam html", "gunakan html",
        "pakai html", "format html", "dalam html", "ke html",
        "jadikan html", "ubah ke html", "convert ke html",
        # CSS
        "dengan css", "menggunakan css", "pakai css", "tambahkan css",
        "beri style", "styling dengan", "kasih style",
        # JavaScript
        "dengan javascript", "pakai javascript", "gunakan javascript",
        "dalam javascript", "pakai js", "gunakan js",
        # JSON / XML / Markdown
        "dalam format json", "output json", "format json",
        "dalam format xml", "output xml", "format xml",
        "dalam format markdown", "pakai markdown",
        # Render / kode
        "render dengan", "tampilkan sebagai kode", "dalam bentuk kode",
        "buatkan tampilan", "buatkan ui", "buatkan interface",
    ]
    q = question.lower()
    return any(p in q for p in format_patterns)


def is_out_of_topic(question: str) -> bool:
    """
    Deteksi apakah pertanyaan di luar topik FMIPA UHO.
    Mengembalikan True jika pertanyaan tidak relevan dengan FMIPA/akademik.
    """

    # -------------------------------------------------------
    # Kata kunci yang DIIZINKAN (konteks akademik FMIPA UHO)
    # Jika ada salah satu dari ini, pertanyaan tetap diproses
    # -------------------------------------------------------
    allowed_keywords = [
        # Identitas institusi
        "fmipa", "uho", "halu oleo", "universitas halu oleo", "kendari",
        "fakultas matematika", "ilmu pengetahuan alam",
        # Akademik umum
        "jurusan", "prodi", "program studi", "kuliah", "mahasiswa", "dosen",
        "akademik", "skripsi", "thesis", "tesis", "wisuda", "krs", "ipk",
        "semester", "mata kuliah", "jadwal kuliah", "ujian", "nilai",
        "pendaftaran", "registrasi", "beasiswa", "tugas akhir", "pkl",
        "kerja praktik", "magang", "laboratorium", "praktikum",
        "dekan", "kaprodi", "ketua jurusan", "ketua prodi", "staff",
        "administrasi", "gedung", "kampus", "fasilitas", "perpustakaan",
        # Jurusan di FMIPA
        "matematika", "fisika", "kimia", "biologi", "farmasi",
        "informatika", "ilmu komputer", "teknik informatika", "ilkom",
        "statistika", "geofisika", "ilmu lingkungan",
    ]

    # -------------------------------------------------------
    # Kata kunci yang DILARANG (di luar topik)
    # -------------------------------------------------------
    blocked_keywords = [
        # Pemrograman / coding
        "codingan", "buatkan kode", "buatkan program", "buatkan script",
        "buatkan fungsi", "buatkan class", "tolong codingkan",
        "debug kode", "perbaiki kode", "error kode", "kode error",
        "source code", "snippet", "tutorial coding", "belajar coding",
        # Bahasa pemrograman spesifik (tanpa konteks akademik)
        "python script", "javascript code", "php code", "java code",
        "html code", "css code", "sql query", "flutter code",
        "laravel code", "react code", "nodejs code", "vue code",
        "angular code", "django code", "spring boot", "golang code",
        # Tools developer
        "github", "gitlab", "git commit", "git push", "pull request",
        "docker", "kubernetes", "deploy aplikasi", "hosting",
        "database mysql", "database postgresql", "mongodb",
        # Tugas / konten umum
        "buatkan essay", "buatkan makalah", "buatkan artikel",
        "buatkan cerita", "buatkan puisi", "buatkan lagu",
        "buatkan cerpen", "buatkan novel", "terjemahkan",
        "translate ke", "translate this",
        # Hiburan & lifestyle
        "rekomendasi film", "rekomendasi anime", "rekomendasi drama",
        "rekomendasi lagu", "rekomendasi game", "rekomendasi buku",
        "carikan lagu", "lirik lagu", "chord gitar",
        "resep masakan", "resep kue", "cara memasak",
        # Keuangan & investasi
        "saham", "kripto", "bitcoin", "ethereum", "forex",
        "investasi", "trading", "reksa dana",
        # Berita & politik
        "berita hari ini", "berita terkini", "berita politik",
        "pilkada", "pilpres", "pemilu", "partai politik",
        # Olahraga umum
        "jadwal bola", "hasil pertandingan", "liga champion",
        "skor pertandingan",
        # Cuaca
        "cuaca hari ini", "prakiraan cuaca", "ramalan cuaca",
        # Jailbreak / manipulasi AI
        "ignore previous", "ignore instruksi", "abaikan instruksi",
        "jailbreak", "bypass", "prompt injection",
        "pura-pura kamu", "anggap kamu bukan", "roleplay sebagai",
        "jadi karakter", "lupakan aturan", "act as", "pretend you are",
        "you are now", "dari sekarang kamu adalah",
        # Konten dewasa / berbahaya
        "konten dewasa", "konten 18+", "pornografi",
        "cara membuat bom", "cara membuat senjata", "cara hack",
        "cara meretas", "cara membobol",
    ]

    q = question.lower()

    # Jika ada konteks akademik FMIPA → izinkan
    if any(kw in q for kw in allowed_keywords):
        return False

    # Jika ada kata kunci terlarang → tolak
    if any(kw in q for kw in blocked_keywords):
        return True

    return False


# Pesan penolakan
REJECTION_TOPIC_MESSAGE = (
    "Maaf, saya hanya dapat membantu pertanyaan seputar "
    "FMIPA Universitas Halu Oleo — seperti informasi jurusan, "
    "program studi, akademik, dosen, mahasiswa, dan layanan fakultas. 😊\n\n"
    "Silakan ajukan pertanyaan yang berkaitan dengan FMIPA UHO."
)

REJECTION_FORMAT_MESSAGE = (
    "Maaf, saya tidak dapat memformat jawaban dalam bentuk HTML, CSS, "
    "JavaScript, JSON, atau format teknis lainnya. 😊\n\n"
    "Saya hanya menyampaikan informasi seputar FMIPA UHO dalam bentuk teks biasa. "
    "Silakan ajukan pertanyaan Anda tanpa meminta format teknis."
)


# ==========================================
# INISIALISASI RAG ENGINE
# ==========================================

def init_rag():
    """Inisialisasi semua komponen RAG saat server start."""
    global rag_chain, llm_only_chain, retriever

    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    # Pastikan API Key tersedia
    if "GROQ_API_KEY" not in os.environ:
        raise ValueError("GROQ_API_KEY belum diset! Tambahkan ke file .env: GROQ_API_KEY=gsk_...")

    VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "data/chroma_db")

    print("⏳ Memuat model embedding...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("⏳ Memuat Vector Database...")
    vector_db = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings_model
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    print("⏳ Inisialisasi Groq LLM...")
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    SYSTEM_BASE = """Kamu adalah asisten AI resmi untuk Fakultas Matematika dan Ilmu Pengetahuan Alam (FMIPA) Universitas Halu Oleo.

ATURAN WAJIB yang harus selalu diikuti:
1. Hanya jawab pertanyaan seputar FMIPA UHO (jurusan, akademik, dosen, mahasiswa, fasilitas, dll).
2. Jika informasi tidak tersedia di konteks, katakan dengan sopan bahwa kamu belum memiliki informasi tersebut.
3. DILARANG KERAS menghasilkan kode HTML, CSS, JavaScript, JSON, XML, atau format teknis apapun.
4. DILARANG mengarang informasi (halusinasi).
5. Jika diminta format teknis atau topik di luar FMIPA UHO, tolak dengan sopan.
6. Selalu gunakan bahasa Indonesia yang ramah, jelas, dan informatif.
7. Jawab HANYA dalam bentuk teks biasa."""

    # ---- RAG Chain (dengan retrieve dokumen) ----
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_BASE + "\n\nKonteks:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # ---- LLM Only Chain (tanpa retrieve, untuk follow-up formatting) ----
    llm_only_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_BASE + "\n\nJawab berdasarkan riwayat percakapan yang sudah ada. Jangan menambahkan informasi baru."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    llm_only_chain = (
        {
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        | llm_only_prompt
        | llm
        | StrOutputParser()
    )

    print("✅ RAG Engine siap digunakan!")


# ==========================================
# ENDPOINTS
# ==========================================

@app.route("/", methods=["GET"])
def index():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "Chatbot FMIPA UHO API",
        "version": "2.2.0",
        "endpoints": {
            "POST /api/chat": "Kirim pertanyaan ke chatbot",
            "POST /api/chat/reset": "Reset riwayat percakapan",
            "GET /api/health": "Cek status server",
            "GET /api/info": "Info API"
        }
    })


@app.route("/api/health", methods=["GET"])
def health():
    """Cek apakah RAG engine sudah siap."""
    is_ready = rag_chain is not None
    return jsonify({
        "status": "ready" if is_ready else "initializing",
        "rag_loaded": is_ready
    }), 200 if is_ready else 503


@app.route("/api/info", methods=["GET"])
def info():
    """Info tentang chatbot ini."""
    return jsonify({
        "name": "Chatbot FMIPA UHO",
        "description": "Asisten AI berbasis RAG untuk FMIPA Universitas Halu Oleo",
        "model_llm": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "vector_db": "ChromaDB",
        "teknologi": ["LangChain", "ChromaDB", "HuggingFace", "Groq"],
        "fitur": ["RAG", "Memory/History", "Follow-up Detection", "Topic Filter", "Format Filter"]
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Endpoint utama chatbot.

    Request body (JSON):
        {
            "question": "Pertanyaan kamu di sini",
            "session_id": "user-123"  (opsional, default: "default")
        }

    Response (JSON):
        {
            "status": "success" | "out_of_topic" | "error",
            "question": "...",
            "answer": "...",
            "sources": [...],
            "session_id": "...",
            "mode": "rag" | "history" | "rejected"
        }
    """
    if rag_chain is None:
        return jsonify({
            "status": "error",
            "message": "RAG engine belum siap. Tunggu sebentar dan coba lagi."
        }), 503

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({
            "status": "error",
            "message": "Request harus mengandung field 'question'."
        }), 400

    question = data["question"].strip()
    if not question:
        return jsonify({
            "status": "error",
            "message": "Pertanyaan tidak boleh kosong."
        }), 400

    if len(question) > 1000:
        return jsonify({
            "status": "error",
            "message": "Pertanyaan terlalu panjang (maksimal 1000 karakter)."
        }), 400

    # Ambil atau buat session
    session_id = data.get("session_id", "default")
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    history = chat_histories[session_id]

    # -------------------------------------------------------
    # LAPISAN FILTER 1: Cek permintaan format teknis (HTML/CSS/JSON dll)
    # Dicek SEBELUM allowed_keywords agar "info ilkom pakai html" tetap ditolak
    # -------------------------------------------------------
    if is_format_request(question):
        return jsonify({
            "status": "out_of_topic",
            "question": question,
            "answer": REJECTION_FORMAT_MESSAGE,
            "sources": [],
            "session_id": session_id,
            "mode": "rejected"
        }), 200

    # -------------------------------------------------------
    # LAPISAN FILTER 2: Cek topik di luar FMIPA UHO
    # -------------------------------------------------------
    if is_out_of_topic(question):
        return jsonify({
            "status": "out_of_topic",
            "question": question,
            "answer": REJECTION_TOPIC_MESSAGE,
            "sources": [],
            "session_id": session_id,
            "mode": "rejected"
        }), 200

    try:
        from langchain_core.messages import HumanMessage, AIMessage

        # Deteksi apakah hanya minta format ulang jawaban sebelumnya
        if is_followup_formatting(question, history):
            answer = llm_only_chain.invoke({
                "question": question,
                "chat_history": history
            })
            sources = []
            mode = "history"
        else:
            # Retrieve dokumen dari VectorDB
            answer = rag_chain.invoke({
                "question": question,
                "chat_history": history
            })
            source_docs = retriever.invoke(question)
            sources = list({doc.metadata.get("source", "Unknown") for doc in source_docs})
            mode = "rag"

        # Simpan ke history
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))

        # Batasi history maksimal 10 pesan terakhir (5 pasang Q&A)
        if len(history) > 10:
            chat_histories[session_id] = history[-10:]

        return jsonify({
            "status": "success",
            "question": question,
            "answer": answer,
            "sources": sources,
            "session_id": session_id,
            "mode": mode
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Terjadi kesalahan pada server: {str(e)}"
        }), 500


@app.route("/api/chat/reset", methods=["POST"])
def reset_chat():
    """
    Reset riwayat percakapan untuk session tertentu.

    Request body (JSON):
        { "session_id": "user-123" }
    """
    data = request.get_json() or {}
    session_id = data.get("session_id", "default")

    if session_id in chat_histories:
        del chat_histories[session_id]
        return jsonify({
            "status": "success",
            "message": f"Riwayat chat untuk session '{session_id}' telah dihapus."
        })
    else:
        return jsonify({
            "status": "success",
            "message": "Tidak ada riwayat untuk session tersebut."
        })


@app.route("/api/chat", methods=["OPTIONS"])
def chat_options():
    """Handle CORS preflight."""
    return "", 204


# ==========================================
# ERROR HANDLERS
# ==========================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "message": "Endpoint tidak ditemukan."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"status": "error", "message": "Method tidak diizinkan."}), 405


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"status": "error", "message": "Internal server error."}), 500


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Memulai Chatbot FMIPA UHO API...")
    print("=" * 50)

    try:
        init_rag()
    except Exception as e:
        print(f"⚠️  Gagal inisialisasi RAG: {e}")
        print("Server tetap berjalan, tapi /api/chat tidak akan berfungsi.")
        print("Pastikan GROQ_API_KEY dan VECTOR_DB_DIR sudah diset di file .env")

    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    print(f"\n✅ Server berjalan di http://localhost:{port}")
    print("Tekan CTRL+C untuk berhenti.\n")

    app.run(host="0.0.0.0", port=port, debug=debug)