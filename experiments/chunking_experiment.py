import os
import sys
import json
import csv
import time
import shutil

# ---------------------------------------------------------
# Allow imports from project root
# ---------------------------------------------------------
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from core.vector_store import (
    get_embeddings,
    get_retriever,
    COLLECTION_NAME
)


# =========================================================
# CONFIGURATION
# =========================================================

TRANSCRIPT_PATH = "experiments/transcript.txt"
QUESTIONS_PATH = "experiments/evaluation_questions.json"
RESULTS_DIR = "experiments/results"

TOP_K = 4

# ---------------------------------------------------------
# Experiment 1:
# Chunk size changes while overlap remains fixed at 50
# ---------------------------------------------------------
CHUNK_SIZE_EXPERIMENTS = [
    (200, 50),
    (500, 50),
    (750, 50),
    (1000, 50),
    (1500, 50)
]

# ---------------------------------------------------------
# Experiment 2:
# Overlap changes while chunk size remains fixed at 750
# ---------------------------------------------------------
OVERLAP_EXPERIMENTS = [
    (750, 0),
    (750, 50),
    (750, 100),
    (750, 150),
    (750, 200)
]

# ---------------------------------------------------------
# Experiment 3:
# Combined chunk size + overlap configurations
# ---------------------------------------------------------
COMBINED_EXPERIMENTS = [
    (250, 25),
    (500, 50),
    (500, 100),
    (750, 75),
    (750, 150),
    (1000, 100),
    (1000, 200),
    (1500, 150)
]


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def load_transcript():
    """
    Load the already-generated Whisper transcript.

    We reuse the same transcript for every experiment so that
    transcription differences do not affect the results.
    """

    if not os.path.exists(TRANSCRIPT_PATH):
        raise FileNotFoundError(
            f"Transcript not found at: {TRANSCRIPT_PATH}\n"
            "Create experiments/transcript.txt first."
        )

    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        transcript = f.read().strip()

    if not transcript:
        raise ValueError("Transcript file is empty.")

    print(f"Loaded transcript: {len(transcript)} characters")

    return transcript


def load_questions():
    """
    Load the fixed evaluation question set.
    """

    if not os.path.exists(QUESTIONS_PATH):
        raise FileNotFoundError(
            f"Question file not found at: {QUESTIONS_PATH}"
        )

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} evaluation questions")

    return questions


# =========================================================
# CHUNKING
# =========================================================

def create_chunks(transcript, chunk_size, chunk_overlap):
    """
    Split transcript using the same splitter as the main project.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_text(transcript)

    return chunks


# =========================================================
# VECTOR STORE
# =========================================================

def build_experiment_vector_store(
    chunks,
    chunk_size,
    chunk_overlap
):
    """
    Build a temporary Chroma vector store for one experiment.

    A separate directory is used for every configuration.
    """

    db_dir = (
        f"experiments/vector_db_"
        f"{chunk_size}_{chunk_overlap}"
    )

    # Remove previous experiment DB
    if os.path.exists(db_dir):
        try:
            shutil.rmtree(db_dir)
        except Exception as e:
            print(
                f"Warning: could not completely remove "
                f"{db_dir}: {e}"
            )

    documents = [
        Document(
            page_content=chunk,
            metadata={
                "chunk_index": i,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
        )
        for i, chunk in enumerate(chunks)
    ]

    # Use the SAME embedding model as the main project
    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=db_dir
    )

    return vector_store


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_documents(retriever, question):
    """
    Retrieve top-k documents for a question and measure
    retrieval latency.
    """

    start = time.perf_counter()

    documents = retriever.invoke(question)

    end = time.perf_counter()

    retrieval_time = end - start

    return documents, retrieval_time


# =========================================================
# RELEVANCE ESTIMATION
# =========================================================

def normalize_text(text):
    """
    Normalize text for simple concept matching.
    """

    return (
        text.lower()
        .replace("-", " ")
        .replace("_", " ")
    )


def concept_present(concept, text):
    """
    Check whether an expected concept appears in retrieved
    context.

    This is a simple automatic approximation, NOT a perfect
    semantic relevance judge.
    """

    concept = normalize_text(concept)
    text = normalize_text(text)

    return concept in text


def calculate_hit_at_k(documents, expected_concepts):
    """
    Hit@K = 1 if at least one expected concept occurs
    in the retrieved top-k context.

    Otherwise Hit@K = 0.
    """

    retrieved_text = "\n".join(
        doc.page_content for doc in documents
    )

    for concept in expected_concepts:
        if concept_present(concept, retrieved_text):
            return 1

    return 0


def calculate_mrr(documents, expected_concepts):
    """
    Mean Reciprocal Rank for a single question.

    Finds the rank of the first retrieved chunk containing
    any expected concept.

    Example:

        relevant chunk at rank 1 -> 1.0
        relevant chunk at rank 2 -> 0.5
        relevant chunk at rank 3 -> 0.333
        no relevant chunk        -> 0
    """

    for rank, doc in enumerate(documents, start=1):

        text = doc.page_content

        for concept in expected_concepts:

            if concept_present(concept, text):
                return 1.0 / rank

    return 0.0


# =========================================================
# SINGLE CONFIGURATION EXPERIMENT
# =========================================================

def run_single_configuration(
    transcript,
    questions,
    chunk_size,
    chunk_overlap,
    experiment_type
):
    """
    Run retrieval evaluation for one chunk configuration.
    """

    print("\n" + "=" * 70)

    print(
        f"Experiment: {experiment_type} | "
        f"Chunk Size: {chunk_size} | "
        f"Overlap: {chunk_overlap}"
    )

    print("=" * 70)

    # -----------------------------------------------------
    # 1. Chunking
    # -----------------------------------------------------

    chunk_start = time.perf_counter()

    chunks = create_chunks(
        transcript,
        chunk_size,
        chunk_overlap
    )

    chunk_end = time.perf_counter()

    chunking_time = chunk_end - chunk_start

    number_of_chunks = len(chunks)

    average_chunk_length = (
        sum(len(chunk) for chunk in chunks)
        / number_of_chunks
    )

    print(f"Number of chunks: {number_of_chunks}")

    print(
        f"Average chunk length: "
        f"{average_chunk_length:.2f} characters"
    )

    # -----------------------------------------------------
    # 2. Build vector database
    # -----------------------------------------------------

    embedding_start = time.perf_counter()

    vector_store = build_experiment_vector_store(
        chunks,
        chunk_size,
        chunk_overlap
    )

    embedding_end = time.perf_counter()

    embedding_time = embedding_end - embedding_start

    print(
        f"Embedding/indexing time: "
        f"{embedding_time:.2f} seconds"
    )

    # -----------------------------------------------------
    # 3. Create retriever
    # -----------------------------------------------------

    retriever = get_retriever(
        vector_store,
        k=TOP_K
    )

    # -----------------------------------------------------
    # 4. Evaluate retrieval
    # -----------------------------------------------------

    total_hit_at_k = 0
    total_mrr = 0
    total_retrieval_time = 0

    question_results = []

    for question_data in questions:

        question_id = question_data["id"]
        question = question_data["question"]

        expected_concepts = question_data[
            "expected_answer_concepts"
        ]

        print(
            f"\n{question_id}: {question}"
        )

        documents, retrieval_time = retrieve_documents(
            retriever,
            question
        )

        hit_at_k = calculate_hit_at_k(
            documents,
            expected_concepts
        )

        mrr = calculate_mrr(
            documents,
            expected_concepts
        )

        total_hit_at_k += hit_at_k
        total_mrr += mrr
        total_retrieval_time += retrieval_time

        question_results.append({
            "question_id": question_id,
            "question": question,
            "hit_at_k": hit_at_k,
            "mrr": round(mrr, 4),
            "retrieval_time_s": round(
                retrieval_time,
                6
            )
        })

        print(
            f"  Hit@{TOP_K}: {hit_at_k}"
        )

        print(
            f"  MRR: {mrr:.4f}"
        )

        print(
            f"  Retrieval time: "
            f"{retrieval_time:.6f}s"
        )

    # -----------------------------------------------------
    # 5. Aggregate metrics
    # -----------------------------------------------------

    num_questions = len(questions)

    hit_at_k = total_hit_at_k / num_questions

    mean_reciprocal_rank = (
        total_mrr / num_questions
    )

    average_retrieval_time = (
        total_retrieval_time / num_questions
    )

    # -----------------------------------------------------
    # 6. Store result
    # -----------------------------------------------------

    result = {
        "experiment_type": experiment_type,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_chunks": number_of_chunks,
        "avg_chunk_length": round(
            average_chunk_length,
            2
        ),
        "chunking_time_s": round(
            chunking_time,
            4
        ),
        "embedding_indexing_time_s": round(
            embedding_time,
            4
        ),
        "avg_retrieval_time_s": round(
            average_retrieval_time,
            6
        ),
        f"hit_at_{TOP_K}": round(
            hit_at_k,
            4
        ),
        "mrr": round(
            mean_reciprocal_rank,
            4
        )
    }

    print("\nRESULT")

    print(
        f"Hit@{TOP_K}: "
        f"{hit_at_k:.4f}"
    )

    print(
        f"MRR: "
        f"{mean_reciprocal_rank:.4f}"
    )

    print(
        f"Average retrieval time: "
        f"{average_retrieval_time:.6f}s"
    )

    return result, question_results


# =========================================================
# SAVE RESULTS
# =========================================================

def save_csv(filename, rows):
    """
    Save experiment results to CSV.
    """

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    path = os.path.join(
        RESULTS_DIR,
        filename
    )

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {path}")


# =========================================================
# MAIN EXPERIMENT
# =========================================================

def run_experiment():

    print("\n")
    print("=" * 70)
    print("AI VIDEO ASSISTANT")
    print("Chunk Size & Overlap Retrieval Experiment")
    print("=" * 70)

    # -----------------------------------------------------
    # Load fixed transcript
    # -----------------------------------------------------

    transcript = load_transcript()

    # -----------------------------------------------------
    # Load fixed evaluation questions
    # -----------------------------------------------------

    questions = load_questions()

    all_results = []

    all_question_results = []

    # =====================================================
    # EXPERIMENT 1: CHUNK SIZE
    # =====================================================

    print("\n\n")
    print("#" * 70)
    print("EXPERIMENT 1: CHUNK SIZE")
    print("#" * 70)

    for chunk_size, overlap in CHUNK_SIZE_EXPERIMENTS:

        result, question_results = (
            run_single_configuration(
                transcript=transcript,
                questions=questions,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                experiment_type="chunk_size"
            )
        )

        all_results.append(result)

        for qr in question_results:
            qr.update({
                "experiment_type": "chunk_size",
                "chunk_size": chunk_size,
                "chunk_overlap": overlap
            })

        all_question_results.extend(
            question_results
        )

    # =====================================================
    # EXPERIMENT 2: OVERLAP
    # =====================================================

    print("\n\n")
    print("#" * 70)
    print("EXPERIMENT 2: CHUNK OVERLAP")
    print("#" * 70)

    for chunk_size, overlap in OVERLAP_EXPERIMENTS:

        result, question_results = (
            run_single_configuration(
                transcript=transcript,
                questions=questions,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                experiment_type="overlap"
            )
        )

        all_results.append(result)

        for qr in question_results:
            qr.update({
                "experiment_type": "overlap",
                "chunk_size": chunk_size,
                "chunk_overlap": overlap
            })

        all_question_results.extend(
            question_results
        )

    # =====================================================
    # EXPERIMENT 3: COMBINED
    # =====================================================

    print("\n\n")
    print("#" * 70)
    print("EXPERIMENT 3: COMBINED CHUNK SIZE + OVERLAP")
    print("#" * 70)

    for chunk_size, overlap in COMBINED_EXPERIMENTS:

        result, question_results = (
            run_single_configuration(
                transcript=transcript,
                questions=questions,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                experiment_type="combined"
            )
        )

        all_results.append(result)

        for qr in question_results:
            qr.update({
                "experiment_type": "combined",
                "chunk_size": chunk_size,
                "chunk_overlap": overlap
            })

        all_question_results.extend(
            question_results
        )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    chunk_size_results = [
        r for r in all_results
        if r["experiment_type"] == "chunk_size"
    ]

    overlap_results = [
        r for r in all_results
        if r["experiment_type"] == "overlap"
    ]

    combined_results = [
        r for r in all_results
        if r["experiment_type"] == "combined"
    ]

    save_csv(
        "chunk_size_results.csv",
        chunk_size_results
    )

    save_csv(
        "overlap_results.csv",
        overlap_results
    )

    save_csv(
        "combined_results.csv",
        combined_results
    )

    save_csv(
        "all_question_results.csv",
        all_question_results
    )

    save_csv(
        "all_results.csv",
        all_results
    )

    # =====================================================
    # PRINT SUMMARY
    # =====================================================

    print("\n\n")
    print("=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)

    print(
        f"\nTotal configurations tested: "
        f"{len(all_results)}"
    )

    print("\nResults:")

    print(
        f"\n{'Type':<12}"
        f"{'Chunk':<10}"
        f"{'Overlap':<10}"
        f"{'Chunks':<10}"
        f"{'Hit@4':<10}"
        f"{'MRR':<10}"
    )

    print("-" * 62)

    for result in all_results:

        print(
            f"{result['experiment_type']:<12}"
            f"{result['chunk_size']:<10}"
            f"{result['chunk_overlap']:<10}"
            f"{result['num_chunks']:<10}"
            f"{result['hit_at_4']:<10}"
            f"{result['mrr']:<10}"
        )

    print("\nAll results saved to:")
    print(RESULTS_DIR)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    run_experiment()
