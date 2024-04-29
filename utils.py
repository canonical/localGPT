import os
import csv
from datetime import datetime
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from constants import EMBEDDING_MODEL_NAME, OLLAMA_URL, MODEL_ID, CHROMA_SETTINGS, PERSIST_DIRECTORY

def log_to_csv(question, answer):

    log_dir, log_file = "local_chat_history", "qa_log.csv"
    # Ensure log directory exists, create if not
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Construct the full file path
    log_path = os.path.join(log_dir, log_file)

    # Check if file exists, if not create and write headers
    if not os.path.isfile(log_path):
        with open(log_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "question", "answer"])

    # Append the log entry
    with open(log_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, question, answer])


def get_embeddings():
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        embed_instruction="Represent the document for retrieval:",
        query_instruction="Represent the question for retrieving supporting documents:",
        base_url=OLLAMA_URL,
    )

def get_llm():
    return Ollama(
        model=MODEL_ID,
        base_url=OLLAMA_URL
        )

def get_db():
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=get_embeddings(),
        client_settings=CHROMA_SETTINGS,
        )
