# rag_app.py
import os
import translator

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough




# ---------- CONFIG ----------
PERSIST_DIR = "chroma_db"
OLLAMA_MODEL = "llama3.1:8b"  # change if you pulled another model


# ---------- 1. LOAD CHROMA DB ----------
def load_retriever():
    if not os.path.exists(PERSIST_DIR):
        raise FileNotFoundError(
            f"Chroma DB not found at {PERSIST_DIR}. Please run 'create_db.py' first."
        )

    print("Loading existing Chroma DB...")
    # 2. EMBEDDINGS (must match what was used for creation)
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)

    # 3. CHROMA VECTORSTORE
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    # 4. RETRIEVER
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    print("Chroma DB loaded.")
    return retriever


# ---------- 5. LLM ----------
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.0)

# ---------- 6. RAG CHAIN UTILS & SETUP ----------
template = """Use ONLY the following context to answer the question.

Context:
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    try:
        retriever = load_retriever()
    except FileNotFoundError as e:
        print(e)
        exit()

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    # ---------- 7. TEST LOOP ----------
    print("\nRAG ჩართულია! დაუსვით შეკითხვა მოცემულ PDF-ზე (გამოსვლა: 'quit/exit')\n")

    while True:
        lang = input("\nაირჩიეთ ენა: ka/en\n").strip().lower()
        if lang in ("ka", "en"):
            break
        print("გთხოვთ შეიყვანოთ მხოლოდ ka ან en")

    while True:
        question = input("კითხვა: ").strip()
        if lang == 'ka':
            translatedQuestion = translator.translate(question, 'en')
        else:
            translatedQuestion = question
        if question.lower() in {"quit", "exit", ""}:
            break
        if not question:
            continue

        # Invoke the RAG chain
        answer = rag_chain.invoke(translatedQuestion)

        if lang == 'ka':
            translatedAnswer = translator.translate(answer, 'ka')
        else:
            translatedAnswer = answer

        print(f"\nanswer: {answer}\n")

        print(f"\nპასუხი: {translatedAnswer}\n")