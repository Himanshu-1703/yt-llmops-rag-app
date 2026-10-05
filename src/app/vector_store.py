from uuid import uuid4

from pypdf import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app.clients import (
    CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, EMBEDDING_MODEL, K,
    MAX_FILES, MAX_PAGES, SEARCH_TYPE, TOKENIZER_ENCODING, logger,
)

embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)


def validate_pdfs(files: list) -> str | None:
    # returns an error message if the upload breaks the limits, else None
    if len(files) > MAX_FILES:
        return f"Please upload at most {MAX_FILES} PDFs (you uploaded {len(files)})."

    for file in files:
        num_pages = len(PdfReader(file).pages)
        if num_pages > MAX_PAGES:
            return (f"{file.name} has {num_pages} pages, "
                    f"the limit is {MAX_PAGES} pages per PDF.")

    return None


def load_pdfs(files: list) -> list[Document]:
    docs = []
    for file in files:
        reader = PdfReader(file)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text.strip():
                docs.append(Document(page_content=text,
                                     metadata={"source": file.name, "page": page_number}))
    return docs


# build a fresh vector store from the uploaded PDFs
def build_vector_store(files: list) -> Chroma:

    # chunk the text, sized in tokens
    chunker = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TOKENIZER_ENCODING,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP)

    chunks = chunker.split_documents(load_pdfs(files))
    logger.info(f"Embedding {len(chunks)} chunks from {len(files)} file(s)")

    # in-memory collection, unique per upload so stores never get shared
    vs = Chroma(collection_name=f"{COLLECTION_NAME}_{uuid4().hex}",
                embedding_function=embedder)
    vs.add_documents(chunks)

    return vs


def get_retriever(vs: Chroma):
    return vs.as_retriever(search_type=SEARCH_TYPE,
                           search_kwargs={"k":K})
