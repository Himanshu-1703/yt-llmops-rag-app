from logging import getLogger, StreamHandler, Formatter, INFO

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# load the api keys
load_dotenv()

# application parameters (moved to params.yaml in the experimentation step)
LLM = "gpt-5-mini"
EMBEDDING_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOKENIZER_ENCODING = "o200k_base"
COLLECTION_NAME = "rag_demo"
SEARCH_TYPE = "similarity"
K = 4
MAX_FILES = 2
MAX_PAGES = 25

# create the logger
logger = getLogger(name="Rag App")
handler = StreamHandler()
logger.addHandler(handler)
logger.setLevel(INFO)
logger.propagate = False
formatter = Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(fmt=formatter)

# shared llm client -- used by the generation step of the RAG graph
llm = ChatOpenAI(model=LLM, temperature=0)
