from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph

from app.clients import llm

system_prompt = """You are a helpful assistant that answers questions about the user's PDF documents.
Answer only from the context provided. If the answer is not in the context, say that you could not find it in the uploaded documents.
Keep the answer clear and concise."""


class RAGState(TypedDict):

    query: str
    retriever: BaseRetriever
    retrieved_docs: list[Document]
    context: str
    prompt: ChatPromptTemplate
    response: str


def retrieve(state: RAGState) -> dict:
    query = state["query"]
    retriever = state["retriever"]
    retrieved_docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return {"retrieved_docs":retrieved_docs,
            "context":context}



def augmentation(state: RAGState) -> dict:

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "context: {context}\n\nquery: {query}")
    ])

    return {"prompt": prompt}



def generation(state: RAGState) -> dict:

    query = state["query"]
    context = state["context"]
    prompt = state["prompt"]

    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context":context, "query":query})

    return {"response":response}



# create the graph
graph_builder = StateGraph(RAGState)

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("augmentation", augmentation)
graph_builder.add_node("generation", generation)

graph_builder.add_edge(START,"retrieve")
graph_builder.add_edge("retrieve","augmentation")
graph_builder.add_edge("augmentation","generation")
graph_builder.add_edge("generation",END)

graph = graph_builder.compile()
