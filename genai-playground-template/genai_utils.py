import logging
from typing import Dict, Optional

from langchain_core.embeddings import Embeddings
from langchain.schema.document import Document

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader

from gen_ai_hub.proxy.core.base import BaseProxyClient
from gen_ai_hub.proxy.core import set_proxy_version
from gen_ai_hub.proxy.langchain import init_llm, init_embedding_model

from hana_ml import ConnectionContext
from hdbcli.dbapi import Connection


     
        
def get_llm_model(genai_hub: BaseProxyClient, model_info: Dict, temperature:Optional[float]=0.0, top_p:Optional[float]=0.95, max_tokens: Optional[int]=3500, do_streaming: Optional[bool]=False)->any:
    """ Serve the required LLM chat model """ 
    active_proxy = None
    model_name = None
    deployment_id = None
    if model_info["platform"] == "SAP Generative AI Hub":
        active_proxy = genai_hub
        set_proxy_version("gen-ai-hub")
        model_name = model_info["model"]
    # elif model_info["platform"] == "SAP BTP Proxy":
    #     active_proxy = BTP_PROXY_CLIENT
    #     deployment_id = model_name = model_info["model"]
    # elif model_info["platform"] == "Ollama on SAP AI Core":
    #     headers = {
    #         'Authorization': AI_CORE_CONNECTION.get_token(),
    #         'AI-Resource-Group': AI_CORE_CONNECTION.get_resource_group()
    #     }
    #     if not temperature:
    #         temperature = 0.1
    #     if not top_p:
    #         top_p = 0.6
    #     chat_ollama_kwargs = {
    #         "base_url": AI_CORE_CONNECTION.get_baseurl(),
    #         "headers": headers,
    #         "model": model_info["model"],
    #         "temperature": temperature,
    #         "top_p": top_p
    #     }
    #     return ChatOllama(**chat_ollama_kwargs)
    init_llm_kwargs = {
        "model_name": model_name,
        "deployment_id": deployment_id,
        "proxy_client": active_proxy,
        "stream": do_streaming
    }
    if temperature is not None:
        init_llm_kwargs["temperature"] = temperature
    if top_p is not None:
        init_llm_kwargs["top_p"] = top_p
    if max_tokens is not None:
        init_llm_kwargs["max_tokens"] = max_tokens
    return init_llm(**init_llm_kwargs)

def get_embedding_model(genai_hub: BaseProxyClient, embedding_model_name: str)->Embeddings:
    init_embedding_kwargs = {
        "model_name": embedding_model_name,
        "proxy_client": genai_hub
    }
    return init_embedding_model(**init_embedding_kwargs)

def get_hana_connection(conn_params: dict)->Connection:
    """ Connect to HANA Cloud """
    connection = None  
    try:
        conn_context = ConnectionContext(
            address = conn_params["host"],
            port = 443,
            user = conn_params["user"],
            password= conn_params["password"],
            encrypt= True
        )    
        connection = conn_context.connection
        logging.debug(conn_context.hana_version())
        logging.debug(conn_context.get_current_schema())
    except Exception as e:
        logging.error(f'Error when opening connection to HANA Cloud DB with host: {conn_params["host"]}, user {conn_params["user"]}. Error was:\n{e}')
    finally:    
        return connection

def convert_to_text(file: list, filename: str, chunk_size: int, chunk_overlap: int, use_ocr: bool)->list[Document]:
    """ Converts file to text """
    if file.endswith('.pdf'):
        loader = PyPDFLoader(file_path=file, extract_images=use_ocr)
    elif file.endswith('.docx'):
        loader = Docx2txtLoader(file_path=file)
    elif file.endswith('.txt'):
        loader = TextLoader(file_path=file, encoding='utf-8')
    elif file.endswith('.pptx'):
        loader = UnstructuredPowerPointLoader(file_path=file, mode="single", strategy="fast") 
    else:
        raise ValueError('File format not supported. Please provide a .pdf or .docx file.')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                chunk_overlap=chunk_overlap, 
                                                length_function=len, 
                                                is_separator_regex=False
                                                )
    documents = loader.load_and_split(text_splitter)
    documents = sanitize_documents_metadata_inplace(documents)
    for doc in documents:
        if doc.metadata.get("source", None) != None:
            doc.metadata["source"]=filename
        if doc.metadata.get("page", None) != None: # increment by 1 to avoid page 0
            doc.metadata["page"] += 1
    return documents

def format_docs(docs):
    """ Concatenates the result documents after vector db retrieval """
    result = ""
    for doc in docs:
        source = f"Source document '{doc.metadata['source']}'" if 'source' in doc.metadata else ""
        page = f"Page {doc.metadata['page']}" if 'page' in doc.metadata else ""
        result +=  f"Below text is from {source} {page}:\n-------------------------------------------\n{doc.page_content}\n"
    return result

def sanitize_metadata_keys(metadata):
    """ Convert metadata keys to comply with HANA Vector DB naming rules """
    sanitized = {}
    for key, value in metadata.items():
        # Replace spaces with underscores
        clean_key = key.replace(' ','_')
        sanitized[clean_key] = value

    return sanitized

def sanitize_documents_metadata_inplace(documents):
    """Sanitize only metadata field without creating new objects"""
    for doc in documents:
        if hasattr(doc, 'metadata'):
            doc.metadata = sanitize_metadata_keys(doc.metadata)
    return documents