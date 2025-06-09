from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from gradio import Blocks
import os, logging, json, time
from textwrap import dedent

from operator import itemgetter

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.runnable import RunnableSerializable, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores.hanavector import HanaDB

from typing import Dict, List

from gen_ai_hub import GenAIHubProxyClient
from genai_utils import get_hana_connection, get_llm_model, get_embedding_model, convert_to_text, format_docs

from hdbcli import dbapi

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL","all-MiniLM-L12-v2") # This model should be deployed in Generate AI Hub
TABLE_NAME_FOR_DOCUMENTS  = "CML2024WS"
CHUNK_SIZE_MAX = 10000

RAG_TEMPLATE = dedent("""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Be as concise as necessary unless there are no specific requests. Write the sources (document and page or paragraph title if available) at the bottom as footnotes.
    Write the source in this format:
    **Sources:**
    - sourcefile name, page information, paragraph title

    Conversation History: {history}
    Retrieved Context: {context}
    Question: {question}
    Answer:""")

LOGO_MARKDOWN = f"""
### AI Playground Template
![AI Playground Template](/img/cml2024.webp)
"""

BLOCK_CSS = """
gradio-app > .gradio-container {
    max-width: 100% !important;
    
}
.contain { display: flex !important; flex-direction: column !important; }
#chat_window { height: 70vh !important; }
#column_left { height: 88vh !important; }
#sql_col_left1 { height: 82vh !important;}
#arch_gallery { height: 88vh !important;}
#buttons button {
    min-width: min(120px,100%);
}
footer {
    display:none !important
}

/* message box background color (not transparent) */
.toast-body.info,
div[class*="toast-body"][class*="info"] {
    background-color: rgba(217, 237, 247, 0.95) !important;
    backdrop-filter: blur(5px) !important;
    border: 1px solid #bee5eb !important;
    color: #0c5460 !important;
}
.info, [class*="info"] {
    background-color: rgba(217, 237, 247, 0.9) !important;
    color: #0c5460 !important;
}
[class*="svelte-"][class*="info"] {
    background-color: rgba(217, 237, 247, 0.95) !important;
    backdrop-filter: blur(3px) !important;
}
"""

def user(state: dict, user_message: str, history: list)->tuple:
    """ Handle user interaction in the chat window """
    state["skip_llm"] = False
    if len(user_message) <= 0:
        state["skip_llm"] = True
        return "", history

    history = history or []

    if history and history[-1]["role"] == "assistant" and history[-1]["content"]=="":
        history.pop()

    # add user message
    history.append({"role": "user", "content": user_message})

    # add assistant message placeholder
    history.append({"role": "assistant", "content": ""})

    return "", history

def call_llm(state: Dict, history: List[Dict[str, str]], model_name: str, system_role: str, rag_active: bool, k_top: int)->any:
    """ Handle LLM request and response """
    do_stream = True
    if not isinstance(state, dict) or state.get("skip_llm",False)==True:
        return history
    # history[-1]["content"] = ""
    llm = state.get("model")
    model_info = next((item for item in state["models"] if item["name"] == model_name), None)
    if not llm:
        state["model"] = get_llm_model(
            genai_hub=GENAI_HUB,
            model_info=model_info,
            temperature=0.0,
            top_p=0.8,
            do_streaming=do_stream      
        )
        state["memory"]=ConversationBufferMemory(memory_key="history", return_messages=True)
        state["memory"].clear()
    llm_chain = None
    if rag_active:
        rag_template = RAG_TEMPLATE
        chatprompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_role),
                ("human", rag_template),
            ]            
        )
        if not state.get("ef", None):
            state["ef"] = get_embedding_model(genai_hub=GENAI_HUB, embedding_model_name=EMBEDDING_MODEL)
        if not state.get("connection", None):
            state["connection"] = get_hana_connection(conn_params=state["conn_data"])
        if not state.get("vector_db", None):
            state["vector_db"] = HanaDB(embedding=state.get("ef"), connection=state["connection"], table_name=TABLE_NAME_FOR_DOCUMENTS)
        retriever = state["vector_db"].as_retriever(search_kwargs={"k": k_top})
        llm_chain = (
            {
                "history": RunnableLambda(state["memory"].load_memory_variables) | itemgetter("history"),
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | chatprompt
            | state["model"]
            | StrOutputParser()
        )
    else:
        chatprompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_role),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template="{input}"))
            ],
        )
        llm_chain = (
            {
                "history": RunnableLambda(state["memory"].load_memory_variables) | itemgetter("history"), 
                "input": RunnablePassthrough()
            }
            | chatprompt
            | state["model"]
            | StrOutputParser()
        )
    # print(f"history:{history}")
    query = history[-2]['content'] # user role message
    state["memory"].chat_memory.add_user_message(query) # add user question to memory
    if rag_active:
        input = query
    else:
        input = {"query": query}
    if do_stream:
        try:
            for response in llm_chain.stream(input): 
                history[-1]["content"] += response
                logging.debug(history[-1]["content"])
                yield history, state
            state["memory"].chat_memory.add_ai_message(history[-1]["content"]) # once streaming is done, add ai response to memory
        except Exception as e:
            history[-1]["content"] += str(f"😱 Oh no! It seems the LLM has some issues. Error was: {e}.")
            yield history, state
    else:
        try:
            response = llm_chain.invoke(input) 
            history[-1]["content"] += response
            logging.debug(history[-1]["content"])
            state["memory"].chat_memory.add_ai_message(response) # add ai response to memory
        except Exception as e:
            history[-1]["content"] += str(f"😱 Oh no! It seems the LLM has some issues. Error was: {e}.")
    return history, state

def retrieve_data(vector_db: HanaDB, llm: BaseLanguageModel)->RunnableSerializable:
    """ Retrieves data from store and passes back result """
    return

def refresh_sources(state):
    """ Update the source information when refresh the page """
    try:
        if state.get("conn_data"):
            embedded_sources = read_hanavs_sources(conn_params=state["conn_data"])
            return [
                gr.update(value=embedded_sources["sources"], visible=True), # file_output
                gr.update(visible=embedded_sources["active"], value=False), # cbox_use_document
                gr.update(visible=embedded_sources["active"]) # btn_remove
            ]
    except Exception as e:
        logging.error(f"Failed to refresh sources: {e}")

    return [
        gr.update(value="### Avaliable sources\n *No documents uploaded yet*", visible=True),
        gr.update(visible=False, value=False),
        gr.update(visible=False)
    ]
   
    
def embed_file(state, files, chunk_size, chunk_overlap, use_ocr):
    """ Embed text file after conversion """
    documents = []
    if files==None: # Happens when the list is cleared
        return 
    embedding_function = get_embedding_model(genai_hub=GENAI_HUB, embedding_model_name=EMBEDDING_MODEL)
    if not state.get("connection"):
        state["connection"] = get_hana_connection(conn_params=state["conn_data"])
    vector_db = HanaDB(embedding=embedding_function, connection=state["connection"], table_name=TABLE_NAME_FOR_DOCUMENTS)
    # Remove existing embeddings from Hana Cloud table
    try:
        vector_db.delete(filter={})
    except Exception as e:
        logging.info(f"Deleting embedding entries failed with error {e}. Maybe there were no embeddings?")
    try:
        for file in files:
            # logging.error(f"file: {file}")
            filename = os.path.basename(file)
            # logging.error(f"filename: {filename}")
            gr.Info(f"Splitting {filename}.")
            documents = convert_to_text(file=file, filename=filename, chunk_size=chunk_size, chunk_overlap=chunk_overlap,use_ocr=use_ocr)
            if len(documents) == 0:
                gr.Warning("The document didn't contain text. Maybe a PDF in an image-only format? You can try with OCR option.")
                return refresh_sources(state)
            gr.Info(f"Embedding {str(len(documents))} documents. Please wait.")
            start_time = time.time()
            logging.info(f"Embedding {str(len(documents))} documents. Please wait.")
            try:
                # logging.error(f"documents: {documents}")
                vector_db.add_documents(documents=documents)
                msg=f"Embedded {len(documents)} documents in table {TABLE_NAME_FOR_DOCUMENTS}."
                logging.info(msg)
                gr.Info(msg)
            except Exception as e:
                logging.error(f"Adding document embeddings failed with error: {e}.")
            elapsed_time = time.time() - start_time
            gr.Info(f"You can query your document now. Elapsed time: {elapsed_time:.2f}s.")
            logging.info(f"You can query your document now. Elapsed time: {elapsed_time:.2f}s.")
        # re-read embedded source
        embedded_sources = read_hanavs_sources(conn_params=state["conn_data"])
        return [state, gr.update(value=embedded_sources["sources"], visible=True), gr.update(visible=embedded_sources["active"], value=True), gr.update(visible=embedded_sources["active"])]
    except Exception as e:
        gr.Warning(f"Embedding failed. {str(e)}")
        return [state] + list(refresh_sources(state))


def read_hanavs_sources(conn_params: Dict)->List[str]:
    """ Checks and reads from HANA Cloud VS existing embedded documents """
    # Open database connection
    conn = dbapi.connect(address=conn_params["host"], port=443, user=conn_params["user"], password=conn_params["password"])
    # Prepare a cursor object using cursor() method
    cursor = conn.cursor()
    try:
        # SQL query to select the VEC_META field from the defined table
        sql = f'SELECT VEC_META FROM {TABLE_NAME_FOR_DOCUMENTS}'
        # Execute the SQL query
        cursor.execute(sql)
        # Fetch all the rows
        rows = cursor.fetchall()
        # Set to store unique sources
        unique_sources = set()
        # Iterate over the rows and extract the 'source' from the JSON data
        for row in rows:
            # Each row is a tuple, and we're interested in the first element (which is a string in this case)
            json_data_str = row[0]
            # Parse the JSON data
            json_data = json.loads(json_data_str)
            # Add the 'source' value to the set of unique sources
            unique_sources.add(json_data['source'])
    except dbapi.ProgrammingError as e:
        if "invalid table name" in str(e): # once upload the document, table will be created
            unique_sources = set()
        else:
            raise
    # Close the cursor and disconnect from the server
    cursor.close()
    conn.close()
    source_text = "### Available sources\n"
    for source in sorted(unique_sources):  # Sorting to ensure consistent order
        source_text += f"{source}\n\r"
    ret_dict = {
        "sources": source_text,
        "active": True,
        "text": "Use document"
    }
    if len(unique_sources) > 1:
        ret_dict["text"] += "s"
    elif len(unique_sources) == 0:
        ret_dict["active"] = False
        ret_dict["sources"] = None
    return ret_dict

    
def remove_file(state):
    """ Remove embedded document """
    try:
        if not state.get("connection", None):
            state["connection"] = get_hana_connection(conn_params=state["conn_data"])
        if state.get("connection"):
            embedding_function = get_embedding_model(genai_hub=GENAI_HUB, embedding_model_name=EMBEDDING_MODEL)
            vector_db = HanaDB(embedding=embedding_function, connection=state["connection"], table_name=TABLE_NAME_FOR_DOCUMENTS)
            vector_db.delete(filter={}) # delete all documents
            gr.Info("All documents removed successfully.")
    except Exception as e:
        logging.error(f"Failed to remove documents: {e}")
        gr.Warning(f"Failed to remove documents: {str(e)}")

    state["db"] = None
    return [state] + list(refresh_sources(state))   
    
    
def model_change(model_name: str, state: Dict)->any:
    """ Toggle image upload visibility """
    is_visible = False
    result = next((item for item in state["models"] if item["name"] == model_name), None)
    if result["hasImage"]:
        is_visible = True
    try:
        state["model"] = None
    except Exception as e:
        pass # No action needed
    return gr.Textbox(value=result["desc"], label=f"{result['platform']}"), gr.Textbox(autofocus=True), state

    
def clear_data(state: dict)->list:
    """ Clears the history of the chat """
    state_new = {
        "conn_data": state.get("conn_data", None),
        "models": state.get("models", None),
        "ef": state.get("ef", None),
        "vector_db": state.get("vector_db", None),
        "connection": state.get("connection", None),
    }
    return [None, state_new]


def get_system_roles()->List:
    """ Some templates for system roles, user can change them during runtime """
    return [
            "You are a helpful assistant. You provide expert-level explanation.",
            "You are a helpful assistant. You explain topics at a general professional level.",
            "You are a helpful assistant. You explain topics in a very easy way for the general public."
    ]
    
    
def slider_chunk_change(state: dict, chunk_size: float, chunk_overlap: float):
    """ Change the overlap slider range to match the chunk size range """
    max_value = int(chunk_size/2)
    if max_value < chunk_overlap:
        chunk_overlap = max_value
    return gr.Slider(maximum=max_value, value=chunk_overlap)


def build_chat_view(models: Dict, conn_data: Dict, sys_roles: List)->Blocks:
    """ Build the view with Gradio blocks """
    embedded_sources = read_hanavs_sources(conn_params=conn_data)
    with gr.Blocks(
            title="CML Workshop - Retrieve-Augment-Generate in a BTP-contained setting", 
            theme=gr.themes.Soft(),
            css=BLOCK_CSS
        ) as chat_view:
        state = gr.State({})
        state.value["models"] = models
        state.value["conn_data"] = conn_data
        with gr.Row(elem_id="overall_row") as main_screen:
            with gr.Column(scale=10, elem_id="column_left"):
                chatbot = gr.Chatbot(
                    type='messages',
                    label="SAP Generative AI Playground - Chat & RAG",
                    elem_id="chat_window",
                    show_copy_button=True,
                    show_share_button=True,
                    avatar_images=(None, "./img/saplogo.png")
                )
                with gr.Row(elem_id="input_row") as query_row:
                    msg_box = gr.Textbox(
                        scale=9,
                        elem_id="msg_box",
                        show_label=False,
                        max_lines=5,
                        placeholder="Enter text and press ENTER",
                        container=False,
                        autofocus=True                    )
            with gr.Column(scale=3, elem_id="column_right") as column_right:
                model_names = [item["name"] for item in models]
                state.value["models"] = models
                model_selector = gr.Dropdown(
                    choices=model_names, 
                    container=True,
                    label="🗨️ Language Model",
                    show_label=True,
                    interactive=True,
                    value=model_names[0] if len(model_names) > 0 else ""
                )
                model_info_box = gr.Textbox(value=models[0]["desc"], lines=3, label=f"🆗 {models[0]['platform']}", interactive=False, elem_id="model_info")
                clear = gr.Button(value="Clear history")
                with gr.Accordion("🎚️ Model parameters", open=False, elem_id="parameter_accordion") as parameter_accordion:
                    system_role = gr.Dropdown(           
                        allow_custom_value=True,             
                        choices=sys_roles,
                        container=False,
                        label="Select or enter a system role",
                        interactive=True,
                        value=sys_roles[0] if len(sys_roles) > 0 else "",
                    )
                with gr.Accordion("🔍 Document Talk (RAG)", open=embedded_sources["active"], elem_id="rag_accordion") as rag_accordion:
                    uplbtn_file = gr.UploadButton(label="Upload",  file_types=[".docx", ".pptx", ".pdf", ".txt"], file_count="multiple", elem_id="rag_btn_upload")
                    # file_output = gr.File(elem_id="rag_file_view", height=60, value=embedded_sources)
                    file_output = gr.Markdown(value=embedded_sources["sources"], elem_id="filelist_markdown_id", label="Available sources", visible=embedded_sources["active"], rtl=True)
                    cbox_use_document = gr.Checkbox(label=embedded_sources["text"], visible=embedded_sources["active"], value=False, elem_id="cb_rag_active")
                    btn_remove = gr.Button(value="Remove", visible=False, elem_id="rag_btn_remove")
                    with gr.Accordion("🎚️ Embedding Options", open=False, elem_id="embedding_options") as embedding_accordion:
                        sldr_chunk_size = gr.Slider(minimum=100, maximum=CHUNK_SIZE_MAX, value=1000, step=100, label="Chunk Size")
                        sldr_chunk_overlap = gr.Slider(minimum=0, maximum=500, value=100, step=100, label="Overlap")
                        cbox_use_ocr = gr.Checkbox(label="⚠️ Use Image OCR", visible=True, value=False, elem_id="cb_image_ocr")
                        sldr_k_top = gr.Slider(minimum=1, maximum=20, value=8, step=1, label="k-top from Vector Store")
                cml2024img = gr.Image(value='./img/cml2024.webp', elem_id="cml2024_box", show_label=False)
                
        # page reload
        chat_view.load(
            refresh_sources,
            inputs=[state],
            outputs=[file_output, cbox_use_document, btn_remove]
        )

        model_selector.change(model_change,
                        inputs=[model_selector, state], 
                        outputs=[model_info_box, msg_box, state]
                        )
        msg_box.submit(user, 
                        inputs=[state, msg_box, chatbot], 
                        outputs=[msg_box, chatbot], 
                        queue=True).then(
                            call_llm, 
                            inputs=[state, chatbot, model_selector, system_role, cbox_use_document, sldr_k_top], 
                            outputs=[chatbot, state]
                            )
        clear.click(clear_data, 
                    inputs=[state], 
                    outputs=[chatbot, state], 
                    queue=True
                    )
        uplbtn_file.upload(embed_file, 
                           inputs=[state, uplbtn_file, sldr_chunk_size, sldr_chunk_overlap, cbox_use_ocr], 
                           outputs=[state, file_output, cbox_use_document, btn_remove]
                           )
        btn_remove.click(remove_file, 
                         inputs=[state], 
                         outputs=[state, file_output, cbox_use_document, btn_remove]
                         )
        sldr_chunk_size.change(slider_chunk_change, [state, sldr_chunk_size, sldr_chunk_overlap], [sldr_chunk_overlap])
        sldr_chunk_overlap.change(slider_chunk_change, [state, sldr_chunk_size, sldr_chunk_overlap], [sldr_chunk_overlap])
    return chat_view    


def main()->None:
    """ Main program of the tutorial for CML """
    args = {}
    args["host"] = os.environ.get("HOSTNAME","0.0.0.0")
    args["port"] = int(os.environ.get("HOSTPORT",51001))
    log_level = int(os.environ.get("APPLOGLEVEL", logging.ERROR))
    if log_level < 10: log_level = 40
    logging.basicConfig(level=log_level,)

    # Load models
    with open('./settings/models.json', 'r') as file:
        ai_models = json.load(file)

    hana_cloud = {
        "host": os.getenv("HOST"),
        "user": os.getenv("HANA_USERNAME",""),
        "password": os.getenv("HANA_PASSWORD","") 
    }
    
    # Get ready to connect to Gen AI Hub
    global GENAI_HUB
    GENAI_HUB = GenAIHubProxyClient(base_url=os.getenv('AICORE_BASE_URL'), auth_url=os.getenv('AICORE_AUTH_URL'), client_id=os.getenv('AICORE_CLIENT_ID'), client_secret=os.getenv('AICORE_CLIENT_SECRET'), resource_group=os.getenv('AICORE_RESOURCE_GROUP','default'))
    
    # Create chat UI
    chat_view = build_chat_view(models=ai_models, conn_data=hana_cloud, sys_roles=get_system_roles())
    # Queue input of each user
    chat_view.queue(max_size=10)
    # Start the Gradio server
    chat_view.launch(
        show_api=False,
        server_name=args["host"],
        server_port=args["port"],
        allowed_paths=["./img", "/tmp"]
    )
    
if __name__ == "__main__":
    main()