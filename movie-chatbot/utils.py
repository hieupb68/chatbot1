from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

HISTORY_LEN = 4

model_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
)

prompt_template = open(f'prompts/answer.txt', "r", encoding='utf-8').read()

vector_db = Chroma(
    collection_name="movie",
    embedding_function=model_embeddings,
    persist_directory="chroma_db",  # Where to save data locally, remove if not necessary
    create_collection_if_not_exists=True
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.6,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

answer_chain = (
    PromptTemplate.from_template(prompt_template)
    | llm
    | StrOutputParser()
)

prompt_template = open(f'prompts/context_filtering.txt', "r", encoding='utf-8').read()

context_filtering_chain = (
    PromptTemplate.from_template(prompt_template)
    | llm
    | StrOutputParser()
)

def semantic_search(query):
    results = vector_db.similarity_search(query, k=10)
    return results

def convert_to_llm_context(documents):
    context = ''
    for i, document in enumerate(documents):
        context += f"** Phim thứ {i+1}:" + "\n" + \
        document.page_content + "\n" + \
        f"URL xem phim: {document.metadata['video_url']}" + "\n" + \
        "\n"
    return context

def context_filtering(query, context):
    context_filtered = context_filtering_chain.invoke({'user_msg': query, 'context': context})
    return context_filtered

def process_messages(messages):
    user_msg = messages[-1].content
    history_chat = ''
    messages = messages[-(HISTORY_LEN+1):-1]

    for i in range(0, len(messages), 2):
        history_chat += f"   - user: {messages[i].content}\n"
        history_chat += f"   - you: {messages[i+1].content}\n\n"

    if not history_chat: 
        history_chat = "Chưa có lịch sử trò chuyện!"
    
    return user_msg, history_chat

def gen_response(messages):
    user_msg, history_chat = process_messages(messages)
    results = semantic_search(user_msg)
    context = convert_to_llm_context(results) 
    context = context_filtering(user_msg, context)
    response = answer_chain.invoke({'user_msg': user_msg, 'history_chat': history_chat, 'context': context})
    return response