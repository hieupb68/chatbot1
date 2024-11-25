import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import answer_chain, semantic_search, convert_to_llm_context, context_filtering, get_history_chat
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Movie Chatbot")
st.header("Movie Chatbot")

def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_msg := st.chat_input("Typing a message"):
        st.chat_message("user").markdown(user_msg)

        with st.chat_message("assistant"):
            # assistant_msg = st.markdown( chain.invoke({'user_msg': user_msg}))
            results = semantic_search(user_msg)
            print(results)
            context = convert_to_llm_context(results) 
            context = context_filtering(user_msg, context)
            print(context)
            history_chat = get_history_chat(st.session_state.messages)
            assistant_msg = st.write_stream(answer_chain.stream({'user_msg': user_msg, 'history_chat': history_chat, 'context': context}))
            print("-"*50)
           
        st.session_state.messages.append({"role": "user", "content": user_msg})
        st.session_state.messages.append({"role": "assistant", "content": assistant_msg})


if __name__ == "__main__":
    main()