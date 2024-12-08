import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
import os
import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Chargement de ma clé openAI

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("Clé API récupérée avec succès !")
else:
    print("Erreur : La clé API n'a pas été trouvée.")

from openai import OpenAI
openai.api_key = api_key


# Chargement des documents et création du vecteur store
# a reprendre pour se connecter a un volume persitent de données

@st.cache_resource
def load_vectorstore():
    
    loader = PyPDFLoader("../src/Conception et construction bas carbone.pdf")
    data = loader.load()
    content=data[0]

    chunk_size=1000
    chunk_overlap=200
    rc_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n"," ",""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)

    chunks = rc_splitter.split_text(content.page_content)

    docs= [Document(page_content=chunk) for chunk in chunks]

    embedding_function = OpenAIEmbeddings(api_key=api_key, model='text-embedding-3-small')
    vectorstore = Chroma.from_documents(docs, embedding=embedding_function, persist_directory="./chroma_db")

    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

retriever = load_vectorstore()

# Initialisation du modèle et de la mémoire
llm = ChatOpenAI(model="gpt-4", api_key=api_key)
memory = ConversationBufferWindowMemory(k=5, return_messages=True)

# Initialisation de la chaîne
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    memory=memory,
    return_source_documents=False,
    output_key="result"
)

# Interface Streamlit
st.title("💬 Chatbot avec LangChain")
st.write("Posez une question au chatbot.")

st.sidebar.title("Projet NLP Datascientest")
st.sidebar.write("Promotion Septembre-2024")

st.sidebar.image("../src/datascientest.png", caption="Datascientest")

for _ in range(20):
    st.sidebar.empty()
    
st.sidebar.write("David MICHEL")
st.sidebar.write("Charafeddine MECHRI")
st.sidebar.write("Marine MERLE")



# Historique de la conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Champ de saisie utilisateur
user_input = st.text_input("Vous :", placeholder="Entrez votre question ici...", key="user_input")

if user_input:
    # Ajouter la question utilisateur à l'historique
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Obtenir la réponse du chatbot
    response = qa_chain.run(user_input)

    # Ajouter la réponse du chatbot à l'historique
    st.session_state.messages.append({"role": "assistant", "content": response})

# Afficher l'historique de la conversation
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**Vous :** {message['content']}")
    else:
        st.markdown(f"**Chatbot :** {message['content']}")
