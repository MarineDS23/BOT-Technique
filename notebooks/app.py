import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Chargement de ma clé OpenAI
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("Clé API récupérée avec succès !")
else:
    st.error("Erreur : La clé API OpenAI n'a pas été trouvée.")
    st.stop()

# Configuration du vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name="articles_reglementaires",
    persist_directory="./data/chroma_db3",
    embedding_function=embedding_model
)

# Définition du retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Initialisation du modèle et de la mémoire
llm = ChatOpenAI(model="gpt-4o", api_key=api_key)
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

st.sidebar.write("David MICHEL")
st.sidebar.write("Charafeddine MECHRI")
st.sidebar.write("Marine MERLE")

# Historique de la conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Champ de saisie utilisateur
user_input = st.text_input("Vous :", placeholder="Entrez votre question ici...")

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
