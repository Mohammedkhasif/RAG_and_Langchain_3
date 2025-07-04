import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Step 1: Load API key from .env file
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Step 2: Provide full path to your Hadith PDF
pdf_path = r"C:\Users\91789\Dropbox\PC\Desktop\gemini_project_3\hadith.pdf"  # 👈 Change if neededwhat are intentions
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f" PDF not found at path: {pdf_path}")

# Step 3: Load and chunk Hadith PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Step 4: Create Gemini embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# Step 5: Create RetrievalQA chain using Gemini Pro
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatGoogleGenerativeAI(model="models/gemini-1.5-flash"),
    retriever=retriever,
    return_source_documents=True
)

# Step 6: Ask Hadith-based questions
print("\n📖 Ask Islamic questions based on the Hadith PDF (type 'exit' to quit)\n")
while True:
    query = input("🕌 Your Question: ")
    if query.lower() == "exit":
        print("🔚 Exiting Hadith QA Bot. JazakAllah Khair.")
        break

    result = qa_chain({"query": query})
    print("\n🤖 Answer:\n", result["result"])
