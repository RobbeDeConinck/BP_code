import os
import re
from typing import List, TypedDict, Dict, Any
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from IPython.display import Image, display
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()  # This will look for .env in the current directory

# Get environment variables with error handling
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Optional LangSmith configuration
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Configure LLM
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.2,
    max_tokens=250,
    api_key=OPENAI_API_KEY,
)

# Configure embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"}
)

# Initialize vector store
vector_store = InMemoryVectorStore(embeddings)

# Configure text splitter for better chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Increased for better context
    chunk_overlap=250,  # Increased overlap for better continuity
    length_function=len,
    separators=[
        "\n\n",
        "\n",
        ".",
        "!",
        "?",
        ";",
        ":",
        " ",
        "",
    ],  # More natural separators
    keep_separator=True,
    strip_whitespace=True,  # Remove extra whitespace
    add_start_index=True,  # Track chunk positions
)


# Add preprocessing function to clean text
def preprocess_text(text: str) -> str:
    """Clean and normalize text before splitting."""
    # Remove multiple dots and dashes
    text = re.sub(r"\.{3,}|-{3,}", " ", text)
    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)
    # Remove page numbers and headers
    text = re.sub(r"Pagina \d+ van \d+", "", text)
    text = re.sub(r"HOGENT\s+", "", text)
    return text.strip()


# Load and process PDF
def load_and_process_pdf(file_path: str) -> List[Document]:
    """Load and process PDF with improved text cleaning."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Preprocess text before splitting
    for page in pages:
        page.page_content = preprocess_text(page.page_content)

    splits = text_splitter.split_documents(pages)
    return splits


# Define state type
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define retrieval function
def retrieve(question: str) -> str:
    """Retrieve relevant documents with improved relevance scoring."""
    # Get initial set of documents
    docs = vector_store.similarity_search_with_score(
        question, k=20  # Increased for better filtering
    )

    # Sort by relevance score
    docs.sort(key=lambda x: x[1])

    # Filter and accumulate context
    relevant_docs = []
    total_length = 0
    max_length = 3000  # Increased max context length

    for doc, score in docs:
        # Skip if score indicates low relevance
        if score > 0.8:
            continue

        # Skip if document is too similar to already included ones
        if any(
            similarity_score(doc.page_content, d.page_content) > 0.7
            for d in relevant_docs
        ):
            continue

        relevant_docs.append(doc)
        total_length += len(doc.page_content)

        # Stop if we have enough context
        if total_length >= max_length:
            break

    if not relevant_docs:
        return "Geen relevante informatie gevonden voor deze vraag."

    # Combine and format context
    context = "\n\n".join(
        f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(relevant_docs)
    )

    return context


def similarity_score(text1: str, text2: str) -> float:
    """Calculate simple text similarity score."""
    # Convert to sets of words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))

    return intersection / union if union > 0 else 0


# Define generation function
def generate(question: str, context: str) -> str:
    """Generate comprehensive answer with improved prompt engineering."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Je bent een expert onderwijsassistent die gespecialiseerd is in het analyseren van educatieve documenten.
        
        Richtlijnen voor het beantwoorden:
        1. Geef een duidelijk gestructureerd antwoord met een inleiding en conclusie
        2. Gebruik bullet points voor meerdere punten
        3. Verwijs naar specifieke voorbeelden uit de context
        4. Wees concreet en praktisch in je adviezen
        5. Als informatie ontbreekt, geef dit duidelijk aan
        6. Vermijd herhalingen en vage uitspraken
        7. Focus op de meest relevante en actuele informatie
        
        Context:
        {context}
        """,
            ),
            ("human", "{question}"),
        ]
    )

    # Format messages with improved context
    messages = prompt.format_messages(question=question, context=context)

    # Generate response with improved parameters
    response = llm.invoke(
        messages,
        temperature=0.2,  # Slightly increased for more natural responses
        max_tokens=400,  # Increased for more comprehensive answers
        top_p=0.9,  # More diverse token selection
        frequency_penalty=0.5,  # Reduce repetition
        presence_penalty=0.3,  # Encourage diversity
        stop=None,
    )

    # Post-process the response
    answer = response.content.strip()

    # Remove any remaining formatting artifacts
    answer = re.sub(r"\s+", " ", answer)
    answer = re.sub(r"\.{3,}|-{3,}", " ", answer)

    # Ensure proper formatting of bullet points
    answer = re.sub(r"^\s*[-•]\s*", "• ", answer, flags=re.MULTILINE)

    return answer.strip()


# Build and compile graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Load prompt template
prompt = hub.pull("rlm/rag-prompt")


# Example usage
def ask_question(question: str) -> Dict[str, Any]:
    """Process a question and return answer with context."""
    # Retrieve relevant context
    context = retrieve(question)

    # Generate answer
    answer = generate(question, context)

    return {"question": question, "context": context, "answer": answer}
