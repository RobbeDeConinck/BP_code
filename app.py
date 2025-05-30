import streamlit as st
import os
import tempfile
from main import load_and_process_pdf, ask_question, vector_store
from generate_summary import generate_course_summary
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# Load environment variables
load_dotenv()

# Initialize OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize vector store
# vector_store is now imported from main.py

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Chat with PDF", "Generate Summary"])

# Disclaimer
st.warning(
    "**Please do not leave or refresh the site while your document is being processed.** This may interrupt the upload or summary generation."
)

if page == "Chat with PDF":
    st.title("PDF Document Assistant")

    # App state management
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "document_loaded" not in st.session_state:
        st.session_state.document_loaded = False

    if "document_text" not in st.session_state:
        st.session_state.document_text = ""

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Document upload section
    st.subheader("1. Upload Your Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file and not st.session_state.document_loaded:
        with st.spinner("Processing document..."):
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Process the PDF using the imported function
            try:
                splits = load_and_process_pdf(tmp_path)
                st.session_state.document_text = "\n".join(
                    [doc.page_content for doc in splits]
                )
                st.session_state.document_loaded = True
                st.success(f"Document '{uploaded_file.name}' processed successfully!")
                # Clean up the temp file
                os.unlink(tmp_path)
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

    # Q&A section - only show if a document is loaded
    if st.session_state.document_loaded:
        st.subheader("2. Ask Questions About Your Document")

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User input for questions
        prompt = st.chat_input("Ask a question about your document:")

        if prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response using the imported ask_question function
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = ask_question(prompt)
                    st.markdown(result["answer"])

                    # Add response to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result["answer"]}
                    )
    else:
        st.info("Please upload a document to start asking questions")

elif page == "Generate Summary":
    st.title("Generate PDF Summary")

    st.write("Upload a PDF file to generate a comprehensive summary.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file", type="pdf", key="summary_uploader"
    )

    if uploaded_file:
        with st.spinner("Processing document..."):
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # Initialize OpenAI model
                model = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.3,
                    max_tokens=1500,
                    api_key=openai_api_key,
                )

                # Only generate summary if we haven't already or if it's a new file
                if (
                    "current_file" not in st.session_state
                    or st.session_state.current_file != uploaded_file.name
                ):
                    # Generate summary using the temporary file path
                    generate_course_summary(model, tmp_path)

                    # Store the current file name and markdown content in session state
                    st.session_state.current_file = uploaded_file.name
                    with open("cursus_studiegids.md", "r", encoding="utf-8") as md_file:
                        st.session_state.md_content = md_file.read()

                st.success("Summary generated successfully!")

                # Use the stored markdown content for download
                st.download_button(
                    label="Download Summary as Markdown",
                    data=st.session_state.md_content,
                    file_name="course_summary.md",
                    mime="text/markdown",
                )

                # Clean up the temp file
                os.unlink(tmp_path)

            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
