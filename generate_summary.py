import os
from main import load_and_process_pdf, vector_store
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


def ask_question_openai(model, question, context):
    """
    Ask a question using OpenAI ChatGPT model
    """
    try:
        # Create the prompt - improved for better study materials
        prompt = f"""Je bent een expert in het maken van leermaterialen en studiesamenvattingen voor cursussen.
        
Je taak is om gedetailleerde studieaantekeningen te maken die studenten kunnen gebruiken om efficiënt te studeren en de stof te beheersen.

De samenvattingen moeten:
1. Belangrijke definities en begrippen duidelijk markeren en uitleggen
2. Kernconcepten in detail behandelen
3. Concrete voorbeelden bevatten om abstracte ideeën te verduidelijken
4. Verbanden leggen tussen verschillende concepten
5. Praktische toepassingen benadrukken
6. Potentiële examenonderwerpen identificeren
7. Vraag- en antwoordformaten bevatten voor zelfstudie
8. Visuele structuur hebben met koppen, subkoppen en opsommingstekens

Context: {context}

Vraag: {question}

Antwoord (maak een uitgebreide, gestructureerde studiegids):"""

        # Generate response
        response = model.invoke(prompt)
        return {"answer": response.content.strip()}
    except Exception as e:
        print(f"Error with OpenAI model: {str(e)}")
        return {"answer": f"Error: {str(e)}"}


def generate_course_summary(model, file_path):
    """
    Generate a comprehensive summary of all texts in the course.
    """
    print("Generating detailed study materials...")

    # First, get the PDF content
    try:
        splits = load_and_process_pdf(file_path)
        print(f"Successfully loaded PDF with {len(splits)} sections")
    except Exception as e:
        print(f"Error loading PDF: {str(e)}")
        return

    # Generate a more study-focused summary with specific learning objectives
    summary_questions = [
        "Maak een uitgebreide studiegids met alle belangrijke concepten, definities en theorieën uit de cursus. Inclusief toepassingen en voorbeelden.",
        "Werk de belangrijkste leerdoelen uit in een format waarmee studenten kunnen controleren of ze de stof beheersen. Voeg oefenvragen toe. geef ook de antwoorden op de oefeningen.",
    ]

    summary = []

    # Process in chunks to handle context limitations
    chunk_size = 5000  # Characters per chunk
    all_text = "\n".join([doc.page_content for doc in splits])
    chunks = [all_text[i : i + chunk_size] for i in range(0, len(all_text), chunk_size)]

    for i, question in enumerate(summary_questions):
        print(f"\nProcessing: {question}")

        # Process each chunk separately and combine results
        chunk_results = []
        for j, chunk in enumerate(chunks):
            print(f"  Processing chunk {j+1}/{len(chunks)}")
            try:
                chunk_context = f"[Deel {j+1}/{len(chunks)} van de cursus] {chunk}"
                result = ask_question_openai(
                    model, question + f" (Deel {j+1}/{len(chunks)})", chunk_context
                )
                chunk_results.append(result["answer"])
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")

        # Combine chunk results
        combined_result = "\n\n".join(chunk_results)

        # Add to summary
        summary.append(f"\n## {question}\n")
        summary.append(combined_result)
        summary.append("\n---\n")

    # Save the summary to a markdown file
    with open("cursus_studiegids.md", "w", encoding="utf-8") as f:
        f.write("# Uitgebreide Cursus Studiegids\n\n")
        f.write("## Inhoudsopgave\n\n")
        for i, question in enumerate(summary_questions, 1):
            sanitized_question = question.split(".")[0].strip()
            f.write(
                f"{i}. [{sanitized_question}](#{sanitized_question.lower().replace(' ', '-').replace('?', '')})\n"
            )
        f.write("\n")
        f.write("".join(summary))

    print("\nUitgebreide studiegids is opgeslagen in 'cursus_studiegids.md'")


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_APIKEY")
    if not openai_api_key:
        print("Error: OPENAI_APIKEY not found in .env file")
        exit(1)

    # Initialize OpenAI model - use GPT-4 for better summaries if available
    try:
        model = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=1500,
            api_key=openai_api_key,
        )
        print(f"Successfully loaded OpenAI model")
    except Exception as e:
        print(f"Error loading OpenAI model: {str(e)}")
        exit(1)

    # Example usage with a file path
    file_path = "./Cursus-Werkplekleren-inspelen-op-onderwijsbehoeften.pdf"  # This is just for testing
    generate_course_summary(model, file_path)
