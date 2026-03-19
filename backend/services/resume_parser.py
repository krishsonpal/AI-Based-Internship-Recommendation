from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from google.api_core.exceptions import ResourceExhausted

llm = None

def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    return text


def get_llm():
    global llm
    if llm is None:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0
            )
        except Exception as e:
            print(f"Warning: Could not initialize Google Generative AI: {e}")

            class MockLLM:
                def invoke(self, messages):
                    return type(
                        'Response', (), {'content': 'Mock response - AI unavailable'}
                    )()

            llm = MockLLM()

    return llm


def extract_resume_summary(cv_text: str) -> dict:
    truncated = cv_text[:4000]

    prompt = f"""
Extract the following information from this resume:

- Skills
- Education
- Work/Project Experience
- Strengths

Resume:
{truncated}
"""

    try:
        llm_instance = get_llm()
        response = llm_instance.invoke([HumanMessage(content=prompt)])

        return {"summary": response.content}

    except ResourceExhausted:
        return {"summary": truncated[:800]}


def make_concise_summary(summary_text: str, max_chars: int = 800) -> str:

    prompt = f"""
Rewrite this resume summary to be concise and precise.
Focus on skills, domains, and results.

Target under {max_chars} characters.

Summary:
{summary_text[:4000]}
"""

    try:
        llm_instance = get_llm()
        resp = llm_instance.invoke([HumanMessage(content=prompt)])

        concise = resp.content.strip()
        return concise[:max_chars]

    except Exception:
        return summary_text[:max_chars]