from db.vectorstore import load_vectorstore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain.schema import HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama  # ✅ fallback LLaMA
from google.api_core.exceptions import ResourceExhausted
from utils.config import get_embeddings_with_fallback
import os
import logging
from typing import List, Dict, Tuple
import re

# ✅ Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Use gemini-1.5-flash instead of pro (fewer quota errors)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

# Fallback model (runs locally via Ollama)
fallback_llm = ChatOllama(model="llama3")

# Base folder for storing per-student chat histories
CHAT_HISTORY_DIR = os.path.join("data", "chat_histories")
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)


def _get_history(student_id: str) -> FileChatMessageHistory:
    """Get or create conversation history for a student."""
    path = os.path.join(CHAT_HISTORY_DIR, f"{student_id}.json")
    return FileChatMessageHistory(path)


def save_conversation_context(student_id: str, role: str, content: str):
    """Save a message to the student's conversation history."""
    history = _get_history(student_id)

    if role == "user":
        history.add_message(HumanMessage(content=content))
    elif role == "ai":
        history.add_message(AIMessage(content=content))
    else:
        raise ValueError("Role must be either 'user' or 'ai'")


def _invoke_with_fallback(messages):
    """Try Gemini first, then fallback to LLaMA if quota exceeded."""
    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        # Check if response is empty or contains "unknown"
        if not content or content.lower() in ["unknown", "i don't know", "i'm not sure"]:
            logger.warning("⚠️ Gemini returned empty or unknown response, trying LLaMA")
            response = fallback_llm.invoke(messages)
            return {"model": "llama", "content": response.content}
        return {"model": "gemini", "content": content}
    except ResourceExhausted as e:
        logger.warning(f"⚠️ Gemini quota exceeded, switching to LLaMA: {e}")
        response = fallback_llm.invoke(messages)
        return {"model": "llama", "content": response.content}
    except Exception as e:
        logger.error(f"⚠️ Unexpected error with Gemini, using LLaMA: {e}")
        try:
            response = fallback_llm.invoke(messages)
            return {"model": "llama", "content": response.content}
        except Exception as fallback_error:
            logger.error(f"⚠️ Both models failed: {fallback_error}")
            return {"model": "fallback", "content": "I apologize, but I'm having trouble processing your request right now. Please try again later or rephrase your question."}


def get_llm_recommendation_reason(student_id: str, recommendations: list) -> str:
    """Generate a concise explanation using prior conversation context and the recommended internships."""
    history = _get_history(student_id)

    # Format recommendations into readable text
    rec_text = "\n\n".join([
        f"Recommendation {idx+1}:\n{rec}" if isinstance(rec, str)
        else f"Recommendation {idx+1}:\n{rec}"
        for idx, rec in enumerate(recommendations)
    ])

    # ✅ Truncate to avoid token overload
    rec_text = rec_text[:3000]

    # Instruction prompt
    prompt = (
        "Given the student's conversation context and the following recommended internships, "
        "explain briefly why these were chosen and how they match the student's profile. "
        "Be helpful and specific."
    )

    # ✅ Only keep last 5 messages to avoid quota issues
    messages = history.messages[-5:] + [
        HumanMessage(content=f"Context recall and recommendations:\n\n{rec_text}\n\n{prompt}")
    ]

    response = _invoke_with_fallback(messages)
    logger.info(f"LLM ({response['model']}) Response: {response['content']}")

    # Save AI's response for continuity
    save_conversation_context(student_id, "ai", response["content"])

    return response["content"]


def get_internship_recommendations(student_summary: str, top_k: int = 5):
    """Get internship recommendations without saving to conversation history."""
    from db.database import SessionLocal
    from db.models import Internship
    
    vectorstore = load_vectorstore()
    if not vectorstore:
        # If the vectorstore cannot be built/loaded (e.g., no internships), return empty recommendations
        print("⚠️ No vectorstore available for recommendations; returning empty list.")
        return []

    results = vectorstore.similarity_search(student_summary, k=top_k)

    # Get job IDs from vectorstore results
    job_ids = [res.metadata.get("job_id") for res in results if res.metadata.get("job_id")]
    
    # Fetch full internship details from database
    db = SessionLocal()
    try:
        internships = db.query(Internship).filter(Internship.job_id.in_(job_ids)).all()
        
        # Create a mapping of job_id to internship data
        internship_map = {internship.job_id: internship for internship in internships}
        
        recs = []
        for res in results:
            job_id = res.metadata.get("job_id")
            if job_id and job_id in internship_map:
                internship = internship_map[job_id]
                recs.append({
                    "job_id": internship.job_id,
                    "id": str(internship.job_id),
                    "_id": str(internship.job_id),
                    "title": internship.title,
                    "description": internship.description,
                    "skills": internship.skills_required,
                    "location": internship.location,
                    "stipend": internship.stipend,
                    "duration": internship.duration,
                    "internship": res.page_content  # Keep original for scoring
                })
    finally:
        db.close()
    
    return recs


def get_internship_recommendations_by_vector(query_embedding: list, top_k: int = 5):
    """Get internship recommendations using a precomputed embedding vector."""
    from db.database import SessionLocal
    from db.models import Internship
    
    vectorstore = load_vectorstore()
    if not vectorstore:
        # Without a vectorstore we can't perform VSM; return no recommendations.
        print("⚠️ No vectorstore available for similarity search; returning empty list.")
        return []

    try:
        results = vectorstore.similarity_search_by_vector(query_embedding, k=top_k)
    except AssertionError as e:
        print(f"⚠️ Embedding dimension mismatch error: {e}")
        print("⚠️ Falling back to text-based search instead of vector search")
        # Fall back to text search if dimensions don't match
        return get_internship_recommendations("resume text placeholder", top_k)
    except Exception as e:
        print(f"⚠️ Vector search error: {e}")
        print("⚠️ Falling back to text-based search")
        return get_internship_recommendations("resume text placeholder", top_k)

    # Get job IDs from vectorstore results
    job_ids = [res.metadata.get("job_id") for res in results if res.metadata.get("job_id")]
    
    # Fetch full internship details from database
    db = SessionLocal()
    try:
        internships = db.query(Internship).filter(Internship.job_id.in_(job_ids)).all()
        
        # Create a mapping of job_id to internship data
        internship_map = {internship.job_id: internship for internship in internships}
        
        recs = []
        for res in results:
            job_id = res.metadata.get("job_id")
            if job_id and job_id in internship_map:
                internship = internship_map[job_id]
                recs.append({
                    "job_id": internship.job_id,
                    "id": str(internship.job_id),
                    "_id": str(internship.job_id),
                    "title": internship.title,
                    "description": internship.description,
                    "skills": internship.skills_required,
                    "location": internship.location,
                    "stipend": internship.stipend,
                    "duration": internship.duration,
                    "internship": res.page_content  # Keep original for scoring
                })
    finally:
        db.close()
    
    return recs


def embed_text(text: str) -> Tuple[List[float], str]:
    """Embed text using current embeddings with fallback. Returns (vector, provider)."""
    embeddings_model, provider = get_embeddings_with_fallback()
    vector = embeddings_model.embed_query(text)
    return vector, provider


def average_embeddings(vectors: List[List[float]]) -> List[float]:
    """Compute element-wise average of multiple embedding vectors."""
    if not vectors:
        return []
    length = len(vectors[0])
    # Guard: ensure uniform length
    filtered = [v for v in vectors if len(v) == length]
    if not filtered:
        return []
    sums = [0.0] * length
    for v in filtered:
        for i, val in enumerate(v):
            sums[i] += val
    count = float(len(filtered))
    return [s / count for s in sums]


def detect_intent(user_query: str) -> str:
    """Classify intent: 'recommend_internships', 'suggest_skills', or 'general'."""
    query_lower = user_query.lower()
    
    # Direct keyword matching for better accuracy
    internship_keywords = [
        "internship", "intern", "job", "position", "role", "opportunity", "opportunities",
        "find", "search", "looking", "apply", "application", "hiring", "recruit",
        "company", "work", "experience", "career", "employment"
    ]
    
    skill_keywords = [
        "skill", "skills", "learn", "learning", "improve", "improvement", "develop", "development",
        "study", "studying", "practice", "practicing", "master", "mastering", "expertise",
        "knowledge", "ability", "abilities", "competency", "competencies", "training"
    ]
    
    # Check for internship-related queries
    if any(keyword in query_lower for keyword in internship_keywords):
        return "recommend_internships"
    
    # Check for skill-related queries
    if any(keyword in query_lower for keyword in skill_keywords):
        return "suggest_skills"
    
    # Use LLM for more complex queries
    prompt = (
        "Classify the user's intent into one of: recommend_internships, suggest_skills, general.\n"
        "- recommend_internships: asking for internships, job opportunities, career advice, application help\n"
        "- suggest_skills: asking what skills to learn/pursue/improve/develop\n"
        "- general: anything else like greetings, general questions, clarifications\n"
        f"User query: {user_query}\n"
        "Answer with only the label."
    )
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        label = resp.content.strip().lower()
        if "recommend" in label or "intern" in label or label == "recommend_internships":
            return "recommend_internships"
        if "skill" in label or label == "suggest_skills":
            return "suggest_skills"
        return "general"
    except Exception:
        return "general"


def score_skill_match(resume_summary: str, internship_text: str) -> Dict:
    """Compute skill match using rule-based extraction first, then LLM, then naive token overlap."""

    def extract_skills_rule_based(text: str) -> List[str]:
        # Prefer line starting with "Skills:" then comma-separated list
        m = re.search(r"(?im)^\s*skills\s*:\s*(.+)$", text)
        skills_line = m.group(1) if m else ""
        skills = []
        if skills_line:
            skills = [s.strip().lower() for s in re.split(r",|/|\||;", skills_line) if s.strip()]
        # Also capture common tech tokens present as standalone words (basic heuristic)
        common = [
            "python","java","javascript","typescript","c++","c#","sql","nosql","mongodb","postgres","mysql",
            "react","angular","vue","node","express","django","flask","fastapi","spring","dotnet",
            "html","css","tailwind","bootstrap","nextjs","nestjs",
            "aws","gcp","azure","docker","kubernetes","terraform","git","linux",
            "ml","machine learning","deep learning","pytorch","tensorflow","sklearn","opencv",
            "nlp","llm","langchain","faiss","retrieval","rag","airflow","spark","hadoop",
            "firebase","supabase","graphql","rest","kafka","redis","rabbitmq"
        ]
        norm_text = text.lower()
        for tok in common:
            if tok in norm_text and tok not in skills:
                skills.append(tok)
        # Dedup and limit
        seen = set()
        out = []
        for s in skills:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out[:50]

    # Rule-based extraction
    resume_skills = extract_skills_rule_based(resume_summary)
    internship_skills = extract_skills_rule_based(internship_text)

    if internship_skills:
        rs = set(resume_skills)
        iset = set(internship_skills)
        matched = sorted(list(rs.intersection(iset)))
        missing = sorted(list(iset - rs))
        denom = max(1, len(iset))
        percent = int(100 * len(matched) / denom)
        # If we have at least some internship skills, trust this computation
        if percent > 0 or len(matched) + len(missing) > 0:
            return {"percent_match": percent, "matched_skills": matched[:10], "missing_skills": missing[:10]}

    # LLM-based scoring as secondary option
    prompt = (
        "You are a precise evaluator. Given a student's resume summary and an internship description,"
        " return a strict JSON with: percent_match (0-100 integer), matched_skills (array of strings),"
        " missing_skills (array of strings). Keep arrays concise and deduplicated.\n\n"
        f"Resume Summary:\n{resume_summary[:2000]}\n\n"
        f"Internship Text:\n{internship_text[:2000]}\n\n"
        "JSON only: {\"percent_match\": 0-100, \"matched_skills\": [], \"missing_skills\": []}"
    )
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        import json
        data = json.loads(content)
        pm = int(max(0, min(100, int(data.get("percent_match", 0)))))
        matched = [s.strip().lower() for s in data.get("matched_skills", []) if isinstance(s, str)]
        missing = [s.strip().lower() for s in data.get("missing_skills", []) if isinstance(s, str)]
        return {"percent_match": pm, "matched_skills": matched[:10], "missing_skills": missing[:10]}
    except Exception:
        # Final fallback: naive token overlap
        rs = set([t.lower() for t in re.findall(r"[a-zA-Z+#\.]+", resume_summary)])
        it = set([t.lower() for t in re.findall(r"[a-zA-Z+#\.]+", internship_text)])
        overlap = rs.intersection(it)
        denom = max(1, len(it))
        percent = int(100 * len(overlap) / denom)
        return {
            "percent_match": percent,
            "matched_skills": sorted(list(overlap))[:10],
            "missing_skills": sorted(list(it - rs))[:10]
        }


def augment_recommendations_with_scoring(resume_summary: str, recs: List[Dict], top_n: int = 3) -> List[Dict]:
    """For top_n recs, add percent_match and skill_gap fields (missing_skills)."""
    enhanced = []
    for idx, rec in enumerate(recs):
        if idx < top_n:
            internship_text = rec.get("internship", "")
            score = score_skill_match(resume_summary, internship_text)
            rec = {
                **rec,
                "match_percentage": score.get("percent_match", 0),  # Changed to match frontend expectation
                "matched_skills": score.get("matched_skills", []),
                "skill_gap": score.get("missing_skills", []),
            }
        enhanced.append(rec)
    return enhanced


def suggest_skills_to_pursue(user_query: str, resume_summary: str) -> str:
    """LLM suggests concise, prioritized skills to learn based on resume and query."""
    prompt = (
        "Given the student's resume summary and question, suggest the most impactful 5-8 skills to pursue."
        " Group by theme if helpful. Keep it concise, actionable, and tailored to the profile.\n\n"
        f"Resume Summary:\n{resume_summary[:2000]}\n\n"
        f"Question:\n{user_query}\n\n"
        "Respond in bullets, each with a 1-line rationale."
    )
    resp = _invoke_with_fallback([HumanMessage(content=prompt)])
    return resp["content"]

def answer_with_context(student_id: str, user_query: str, resume_summary: str = "", recommendations: list = None, conversation_context: dict = None) -> str:
    """Answer a user question with resume context and recommendations, with improved context handling."""
    # Save user's question
    save_conversation_context(student_id, "user", user_query)

    # Build comprehensive context
    context_parts = []
    if resume_summary:
        context_parts.append(f"Student Resume Summary: {resume_summary[:800]}")  # Limit resume text
    
    if recommendations:
        rec_details = []
        for i, rec in enumerate(recommendations[:3], 1):
            rec_details.append(f"{i}. {rec.get('title', 'N/A')} - {rec.get('location', 'N/A')} - {rec.get('stipend', 'N/A')} - Match: {rec.get('match_percentage', 0)}%")
        context_parts.append(f"Recommended Internships:\n" + "\n".join(rec_details))

    context = "\n\n".join(context_parts)
    
    # Add conversation context if available
    conversation_info = ""
    if conversation_context:
        stage = conversation_context.get("conversation_stage", "initial")
        last_topic = conversation_context.get("last_topic", "")
        
        if stage == "discussing_internships":
            conversation_info = "\nCONVERSATION CONTEXT: The student is currently discussing internship opportunities. They may want to know more about specific roles, how to apply, or what makes them a good fit."
        elif stage == "discussing_skills":
            conversation_info = "\nCONVERSATION CONTEXT: The student is focused on skill development. They want to know what skills to learn or improve to strengthen their profile."
        elif stage == "discussing_applications":
            conversation_info = "\nCONVERSATION CONTEXT: The student is asking about the application process. They need practical guidance on resumes, interviews, or application strategies."
        elif stage == "discussing_career":
            conversation_info = "\nCONVERSATION CONTEXT: The student is thinking about their career path and future goals. They want guidance on career direction and planning."
    
    # Enhanced prompt that provides proactive guidance
    prompt = f"""You are an expert career advisor helping a student with their internship search and career development. 

CONTEXT:
{context}{conversation_info}

STUDENT'S QUESTION: {user_query}

INSTRUCTIONS:
- Provide specific, actionable advice based on their resume and the recommended internships
- If they ask about internships, reference the specific recommendations above and explain why they're a good fit
- If they ask about skills, suggest specific skills based on their profile and the job requirements
- If they ask about applications, provide step-by-step guidance tailored to their background
- If they ask about career paths, suggest relevant options based on their background
- Be encouraging and specific, not generic
- Reference specific details from their resume and recommendations when relevant
- If the question is vague, provide helpful suggestions based on their profile and current conversation context
- Build on the conversation flow naturally

RESPONSE:"""

    messages = [HumanMessage(content=prompt)]

    response = _invoke_with_fallback(messages)
    content = response["content"]

    # Enhanced fallback that's more helpful
    if not content or content.lower().strip() in ["unknown", "i don't know", "i'm not sure", ""]:
        # Provide context-aware fallback based on the question type
        if any(word in user_query.lower() for word in ["internship", "job", "position", "role"]):
            content = f"Based on your profile, I can see you have strong skills in your field. I've found some great internship opportunities that match your background. Would you like me to explain why these specific roles are a good fit for you, or would you prefer guidance on how to apply?"
        elif any(word in user_query.lower() for word in ["skill", "learn", "improve", "develop"]):
            content = f"Looking at your resume and the recommended positions, I can suggest specific skills that would strengthen your profile. Would you like me to recommend skills to learn based on the job requirements, or are you interested in a particular area of development?"
        elif any(word in user_query.lower() for word in ["apply", "application", "resume", "interview"]):
            content = f"I'd be happy to help you with the application process! I can provide guidance on tailoring your resume for specific roles, preparing for interviews, or writing compelling cover letters. What aspect of the application process would you like to focus on?"
        else:
            content = f"I understand you're asking about '{user_query}'. Based on your profile, I can help you with:\n\n• Finding the right internship opportunities\n• Developing skills to strengthen your applications\n• Guidance on the application and interview process\n• Career path recommendations\n\nWhat would be most helpful for you right now?"

    # Save AI's answer
    save_conversation_context(student_id, "ai", content)

    return content
