import os
from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI
from pydantic import BaseModel
from prompts import reviewer_prompt, validator_prompt, rewriter_prompt

# Load env
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"

app = FastAPI(title="SOP Reviewer API")

# ----- Response Models -----
class SOPReview(BaseModel):
    summary: str
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]


class ValidationFeedback(BaseModel):
    clarity_issues: List[str]
    missing_points: List[str]
    tone_problems: List[str]
    improvement_suggestions: List[str]


# call LLM
def call_llm(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

def safe_json_loads(text):
    try:
        return json.loads(text)
    except:
        return None


# ----------- API ENDPOINT -----------
@app.post("/review-sop")
def review_sop(request: SOPRequest):

    sop_text = request.sop_text.strip()

    # AGENT 1: REVIEWER 
    review_raw = call_llm(reviewer_prompt, sop_text)
    review_data = safe_json_loads(review_raw)

    if review_data is None:
        return {"error": "Reviewer agent returned invalid JSON"}
    
    # AGENT 2: VALIDATOR
    feedback_raw = call_llm(validator_prompt, json.dumps(review_data))
    feedback_data = safe_json_loads(feedback_raw)

    if feedback_data is None:
        return {"error": "Validator agent returned invalid JSON"}

    # AGENT 1 AGAIN: REWRITER
    final_raw = call_llm(
        rewriter_prompt,
        f"Original Review:\n{json.dumps(review_data)}\n\nFeedback:\n{json.dumps(feedback_data)}"
    )

    final_data = safe_json_loads(final_raw)

    if final_data is None:
        return {"error": "Rewriter agent returned invalid JSON"}

    # FINAL RESPONSE-
    return {
        "review": final_data
        #"validation_feedback": feedback_data 
    }

