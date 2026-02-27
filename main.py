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

class SOPRequest(BaseModel):
    sop_text: str

def call_llm(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.4
    )
    return response.choices[0].message.content
    
# Api Endpoint
@app.post("/review-sop")
def review_sop(request: SOPRequest):
    sop_text = request.sop_text
    # Agent 1 -> reviewr
    draft_review = call_llm(reviewer_prompt, sop_text)

    # Agent 2 -> validator
    feedback = call_llm(validator_prompt, draft_review)

    # Agent 1 -> impovement
    final_review = call_llm(
        rewriter_prompt,
        f"Original Review:\n{draft_review}\n\nFeedback:\n{feedback}"
    )

    return {
        "final_review": final_review
    }
