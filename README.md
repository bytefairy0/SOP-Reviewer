# SOP Reviewer

An AI-powered API that reviews and provides feedback on Standard Operating Procedures (SOPs) using multi-agent LLM processing.

## Overview

SOP Reviewer uses three agents working together:
1. **Reviewer Agent** - Analyzes the SOP and provides initial review
2. **Validator Agent** - Validates the review for clarity and completeness
3. **Rewriter Agent** - Refines the review based on validation feedback

## Setup

### Prerequisites
- Python 3.8+
- Groq API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/bytefairy0/SOP-Reviewer.git
cd SOP-Reviewer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file
echo "GROQ_API_KEY=your_api_key_here" > .env
```

## Usage

### Running the API

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoint

**POST** `/review-sop`

Request body:
```json
{
  "sop_text": "Your SOP content here..."
}
```

Response:
```json
{
  "review": {
    "summary": "...",
    "strengths": ["..."],
    "weaknesses": ["..."],
    "suggestions": ["..."]
  }
}
```

## Example

```bash
curl -X POST "http://localhost:8000/review-sop" \
  -H "Content-Type: application/json" \
  -d '{"sop_text": "Your SOP here"}'
```

## API Documentation

Interactive API docs available at `http://localhost:8000/docs`

## Requirements

- fastapi
- uvicorn
- groq
- pydantic
- python-dotenv
