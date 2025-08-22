import os
import json
from dotenv import load_dotenv
import requests

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3-70b-8192")

def generate_llm_feedback(
    metrics_path='results_summary.json',
    transcript_path='transcript_with_timestamps.json',
):
    if not os.path.exists(metrics_path):
        return {"error": f"Metrics file '{metrics_path}' not found."}
    if not os.path.exists(transcript_path):
        return {"error": f"Transcript file '{transcript_path}' not found."}

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)

    transcript_text = "\n".join(
        [f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}" for seg in transcript_data]
    )

    # Compose prompt with full rubric and instructions to ONLY return JSON object
    prompt = f"""
You are a presentation evaluation expert.

Your task is to analyze the speaker's effectiveness using their video transcript and provided metrics.

---

### TRANSCRIPT (timestamps included)
{transcript_text}

---

### METRICS (extracted from audio/video analysis)
{json.dumps(metrics, indent=2)}

---

### INSTRUCTIONS

You MUST reason carefully using the rubric advice below for scoring.

Address the speaker directly as "you".

---

## RUBRIC

**Category: Quality of Speech**

* **Speech Rate**
  * Low: < 90 or > 170 wpm
  * Medium: 90–170 wpm
  * High: Optimal 110–150 wpm

* **Fluency & Pauses**
  * Low: >6 long pauses (>1.5s) and >5 stumbles
  * Medium: 2–5 long pauses or hesitations
  * High: <2 long pauses; smooth delivery

* **Voice Modulation**
  * Low: Flat pitch (<20 Hz range)
  * Medium: Some variation (20–60 Hz)
  * High: Expressive tone (>60 Hz pitch range)

---

**Category: English Proficiency**

* **Grammar Accuracy**
  * Low: >8 grammar errors/min
  * Medium: 3–8 grammar issues/min
  * High: <3 grammar issues/min

* **Vocabulary Use**
  * Low: Limited vocabulary; poor word choice
  * Medium: Moderate variety; mostly appropriate
  * High: Wide, appropriate word variety

* **Pronunciation**
  * Low: >10 mispronounced/unintelligible words
  * Medium: 3–10 issues
  * High: 0–2 mispronunciations

---

**Category: Filler Words & Pauses**

* **Filler Word Use**
  * Low: >10 fillers (um, like, etc.)
  * Medium: 4–10 filler words
  * High: 0–3 filler words

* **Pausing Patterns**
  * Low: >5 long/awkward pauses
  * Medium: 2–5 noticeable pauses
  * High: Smooth flow, <2 pauses

* **Word Repetition**
  * Low: >30% key word repetition
  * Medium: <20%
  * High: Diverse word use

---

**Category: Confidence & Body Language**

* **Eye Contact**
  * Use attention% from metrics:
    * Low: <30%
    * Medium: 30–70%
    * High: >70%

* **Facial Expression**
  * Low: Flat or disengaged >80% of the time
  * Medium: Some emotion and expression
  * High: Expressive and reactive face

* **Gestures & Posture**
  * Low: Stiff or no gestures
  * Medium: Some gestures; mostly relaxed
  * High: Natural gestures; confident

---

### OUTPUT

Provide your response **only** in the following JSON format with no additional text or explanation:

{{
  "Detailed Feedback": "[Your detailed feedback here as a string]",
  "Rubric Scoring Report": {{
    "Quality of Speech": {{
      "Speech Rate": "Low | Medium | High",
      "Fluency & Pauses": "Low | Medium | High",
      "Voice Modulation": "Low | Medium | High"
    }},
    "English Proficiency": {{
      "Grammar Accuracy": "Low | Medium | High",
      "Vocabulary Use": "Low | Medium | High",
      "Pronunciation": "Low | Medium | High"
    }},
    "Filler Words & Pauses": {{
      "Filler Word Use": "Low | Medium | High",
      "Pausing Patterns": "Low | Medium | High",
      "Word Repetition": "Low | Medium | High"
    }},
    "Confidence & Body Language": {{
      "Eye Contact": "Low | Medium | High",
      "Facial Expression": "Low | Medium | High",
      "Gestures & Posture": "Low | Medium | High"
    }}
  }}
}}
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert in presentation coaching."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2048
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=body
    )

    if response.status_code != 200:
        return {"error": f"LLM API Error {response.status_code}: {response.text}"}

    result = response.json()
    llm_response_text = result['choices'][0]['message']['content']

    # Directly parse the returned JSON object from LLM (expecting pure JSON)
    try:
        return json.loads(llm_response_text)
    except json.JSONDecodeError as e:
        # Return error info if JSON parsing fails
        return {"error": f"Failed to parse LLM JSON output: {str(e)}", "raw_output": llm_response_text}
