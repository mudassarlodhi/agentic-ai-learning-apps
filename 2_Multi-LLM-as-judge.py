import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

# 1. Initialize all clients separately to keep code clean
openai_client = OpenAI(api_key=openai_api_key)

gemini_client = OpenAI(
    api_key=google_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

groq_client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

ACTOR_MODEL = "gpt-4o-mini"

# 🎬 The Input Data
MOVIE_SCENE = """The ending of The Dark Knight Rises movie where John Blake (Robin) found the Batcave.
In a short answer use reasoning and clues from the scene and followup scenes to tell whether John Blake took on the Batman's
legacy as a Batman or let the Batman be dead"""

# Actor generates the answer
response = openai_client.chat.completions.create(
    model=ACTOR_MODEL,
    messages=[{"role": "user", "content": MOVIE_SCENE}],
    temperature=0
)

actor_answer = response.choices[0].message.content

print(f"Question: {MOVIE_SCENE}")
print(f"Actor's Answer: '{actor_answer}'\n")
print("⏳ Gathering cross-provider jury votes...\n")

JUDGE_SYSTEM_PROMPT = "You are a Movie Critic. Grade this answer based on the movie's plot, director's intention and logic. Reply with exactly 'SATISFIED' or 'NOT SATISFIED' followed by a one-sentence reason."
USER_MESSAGE = f"Scene: {MOVIE_SCENE}\nAnswer to evaluate: {actor_answer}"

satisfied_votes = 0

# --- JUDGE 1: OpenAI ---
response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": USER_MESSAGE}
    ],
    temperature=0
)
judge_verdict = response.choices[0].message.content
print(f"⚖️ OpenAI (gpt-4o-mini) verdict: {judge_verdict}")

if judge_verdict.strip().upper().startswith("SATISFIED"):
    satisfied_votes += 1


# --- JUDGE 2: Google Gemini ---
response = gemini_client.chat.completions.create(
    model="gemini-3.5-flash",
    messages=[
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": USER_MESSAGE}
    ],
    temperature=0
)
judge_verdict = response.choices[0].message.content
print(f"⚖️ Gemini (gemini-2.5-flash) verdict: {judge_verdict}")

if judge_verdict.strip().upper().startswith("SATISFIED"):
    satisfied_votes += 1


# --- JUDGE 3: Groq (Llama) ---
response = groq_client.chat.completions.create(
    model="openai/gpt-oss-20b",  # Using a standard Groq model ID
    messages=[
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": USER_MESSAGE}
    ],
    temperature=0
)
judge_verdict = response.choices[0].message.content
print(f"⚖️ Groq (gpt-oss-120b) verdict: {judge_verdict}\n")

if judge_verdict.strip().upper().startswith("SATISFIED"):
    satisfied_votes += 1


# --- FINAL CONSENSUS ---
print("--- FINAL RESULT ---")
print(f"Total Satisfied Votes: {satisfied_votes} out of 3")

if satisfied_votes >= 2:
    print("🏆 SUCCESS: The majority of the cross-provider jury is satisfied with the answer!")
else:
    print("❌ FAILED: The cross-provider jury has rejected the answer.")
