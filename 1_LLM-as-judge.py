import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import Markdown, display

load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

request = """Please come up with a challenging, nuanced question that
I can ask a number of LLMs to evaluate their intelligence. """
request += "Answer only with the question, no explanation."
messages = [{"role": "user", "content": request}]

openai = OpenAI()
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)
question = response.choices[0].message.content
print("------ QUESTION --------")
print(question)

competitors = []
answers = []
messages = [{"role": "user", "content": question}]

# The API we know well
model_name = "gpt-4o-mini"
response = openai.chat.completions.create(model=model_name, messages=messages)
answer = response.choices[0].message.content
# display(Markdown(answer))
competitors.append(model_name)
answers.append(answer)

gemini = OpenAI(api_key=google_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
model_name = "gemini-3.5-flash"

response = gemini.chat.completions.create(model=model_name, messages=messages)
answer = response.choices[0].message.content

# display(Markdown(answer))
competitors.append(model_name)
answers.append(answer)

# Updated with the latest Open Source model from OpenAI
groq = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
model_name = "openai/gpt-oss-120b"

response = groq.chat.completions.create(model=model_name, messages=messages)
answer = response.choices[0].message.content

# display(Markdown(answer))
competitors.append(model_name)
answers.append(answer)

# So where are we?
print(competitors)
# print(answers)

for competitor, answer in zip(competitors, answers):
    print(f"Competitor: {competitor}\n\n{answer}")

# Let's bring this together - note the use of "enumerate"

together = ""
for index, answer in enumerate(answers):
    together += f"# Response from competitor {index+1}\n\n"
    together += answer + "\n\n"


judge = f"""You are judging a competition between {len(competitors)} competitors.
Each model has been given this question:

{question}

Your job is to evaluate each response for clarity and strength of argument, and rank them in order of best to worst.
Respond with JSON, and only JSON, with the following format:
{{"results": ["best competitor number", "second best competitor number", "third best competitor number", ...]}}

Here are the responses from each competitor:

{together}

Now respond with the JSON with the ranked order of the competitors, nothing else. Do not include markdown formatting or code blocks."""

judge_messages = [{"role": "user", "content": judge}]

# Judgement time!

openai = OpenAI()
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=judge_messages,
)
results = response.choices[0].message.content
print(results)


# OK let's turn this into results!

results_dict = json.loads(results)
ranks = results_dict["results"]
for index, result in enumerate(ranks):
    competitor = competitors[int(result)-1]
    print(f"Rank {index+1}: {competitor}")
