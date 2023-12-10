import openai
import time
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_conversation(model, messages):
    try:
        response = openai.ChatCompletion.create(model=model, messages=messages,)
        return response.choices[0].message["content"]
    except Exception as e:
        return str(e)


def write_to_file(file, message):
    with open(file, "a") as f:
        f.write(message + "\n")


conversation_history = [
    {"role": "system", "content": "You are a helpful assistant."},
]

model = "gpt-3.5-turbo-1106"
output_file = "conversation_history.txt"  # Output file name

# Write initial system message to file
write_to_file(output_file, f"System: {conversation_history[0]['content']}")

input_prompt = """I am interested in the following imaginary patient:
    Patient Profile

    Name: Thomas Bennett
    Age: 75
    Location: Greenfield, USA
    Occupation: Retired Military Veteran
    Marital Status: Married
    Children: One daughter (Laura, 50, living in the same city)
    Grandchildren: Two (ages 12 and 15)
    Medical History

    Complicated Cardiac Disease: Including atrial fibrillation, managed with amiodarone.
    Recent Weight Loss: Noticed over the past 8 weeks since starting amiodarone.
    Hypertension: Managed with medication.
    Cholesterol: High cholesterol, controlled with statins.
    Prostate Surgery: Underwent successful prostate surgery 5 years ago.
    Family Medical History

    Father: Passed away at age 80 due to heart failure.
    Mother: Lived to 85, had a history of hypertension.
    Siblings: One brother (72, living with type 2 diabetes); one sister (70, history of breast cancer).
    Lifestyle

    Smoking: Quit smoking 20 years ago.
    Alcohol: Occasional beer or wine.
    Diet: Balanced diet, recently less appetite.
    Physical Activity: Limited due to cardiac condition. Enjoys short walks and light gardening.

    Can you generate an imaginary conversation between an AI assistant called Nora and this patient?
    Nora should ask the patient how they are feeling and talk about their general life also. Like
    a friendly companion that also makes some notes on their health. It is very important that the AI assistant
    does not give any medical advice or make any diagnosis."""
user_inputs = [
    input_prompt,
]
user_inputs += [
    "Great, can you follow on from this and make another conversation?"
] * 50

for user_input in user_inputs:
    conversation_history.append({"role": "user", "content": user_input})
    ai_response = generate_conversation(model, conversation_history)
    print("AI:", ai_response)
    conversation_history.append({"role": "assistant", "content": ai_response})
    write_to_file(output_file, f"AI: {ai_response}")
    time.sleep(120)
import openai
import time
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_conversation(model, messages):
    try:
        response = openai.ChatCompletion.create(model=model, messages=messages,)
        return response.choices[0].message["content"]
    except Exception as e:
        return str(e)


def write_to_file(file, message):
    with open(file, "a") as f:
        f.write(message + "\n")


def read_from_file(file):
    with open(file, "r") as f:
        lines = f.readlines()
    return lines


def parse_conversation(lines):
    history = []
    for line in lines:
        if line.startswith("System:"):
            content = line.replace("System: ", "").strip()
            history.append({"role": "system", "content": content})
        elif line.startswith("AI:"):
            content = line.replace("AI: ", "").strip()
            history.append({"role": "assistant", "content": content})
        elif line.startswith("User:"):
            content = line.replace("User: ", "").strip()
            history.append({"role": "user", "content": content})
    return history


model = "gpt-3.5-turbo-1106"
output_file = "conversation_history.txt"
existing_conversations_file = (
    "existing_conversations.txt"  # File with existing conversations
)

# Read and parse existing conversations
existing_lines = read_from_file(existing_conversations_file)
conversation_history = parse_conversation(existing_lines)

# Add any additional system message to the history
conversation_history.append(
    {"role": "system", "content": "You are a helpful assistant."}
)

input_prompt = "Your input prompt here..."
user_inputs = [input_prompt] + ["Continue the conversation..."] * 50

for i, user_input in enumerate(user_inputs):
    conversation_history.append({"role": "user", "content": user_input})
    ai_response = generate_conversation(model, conversation_history)
    print("AI:", ai_response)
    conversation_history.append({"role": "assistant", "content": ai_response})
    write_to_file(output_file, f"AI: {ai_response}")
    time.sleep(120)
