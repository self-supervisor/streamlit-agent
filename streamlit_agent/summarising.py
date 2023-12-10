import openai

# from dotenv import load_dotenv
# import os

# load_dotenv()


def summarise_individual_chats(conversation, elder_profile, model="gpt-4"):
    PROMPT = f"""
    Imagine you have been talking to the following patient:

    {elder_profile}

    the conversation went as follows:

    {conversation}

    Succinctly, list information from the conversation that could indicate a serious health issue that needs to be adressed.

    Use the following template:

    Summary of symptoms:

    1. [summary of symptom 1]

    etc.

    Also, include a summary of the patient's general mood and actvities:

    General summary:

    [summary of general mood]

    Keep the general summary in one small paragraph.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": PROMPT},
            ],
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return str(e)


def aggregate_chats(summaries, model="gpt-4"):
    PROMPT = "You have the following list of summaries below of a patient's conversations with an assistant\n"
    PROMPT += "\nThey contain a list of symptoms and a general summary of the patient's mood\n"
    PROMPT += "\nCan you aggregate the symptoms into a numbered list?\n"
    for summary in summaries:
        PROMPT += f"\n{summary}\n"

    try:
        symptoms = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": PROMPT},
            ],
        )
    except Exception as e:
        return str(e)

    PROMPT = "You have the following list of summaries below of a patient's conversations with an assistant\n"
    PROMPT += "\nThey contain a list of symptoms and a general summary of the patient's mood\n"
    PROMPT += "\nCan you generate a succinct summary of their activities, general health, and mood? Do not make any reccomendations or give opinions, just relay information.\n"
    for summary in summaries:
        PROMPT += f"\n{summary}\n"

    try:
        general_mood = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": PROMPT},
            ],
        )
    except Exception as e:
        return str(e)

    return (
        symptoms.choices[0].message["content"],
        general_mood.choices[0].message["content"],
    )


def separate_conversations(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    conversations = []
    current_conversation = []

    for line in lines:
        if line.strip() == "NEW CONVERSATION":
            if current_conversation:
                conversations.append("".join(current_conversation))
                current_conversation = []
        else:
            current_conversation.append(line)

    if current_conversation:
        conversations.append("".join(current_conversation))

    return conversations


def load_txt_into_string(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    return "".join(lines)


def generate_overall_summary(api_key):
    openai.api_key = api_key
    file_path = "streamlit_agent/conversation_history_extra_long.txt"
    elder_profile = load_txt_into_string("streamlit_agent/elder_profile.txt")
    conversation_list = separate_conversations(file_path)
    summary_list = []
    for conversation in conversation_list:
        summary = summarise_individual_chats(conversation, elder_profile)
        summary_list.append(summary)

    overall_summary = aggregate_chats(summary_list)
    return overall_summary
