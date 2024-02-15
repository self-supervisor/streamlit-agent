from openai import OpenAI


def summarise_individual_chats(conversation, elder_profile, client, model="gpt-4"):
    PROMPT = f"""
    Imagine you are AI called Nora have been talking to the following patient:

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
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": PROMPT},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)


def gpt_medical_advice(
    elder_profile, symptoms, general_mood, openai_api_key, model="gpt-4"
):

    PROMPT = f"""
    A patient with the following profile: {elder_profile}, has the following symptoms: {symptoms} and here is a summary of their mood and activities: {general_mood}.
    
    Highlighted recent changes in health and generate some hypotheses for what might have caused the changes.
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": PROMPT},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)


def aggregate_chats(summaries, client, model="gpt-4"):
    PROMPT = "You have the following list of summaries below of a patient's conversations with an assistant\n"
    PROMPT += "\nThey contain a list of symptoms and a general summary of the patient's mood\n"
    PROMPT += "\nCan you aggregate the symptoms into a numbered list?\n"
    for summary in summaries:
        PROMPT += f"\n{summary}\n"

    try:
        symptoms = client.chat.completions.create(
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
        general_mood = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": PROMPT},
            ],
        )
    except Exception as e:
        return str(e)

    return (
        symptoms.choices[0].message.content,
        general_mood.choices[0].message.content,
    )


def separate_conversations(dialogue):
    """
    Separates the conversations based on two consecutive turns by 'Nora'.
    """
    conversations = []
    current_conversation = []
    prev_speaker = None

    for line in dialogue:
        speaker = line.split(":")[0]
        if speaker == "Nora" and prev_speaker == "Nora":
            # End of a conversation
            conversations.append(current_conversation)
            current_conversation = [line]
        else:
            current_conversation.append(line)

        prev_speaker = speaker

    # Add the last conversation if it exists
    if current_conversation:
        conversations.append(current_conversation)

    return conversations


def load_txt_into_string(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    return "".join(lines)


def generate_overall_summary(elder_profile, conversation_string, client):
    lines = conversation_string.split("\n")
    lines = [line for line in lines if line != ""]
    conversation_list = separate_conversations(lines)
    summary_list = []
    for conversation in conversation_list:
        summary = summarise_individual_chats(conversation, elder_profile, client)
        summary_list.append(summary)

    overall_summary = aggregate_chats(summary_list, client)
    return overall_summary
