def summarise_individual_chats(conversation, elder_profile, client, model="gpt-4"):
    PROMPT = f"""
    Imagine you are an AI that has been talking to the following clinical trial participant:

    {elder_profile}

    the conversation went as follows:

    {conversation}

    Generate a summary of the patient's symptoms and cognitive state using the following format:

    Summary cognitive state:

    1. memory defects: [summary of memory defects]
    2. verbal acuity: [summary of verbal acuity]
    3. spoken clarity: [summary of spoken clarity]
    4. repetitiveness of conversation [summary of repetitiveness of conversation]
    5. attention: [summary of attention]
    6. reasoning: [summary of reasoning]
    7. prospective memory: [summary of prospective memory]
    8. motility: [summary of motility]

    If there are no symptoms for a particular category, please write "no symptoms".
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
    PROMPT = "You have the following list of lists of summaries of cognitive state below of a patient's conversations with an assistant.\n"
    PROMPT += "\nCan you aggregate them into one total list of the same format?\n"
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

    return (symptoms.choices[0].message.content,)


def separate_conversations(dialogue):
    """
    Separates the conversations based on two consecutive turns by 'Nora'.
    """
    conversations = []
    current_conversation = []
    prev_speaker = None

    for line in dialogue:
        speaker = line.split(":")[0]
        if speaker == "AI" and prev_speaker == "AI":
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
