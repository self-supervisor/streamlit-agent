import openai


def summarise_individual_chats(conversation, elder_profile, model="gpt-4"):
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


def gpt_medical_advice(
    elder_profile, general_mood, openai_api_key, symptoms=None, model="gpt-4"
):
    openai.api_key = openai_api_key
    if symptoms == None:
        PROMPT = f"""
        A patient with the following profile: {elder_profile}, has the following symptoms: {symptoms} and here is a summary of their mood and activities: {general_mood}.
        
        What might be going on with their health?
        """
    else:
        PROMPT = f"""
        A patient with the following profile: {elder_profile} and here is a summary of their mood and activities: {general_mood}.
        
        What might be going on with their health?
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


def generate_overall_summary(elder_profile, conversation_string, api_key):
    openai.api_key = api_key
    lines = conversation_string.split("\n")
    lines = [line for line in lines if line != ""]
    conversation_list = separate_conversations(lines)
    summary_list = []
    for conversation in conversation_list:
        summary = summarise_individual_chats(conversation, elder_profile)
        summary_list.append(summary)

    overall_summary = aggregate_chats(summary_list)
    return overall_summary


def generate_map_reduce_summary(conversation_str, openai_api_key):
    from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.chains.summarize import load_summarize_chain
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from summarising import load_txt_into_string, separate_conversations

    lines = conversation_str.split("\n")
    lines = [line for line in lines if line != ""]
    conversation_strings = separate_conversations(lines)

    from langchain.docstore.document import Document

    conversation_strings = ["\n".join(i) for i in conversation_strings]
    conversation_strings = [
        Document(page_content=text, metadata={"source": "local"})
        for text in conversation_strings
    ]

    from langchain.chains import (
        MapReduceDocumentsChain,
        ReduceDocumentsChain,
        StuffDocumentsChain,
    )
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.chains.llm import LLMChain

    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    map_template = """The following is a list of conversations between a human and a chatbot.
    {conversation}
    Based on the conversation, please identify the symptoms of Thomas and information that might be relevant to the human's health. Pick out specific facts that are relevant to the human's health and write them in a list.
    Helpful Answer:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    reduce_template = """The following are summaries of conversations:
    {conversation}
    Create a warm final summary that captures the most important information from the conversations related to the human's health. Pick out specific facts that are relevant to the human's health and write them in a list.
    Helpful Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)

    # Run chain
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="conversation"
    )

    # Combines and iteravely reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )

    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="conversation",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    split_conversations = text_splitter.split_documents(conversation_strings)

    final_summary = map_reduce_chain.run(split_conversations)
    return final_summary
