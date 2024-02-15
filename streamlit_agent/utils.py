import faiss
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
import os


def setup_vector_db(openai_api_key):
    from langchain.memory import VectorStoreRetrieverMemory

    embedding_size = 1536  # Dimensions of the OpenAIEmbeddings
    index = faiss.IndexFlatL2(embedding_size)
    embedding_fn = OpenAIEmbeddings(openai_api_key=openai_api_key).embed_query
    vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})

    retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
    vector_memory = VectorStoreRetrieverMemory(retriever=retriever)
    return vector_memory


def load_profile_into_memory(file_path, memory_object):
    line_list = []
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            for line in file:
                line_list.append(line)
                if ":" in line:
                    line_split = line.split(":", 1)
                    input = line_split[0]
                    output = line_split[1]
                    memory_object.save_context({"input": input}, {"output": output})
    return memory_object, line_list


def generate_basic_profile_str(line_list):
    basic_profile = ""
    basic_profile += "**Profile:**\n"
    for line in line_list:
        if "Patient Profile" in line or not line.strip():
            continue
        if ":" not in line:
            # Bold the line using Markdown syntax
            basic_profile += "\n **" + line[:-1] + "**" + "\n"
        else:
            basic_profile += "\n * " + line
    return basic_profile


def load_memory(file_path, memory_object):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            chat_input, chat_output = None, None
            for line in file:
                if line.startswith("AI Assistant:"):
                    # Check if there is a previous human input that hasn't been paired yet
                    if chat_input is not None and chat_output is not None:
                        memory_object.save_context(
                            {"input": chat_input}, {"output": chat_output}
                        )
                        chat_output = None  # Reset chat_output for the new pair

                    chat_input = line.split(":", 1)[1].strip().strip('"')
                elif line.startswith("Patient:"):
                    chat_output = line.split(":", 1)[1].strip().strip('"')

                    # Save the pair of input and output to memory
                    if chat_input is not None:
                        memory_object.save_context(
                            {"input": chat_input}, {"output": chat_output}
                        )
                        chat_input, chat_output = None, None  # Reset for the next pair

            # Handle any remaining pair at the end of the file
            if chat_input is not None and chat_output is not None:
                memory_object.save_context(
                    {"input": chat_input}, {"output": chat_output}
                )

    return memory_object


def format_to_markdown(text):
    # Split the text into lines
    lines = text.split("\n")

    # Initialize an empty list to hold the formatted lines
    formatted_lines = []

    for line in lines:
        if line.strip() == "":
            # Skip empty lines
            continue
        elif ":" in line:
            # Convert lines with ':' into bullet points
            formatted_lines.append(f"- **{line}**")
        else:
            # Convert other lines into headers
            formatted_lines.append(f"#### {line}")

    # Join the formatted lines into a single string
    return "\n\n".join(formatted_lines)


def format_to_markdown_LLM(text, client):
    PROMPT = f"Convert the following text to pretty formatted markdown:\n\n{text}"
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": PROMPT},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)
