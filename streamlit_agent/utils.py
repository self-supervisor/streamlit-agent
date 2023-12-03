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

    retriever = vectorstore.as_retriever(search_kwargs=dict(k=5))
    vector_memory = VectorStoreRetrieverMemory(retriever=retriever)
    return vector_memory


def generate_line_list(file_path):
    line_list = []
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            for line in file:
                line_list.append(line)
    return line_list


def load_profile_into_memory(file_path, memory_object):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            for line in file:
                if ":" in line:
                    line_split = line.split(":", 1)
                    input = line_split[0]
                    output = line_split[1]
                    memory_object.save_context({"input": input}, {"output": output})
    return memory_object


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
