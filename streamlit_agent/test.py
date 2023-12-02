import pytest
import os
from dotenv import load_dotenv
import openai


# @pytest.fixture
# def vector_db():
#     from .utils import setup_vector_db

#     load_dotenv()
#     openai.api_key = os.getenv("OPENAI_API_KEY")
#     return setup_vector_db(openai.api_key)


# @pytest.fixture
# def line_list(vector_db):
#     from .utils import load_profile_into_memory

#     _, line_list = load_profile_into_memory(
#         file_path="elder_conversation.txt", memory_object=vector_db
#     )
#     return line_list


# def test_load_profile_into_memory(vector_db):
#     from .utils import load_profile_into_memory

#     _, line_list = load_profile_into_memory(
#         file_path="elder_conversation.txt", memory_object=vector_db
#     )
#     assert len(line_list) > 0


# def test_generate_basic_profile_str(line_list):
#     from .utils import generate_basic_profile_str

#     profile = generate_basic_profile_str(line_list)
#     print("profile: ", profile)


def test_generate_overall_summary():
    from .summarising import generate_overall_summary

    summary = generate_overall_summary()
    print("summary: ", summary)
