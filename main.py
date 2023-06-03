import os
from langchain.llms import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY is not None, "Please set OPENAI_API_KEY environment variable"


def get_socks_name(llm):
    text = "What would be a good company name for a company that makes colorful socks?"
    return llm(text)


def get_joke_and_poem(llm, num_of_each=1):
    return llm(["Tell me a joke", "Tell me a poem"] * num_of_each)


if __name__ == "__main__":
    llm = OpenAI(temperature=0.9)
    # get_socks_name(llm)
    # get_joke_and_poem(llm)
