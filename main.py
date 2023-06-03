import os

from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from serpapi import GoogleSearch

# it's used internally by OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY is not None, "Please set OPENAI_API_KEY environment variable"


def get_socks_name(llm):
    text = "What would be a good company name for a company that makes colorful socks?"
    return llm(text)


def get_joke_and_poem(llm, num_of_each=1):
    return llm(["Tell me a joke", "Tell me a poem"] * num_of_each)


def get_company_name_prompt_template(product="colorful socks"):
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    print(prompt.format(product=product))


def get_serpapi_key():
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
    assert SERPAPI_API_KEY is not None, "Please set SERP_API_KEY environment variable"
    return SERPAPI_API_KEY


def test_serpapi_api_key():
    SERPAPI_API_KEY = get_serpapi_key()
    search = GoogleSearch({"q": "coffee", "location": "Austin,Texas", "api_key": SERPAPI_API_KEY})
    result = search.get_dict()
    return result


def get_demo_conversation_chain(convo_msg_starter="Hi there!"):
    from langchain import ConversationChain

    llm = OpenAI(temperature=0)
    conversation = ConversationChain(llm=llm, verbose=True)
    output = conversation.predict(input=convo_msg_starter)
    print(output)
    return output


def get_message_completions_from_chat_model():
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage

    chat = ChatOpenAI(temperature=0)

    # basic example
    # chat([HumanMessage(content="Translate this sentence from English to French. I love programming.")])
    # -> AIMessage(content="J'aime programmer.", additional_kwargs={})

    # example with multiple messages
    # messages = [
    #     SystemMessage(content="You are a helpful assistant that translates English to French."),
    #     HumanMessage(content="I love programming."),
    # ]
    # chat(messages)
    # -> AIMessage(content="J'aime programmer.", additional_kwargs={})

    # example with multiple pairs of messages
    batch_messages = [
        [
            SystemMessage(content="You are a helpful assistant that translates English to French."),
            HumanMessage(content="I love programming."),
        ],
        [
            SystemMessage(content="You are a helpful assistant that translates English to French."),
            HumanMessage(content="I love artificial intelligence."),
        ],
    ]
    result = chat.generate(batch_messages)


def get_chat_prompt_template_examples():
    """
    https://python.langchain.com/en/latest/getting_started/getting_started.html#chat-prompt-templates
    """
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )

    chat = ChatOpenAI(temperature=0)

    template = "You are a helpful assistant that translates {input_language} to {output_language}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # get a chat completion from the formatted messages
    chat(
        chat_prompt.format_prompt(
            input_language="English", output_language="French", text="I love programming."
        ).to_messages()
    )
    # -> AIMessage(content="J'aime programmer.", additional_kwargs={})


def sample_serpapi_search_results(
    sample_query="",
):
    SERPAPI_API_KEY = get_serpapi_key()
    sample_query = (
        sample_query
        if sample_query
        else (
            "What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?"
        )
    )
    llm = OpenAI(temperature=0)
    # llm-math needs the llm
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return agent.run(sample_query)


if __name__ == "__main__":
    llm = OpenAI(temperature=0.9)
    # get_socks_name(llm)
    # get_joke_and_poem(llm)
    # get_company_name_prompt_template()
    # test_serpapi_api_key()
    # sample_serpapi_search_results()
    # get_demo_conversation_chain()
    # get_message_completions_from_chat_model()
