from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, ConfigurableField

load_dotenv()

"""
See related langchain discussion
[How to correctly identify or test configurable ids when using RunnableLambda and ConfigurableField · langchain-ai/langchain · Discussion #24471](https://github.com/langchain-ai/langchain/discussions/24471) 

This module tests a variation of 
[How to route between sub-chains](https://python.langchain.com/v0.2/docs/how_to/routing/#using-a-custom-function-recommended)
that uses ConfigurableFields; see also [https://python.langchain.com/v0.2/docs/how_to/configure/](https://python.langchain.com/v0.2/docs/how_to/configure/)

While the original version without configuration works, it fails with configuration
"""


def get_llm():
    # substitute the LLM you have creds for
    from langchain_aws import ChatBedrock

    return ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0")


chain = (
    PromptTemplate.from_template(
        """Given the user question below, classify it as either being about `LangChain`, `Anthropic`, or `Other`.
                                
                                Do not respond with more than one word.
                                
                                <question>
                                {question}
                                </question>
                                
                                Classification:"""
    )
    | get_llm()
    | StrOutputParser()
)

LANGCHAIN_DEFAULT_PROMPT = """You are an expert in langchain. \
                        Always answer questions starting with "As Harrison Chase told me". \
                        Respond to the following question:
                        
                        Question: {question}
                        Answer:"""

langchain_chain = (
    PromptTemplate.from_template(LANGCHAIN_DEFAULT_PROMPT).configurable_fields(
        # THIS IS NEW relative to [How to route between sub-chains | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/how_to/routing/#using-a-custom-function-recommended)
        # See also [How to configure runtime chain internals | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/how_to/configure/) for more details
        template=ConfigurableField(id="langchain_prompt", name="LangChain", description="LangChain")
    )
    | get_llm()
)
anthropic_chain = (
    PromptTemplate.from_template(
        """You are an expert in anthropic. \
                                Always answer questions starting with "As Dario Amodei told me". \
                                Respond to the following question:
                                
                                Question: {question}
                                Answer:"""
    )
    | get_llm()
)
general_chain = (
    PromptTemplate.from_template(
        """Respond to the following question:
                                
                                Question: {question}
                                Answer:"""
    )
    | get_llm()
)


def route(info):
    if "anthropic" in info["topic"].lower():
        return anthropic_chain
    elif "langchain" in info["topic"].lower():
        return langchain_chain
    else:
        return general_chain


full_chain = {"topic": chain, "question": lambda x: x["question"]} | RunnableLambda(route)


def test_invoke():
    result = full_chain.invoke({"question": "how do I use Anthropic?"})
    assert "Dario" in result.content
    result = full_chain.invoke({"question": "how do I use LangChain?"})
    assert "Harrison" in result.content

    # okay, so far, so good. Now let's try configuring.
    configurable_id_fields_langchain = [c.id for c in langchain_chain.config_specs]
    assert "langchain_prompt" in configurable_id_fields_langchain

    substitute_string = "Harrison Chase"
    substitute_with = "Nuno Campos"
    substituted_prompt = LANGCHAIN_DEFAULT_PROMPT.replace(substitute_string, substitute_with)
    configured_langchain_chain = langchain_chain.with_config(configurable={"langchain_prompt": substituted_prompt})
    langchain_result = configured_langchain_chain.invoke({"question": "how do I use LangChain?"})
    assert substitute_with in langchain_result.content

    configured_full_chain = full_chain.with_config(configurable={"langchain_prompt": substituted_prompt})
    configured_full_chain_result = configured_full_chain.invoke({"question": "how do I use LangChain?"})
    assert substitute_with in configured_full_chain_result.content

    configurable_id_fields_full_chain = [c.id for c in full_chain.config_specs]
    assert "langchain_prompt" in configurable_id_fields_full_chain
    # E       AssertionError: assert 'langchain_prompt' in []
