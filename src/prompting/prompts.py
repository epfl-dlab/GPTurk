from langchain import LLMChain
from langchain.chat_models import ChatOpenAI

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import numpy as np

from langchain.schema import HumanMessage, AIMessage, SystemMessage

from langchain.chains import SimpleSequentialChain, SequentialChain

from dotenv import load_dotenv

load_dotenv()


class DataTemplates:
    """Class for storing the templates for the different generation tasks."""

    def medicine_abstract_prompt(self) -> ChatPromptTemplate:
        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["text"],
                template="""Read a 400-word text with medicine-related information and write a 100-word summary.\n\nText: {text}""",
            )
        )
        return [human_message]