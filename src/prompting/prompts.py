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
                template="""You will be given a short text (around 400 words) with medicine-related information.

Your task is to:

Read the text carefully.
Write a summary of the text. Your summary should:
Convey the most important information in the text, as if you are trying to inform another person about what you just read.
Contain at least 100 words.
We expect high quality summaries and will manually inspect some of them\n\nText: {text}""",
            )
        )
        return [human_message]