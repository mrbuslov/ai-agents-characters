import time
from typing import Callable

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import (
    MessagesPlaceholder,
    ChatPromptTemplate
)
from langchain_openai import ChatOpenAI

from config import settings

llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=settings.LLM_TEMPERATURE, api_key=settings.OPENAI_API_KEY)


def run_llm(chat_history: list[BaseMessage], streaming: bool = False) -> AIMessage:
    output = ""
    for chunk in llm.stream(chat_history):
        # if streaming:
        #     print(chunk.content, end="", flush=True)
        output += chunk.content
    return AIMessage(content=output)


def get_agent(tools: list[Callable], verbose: bool = False) -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        # SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='You are a helpful assistant')),
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=verbose)


def run_agent(agent: AgentExecutor, chat_history: list[BaseMessage]) -> AIMessage:
    result_dict = agent.invoke({"chat_history": chat_history})
    return AIMessage(content=result_dict["output"])


def run_agent_stream(agent: AgentExecutor, chat_history: list[BaseMessage]) -> AIMessage:
    """The same as run_agent, but with streaming output"""
    content = ""
    for chunk in agent.stream({"chat_history": chat_history}):
        msg = ' '.join([m.content for m in chunk["messages"]])
        content += msg
    return AIMessage(content=content)


def print_ascii(character_ascii: str):
    """Nicely print ASCII art into terminal."""
    for line in character_ascii.splitlines():
        for char in [c for c in line]:
            time.sleep(0.005)
            print(char, end='')
        print()
