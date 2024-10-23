from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    ChatPromptTemplate
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from config import settings

llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=settings.LLM_TEMPERATURE, api_key=settings.OPENAI_API_KEY)


def get_agent(tools: list[BaseTool]) -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        # SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='You are a helpful assistant')),
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def run_agent(agent: AgentExecutor, chat_history: list[BaseMessage]) -> AIMessage:
    result_dict = agent.invoke({"chat_history": chat_history})
    return AIMessage(content=result_dict["output"])
