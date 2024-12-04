# Talk to your favourite character
AI agents that talk to you like fictional characters or celebrities.   
Interact with AI agents that simulate conversations with fictional characters, historical figures, or celebrities.

# How to talk
1. Create `.env` file like `.env.example`
2. Run any `.py` file from the directory you like. Every directory is divided by category  
To run file you can type 
```bash
python <directory>/<filename>.py
```
Example: 
```bash
python fictional/heisenberg_walter_white.py
```

If you face error `No module named 'config'`, run this:
```bash
PYTHONPATH=. python <directory>/<filename>.py
```


## Categories
- **fictional**: Characters from movies, books, or TV shows.
- **historical**: Famous historical figures.
- **celebrity**: Interact with simulated versions of well-known people.

## Create you own agents or just use llm run
### Example of agent
```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from utils.common import run_agent, get_agent

@tool
def some_tool(some_param: str) -> str:
    """Tool description"""
    return "response"


character_name = ""
character_description = ""
def main():
    chat_history = [SystemMessage(content=character_description)]
    agent = get_agent([some_tool])
    while True:
        input_msg = input("You: ").encode('utf-8').decode('utf-8')
        chat_history.append(HumanMessage(content=input_msg))
        agent_output = run_agent(agent, chat_history)
        print(character_name + ': ' + agent_output.content)
        chat_history.append(AIMessage(content=agent_output.content))


if __name__ == "__main__":
    main()
```
### Example of llm run
```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from utils.common import run_llm

character_name = ""
character_description = ""
def main():
    chat_history = [SystemMessage(content=character_description)]
    while True:
        input_msg = input("You: ").encode('utf-8').decode('utf-8')
        chat_history.append(HumanMessage(content=input_msg))
        llm_output = run_llm(chat_history)
        print(character_name + ': ' + llm_output.content)
        chat_history.append(AIMessage(content=llm_output.content))


if __name__ == "__main__":
    main()
```
