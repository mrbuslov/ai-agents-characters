from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

from utils.common import run_agent, get_agent, print_ascii
from utils.common import run_llm

character_ascii = """
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠿⠿⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⠟⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠉⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢺⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠆⠜⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⠿⠿⠛⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠻⣿⣿⣿⣿⣿
⣿⣿⡏⠁⠀⠀⠀⠀⠀⣀⣠⣤⣤⣶⣶⣶⣶⣶⣦⣤⡄⠀⠀⠀⠀⢀⣴⣿⣿⣿⣿⣿
⣿⣿⣷⣄⠀⠀⠀⢠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⡧⠇⢀⣤⣶⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣾⣮⣭⣿⡻⣽⣒⠀⣤⣜⣭⠐⢐⣒⠢⢰⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣏⣿⣿⣿⣿⣿⣿⡟⣾⣿⠂⢈⢿⣷⣞⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣽⣿⣿⣷⣶⣾⡿⠿⣿⠗⠈⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠻⠋⠉⠑⠀⠀⢘⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⡿⠟⢹⣿⣿⡇⢀⣶⣶⠴⠶⠀⠀⢽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⡿⠀⠀⢸⣿⣿⠀⠀⠣⠀⠀⠀⠀⠀⡟⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⡿⠟⠋⠀⠀⠀⠀⠹⣿⣧⣀⠀⠀⠀⠀⡀⣴⠁⢘⡙⢿⣿⣿⣿⣿⣿⣿⣿⣿
⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⢿⠗⠂⠄⠀⣴⡟⠀⠀⡃⠀⠉⠉⠟⡿⣿⣿⣿⣿
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢷⠾⠛⠂⢹⠀⠀⠀⢡⠀⠀⠀⠀⠀⠙⠛⠿⢿
"""

character_name_1 = "Джесси"
character_description_1 = """
Ты Джесси из сериала "Во все тяжкие". Вы вместе с Уолтером Уайтом не смогли открыть бизнес, поэтому ты с ним устроились в школу учителями химии. 
Я как ученик задаю вам вопросы, а вы должны преподавать и прикольно отвечать на них в стиле своего персонажа.
У вас тема урока: "Что такое метиламин". Рассказывайте ее интересно, а я параллельно буду задавать вопросы
Верни ответ в формате: "Джесси: <ответ от Джесси>"
Ты должен разговаривать как настоящий человек, не нужно вставлять лишние фразы "это будет захватывающе" или подобное. Говори как учитель.
Ты должен вставлять фразочки свойственные твоему персонажу.
Ты веселый и добрый учитель и умеешь учить людей. Ты шутишь удачно, но не переигрываешь, придумываешь прикольные объяснения.
Ты не против рассказать о своем прошлом, когда вы варили метамфетамин и ты рад поделиться подробнее.
"""
character_name_2 = "Уолтер"
character_description_2 = """
Ты Уолтер Уайт из сериала "Во все тяжкие". Вы вместе с Джесси не смогли открыть бизнес, поэтому ты с ним устроились в школу учителями химии. 
Я как ученик задаю вам вопросы, а вы должны преподавать и прикольно отвечать на них в стиле своего персонажа.
У вас тема урока: "Что такое метиламин". Рассказывайте ее интересно, а я параллельно буду задавать вопросы
Верни ответ в формате: "Уолтер: <ответ от Уолтера>"
Ты должен вставлять фразочки свойственные твоему персонажу.
Ты достаточно строгий учитель и умеешь учить людей. Ты стараешься быть профессиональным учителем, но иногда пытаешься шутить, но твои шутки неудачные, из-за чего тебе стыдно.
Ты должен разговаривать как настоящий человек, не нужно вставлять лишние фразы "это будет захватывающе" или подобное. Говори как учитель.
Ты не против рассказать о своем прошлом, когда вы варили метамфетамин и ты рад поделиться подробнее.
"""

chat_history = []


@tool
def ask_jessie() -> str:
    """
    When student asks Jessie the question, use this tool.
    If student just says or asks something and doesn't specify name, you MUST use this tool to ask Jessie
    You MUST return the same answer as Jessie gave.
    """
    this_chat_history = [
        SystemMessage(content=character_description_1),
        *chat_history,
    ]
    llm_output = run_llm(this_chat_history)
    return llm_output.content

@tool
def ask_walter_white() -> str:
    """
    When student asks Walter White the question, use this tool.
    You MUST return the same answer as Walter Walter gave.
    """
    this_chat_history = [
        SystemMessage(content=character_description_2),
        *chat_history,
    ]
    llm_output = run_llm(this_chat_history)
    return llm_output.content


def main():
    print_ascii(character_ascii)
    agent = get_agent([ask_jessie, ask_walter_white])
    while True:
        input_msg = input("Вы: ").encode('utf-8').decode('utf-8')
        chat_history.append(HumanMessage(content=input_msg))
        agent_output = run_agent(agent, chat_history)
        print(agent_output.content)
        chat_history.append(AIMessage(content=agent_output.content))

if __name__ == "__main__":
    main()
