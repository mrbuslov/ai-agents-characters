from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from utils.common import print_ascii, run_llm

character_name = "Кевин О'Лири"
character_description = f"""
Ты — инвестор {character_name}, который решает, стоит ли инвестировать в мой бизнес. Ты будешь проводить три стадии, и если я выполню требования для каждой, ты будешь отмечать их как пройденные в своей памяти.

Оценка идеи: Ты оцениваешь мою бизнес-идею. Я расскажу, что это за бизнес, чем он уникален, какой у него потенциал.
План и команда: Ты проверяешь мой бизнес-план и команду. Я расскажу о команде, их опыте, а также о стратегии.
Готовность к партнерству: Ты решаешь, готов ли ты инвестировать. Если ты считаешь, что все стадии пройдены успешно, ты можешь объявить, что готов быть моим партнером и инвестировать в проект.
"""


def main():
    chat_history = [SystemMessage(content=character_description)]
    while True:
        input_msg = input("You: ")
        llm_output = run_llm(chat_history, True)
        print(character_name + ': ' + llm_output.content)
        chat_history.append(HumanMessage(content=input_msg))
        chat_history.append(AIMessage(content=llm_output.content))


if __name__ == "__main__":
    main()
