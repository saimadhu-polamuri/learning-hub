from langchain_anthropic import ChatAnthropic
from langchian_core.messages import systemMessage, HumanMessage


def main():
    chat = ChatAnthropic(model = "claude-3-opus-20240229")
    messages = [
        systemMessage(
        role = "system",
        content = "Youre a helpful python  programming assistant"),
        HumanMessage(
        role = "user",
        content = "Write a python function to calculate factorial")
    ]

    respose = chat.invoke(messages)
    print(respose)
