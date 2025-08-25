import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


## Initialize and add openai keys to the script
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def main():

    ## create the template
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a problem solving assistant"),
        ("user", "{problem}")])

    ## Initialize the chat with reasoning_effort parameter
    chat = ChatOpenAI(
        model_name = "o3-mini",
        reasoning_effort = "high" )

    ## create the chain
    chain = template | chat

    ## Get response from the chain
    response = chain.invoke({"problem": "Calulate the optimal strategy for shortest distance problem"})

    print(response.content)

if __name__ == "__main__":
    main()
