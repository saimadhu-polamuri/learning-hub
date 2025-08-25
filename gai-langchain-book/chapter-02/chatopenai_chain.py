
import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

## load keys
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def main():

    ## Creating the template
    template = ChatPromptTemplate.from_messages([
    ("system", "You are experienced programmer and mathematica analyst"),
    ("user", "{problem}")])

    ## Initialize the openai chat model

    chat = ChatOpenAI(model_name = "gpt-4o")

    ## Chain the template with chat

    chain = template | chat

    # Invoke the chain
    respose = chain.invoke({"problem": "Write a even odd python program"})

    print(respose.content)


if __name__ == "__main__":
    main()
