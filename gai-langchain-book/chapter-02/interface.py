import os
from dotenv import load_dotenv

from langchain_openai import OpenAI
from langchain_google_genai import GoogleGenerativeAI


## Load env variables
load_dotenv()

## Set  LLM keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")


def openai_model_respose(question):

    ## Initialize openai model
    openai_llm = OpenAI()

    respose = openai_llm.invoke(question)
    return respose

def gemini_model_respose(question):

    ## initalize gemini model

    gemini_llm = GoogleGenerativeAI(model="gemini-2.5-pro")

    respose = gemini_llm.invoke(question)
    return respose

def main():


    question = "Expain LLM's in 20 words"
    print("******** OpenAi LLM's Response ********")
    print(openai_model_respose(question))

    print("******** Gemini LLM's Response ********")
    print(gemini_model_respose(question))


if __name__ == "__main__":
    main()
