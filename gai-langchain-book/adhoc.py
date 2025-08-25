import sys

from dotenv import load_dotenv
import os
# print("Python Version {}".format(sys.version))


# load enviroment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

print("Openai API Key: {}".format(openai_api_key))
print("Huggingface API Key: {}".format(huggingface_api_key))
