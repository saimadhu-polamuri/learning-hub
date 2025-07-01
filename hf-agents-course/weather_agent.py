import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

load_dotenv()

hf_token = os.getenv("HF_TOKEN")


def main():

    # client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct")
    client = InferenceClient(os.getenv("LLAMA_ENDPOINT_URL"))

    # print(text_generation(client))
    # print(chat_method(client))


def weather_agent(client):

    messages = [
        {"role": "system", "content": os.getenv(SYSTEM_PROMPT)},
        {"role": "user", "content": "What's the weather in London ?"},
        ]

    tokenizer = AutoTokenizer.from_pretrained(os.getenv(PRETRAIN_MODEL))
    tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    output = client.text_generation(prompt,max_new_tokens=200,)

    print(output)

def text_generation(client):


    output = client.text_generation(
        "The Capital of France Is",
        max_new_tokens = 100,
    )

    prompt="""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

    The capital of france is<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """
    output = client.text_generation(
        prompt,
        max_new_tokens=100,
    )

    return output

def chat_method(client):

    output = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "The capital of france is"},],
    stream=False,
    max_tokens=1024,)

    return output.choices[0].message.content



if __name__ == "__main__":
    main()
