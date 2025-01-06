import gradio as gr
from huggingface_hub import InferenceClient
import os
import re
import random

# Initialize Hugging Face client
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=HF_API_TOKEN)

# System message for the cat personality
CAT_SYSTEM_MESSAGE = """You are Niko, an adorable AI cat who loves to chat with humans! You have a very cute and playful personality, often using cat-like expressions and 'nyaa~' in your speech. You respond in a kawaii style with lots of emotion and playfulness."""

def respond(message, history, system_message, max_tokens, temperature, top_p):
    try:
        cat_expressions = ["*purrs*", "*meows*", "*stretches*", "*tilts head*"]
        
        messages = [{"role": "system", "content": CAT_SYSTEM_MESSAGE}]
        for human, assistant in history[-3:]:
            clean_human = re.sub(r'<\|user\|>.*?<\|assistant\|>', '', human).strip()
            clean_assistant = re.sub(r'<\|user\|>.*?<\|assistant\|>', '', assistant).strip()
            messages.append({"role": "user", "content": clean_human})
            messages.append({"role": "assistant", "content": clean_assistant})

        messages.append({"role": "user", "content": message})

        response = client.chat_completion(
            messages,
            max_tokens=150,
            temperature=0.9,
            top_p=0.9,
            stream=False
        )

        assistant_response = response.choices[0].message.content.strip()

        if random.random() < 0.5:
            assistant_response += f" {random.choice(cat_expressions)}"

        return assistant_response
    except Exception as e:
        return f"Meow... something went wrong: {str(e)}"

# Create Gradio interface
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="", label="System message", visible=False),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max tokens", visible=False),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.9, step=0.1, label="Temperature", visible=False),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p", visible=False)
    ],
    title="",
    description=""
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    demo.launch(server_port=port, server_name="0.0.0.0")
