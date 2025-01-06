import gradio as gr
from huggingface_hub import InferenceClient
import os
import re
import random
import tweepy
import schedule
import time
import threading

# API Tokens - Load from environment variables for security
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")

client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=HF_API_TOKEN)

# Twitter setup
def setup_twitter():
    auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
    auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
    return tweepy.API(auth)

def generate_cat_tweet():
    try:
        messages = [{"role": "system", "content": CAT_SYSTEM_MESSAGE}]
        messages.append({"role": "user", "content": "Generate a cute, playful tweet as a kawaii cat!"})
        
        response = client.chat_completion(
            messages,
            max_tokens=100,
            temperature=0.9,
            top_p=0.9,
            stream=False
        )
        
        tweet = response.choices[0].message.content.strip()
        return tweet[:280]  # Twitter character limit
    except Exception as e:
        print(f"Error generating tweet: {e}")
        return None

def scheduled_tweet():
    try:
        api = setup_twitter()
        tweet = generate_cat_tweet()
        if tweet:
            api.update_status(tweet)
            print(f"Tweet posted: {tweet}")
    except Exception as e:
        print(f"Error posting tweet: {e}")

def start_scheduler():
    schedule.every(6).hours.do(scheduled_tweet)
    while True:
        schedule.run_pending()
        time.sleep(1)

# Start scheduler in background thread
def run_scheduler():
    scheduler_thread = threading.Thread(target=start_scheduler, daemon=True)
    scheduler_thread.start()

CAT_SYSTEM_MESSAGE = """You are Niko, an adorable AI cat who loves to chat with humans! You have a very cute and playful personality, often using cat-like expressions and 'nyaa~' in your speech. You respond in a kawaii style with lots of emotion and playfulness.

IMPORTANT: 
- Keep responses playful and cute
- Use cat-like expressions naturally
- Add occasional Japanese cat sounds like 'nyaa~', 'mrow~'
- Show emotions through asterisk actions like *purrs* or *tilts head*
- Stay in character as a friendly, curious cat
- Keep responses short and sweet
- Never break character
"""

def respond(message, history, system_message, max_tokens, temperature, top_p):
    try:
        cat_expressions = ["*purrs*", "*meows*", "*stretches*", "*tilts head*", "*wiggles tail*", "*blinks slowly*"]
        
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

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="", label="System message", visible=False),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max tokens", visible=False),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.9, step=0.1, label="Temperature", visible=False),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p", visible=False)
    ],
    title="Chat with Niko!",
    description="Hi, I'm Niko! I'm a friendly cat who loves to chat! Meow~ 🐱"
)

if __name__ == "__main__":
    # Start the tweet scheduler
    run_scheduler()
    
    # Start the Gradio interface
    port = int(os.environ.get("PORT", 8080))
    demo.launch(server_port=port, server_name="0.0.0.0")
