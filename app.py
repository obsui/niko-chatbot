import gradio as gr
from huggingface_hub import InferenceClient
import os
import re
import random
import tweepy
import schedule
import time
import threading
from flask import Flask, render_template, jsonify
import requests

app = Flask(__name__)
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=HF_API_TOKEN)

# Twitter Monitoring Setup
def setup_twitter():
    auth = tweepy.OAuthHandler(
        os.getenv("TWITTER_API_KEY"),
        os.getenv("TWITTER_API_SECRET")
    )
    auth.set_access_token(
        os.getenv("TWITTER_ACCESS_TOKEN"),
        os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
    )
    return tweepy.API(auth)

def monitor_twitter():
    try:
        api = setup_twitter()
        class CatStreamListener(tweepy.StreamListener):
            def on_status(self, status):
                if (status.user.verified or status.retweet_count > 100) and \
                   ('cat' in status.text.lower() or '@NikoCat' in status.text):
                    try:
                        response = generate_cat_tweet()
                        if response:
                            api.update_status(
                                status=response,
                                in_reply_to_status_id=status.id
                            )
                    except Exception as e:
                        print(f"Error responding to tweet: {e}")

        stream_listener = CatStreamListener()
        stream = tweepy.Stream(auth=api.auth, listener=stream_listener)
        stream.filter(track=['cat', '@NikoCat'], languages=['en'])
    except Exception as e:
        print(f"Error in Twitter monitor: {e}")
        time.sleep(60)

# Pump.fun API integration
def get_token_holdings(token_address):
    try:
        # Replace with actual Pump.fun API endpoint
        api_url = f"https://api.pump.fun/token/{token_address}/holders"
        response = requests.get(api_url)
        return response.json()
    except Exception as e:
        print(f"Error fetching token holdings: {e}")
        return []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/leaderboard')
def get_leaderboard():
    # Replace with your KIBL token address
    holdings = get_token_holdings('YOUR_KIBL_TOKEN_ADDRESS')
    return jsonify(holdings)

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
    # Start Twitter monitor in background
    twitter_thread = threading.Thread(target=monitor_twitter, daemon=True)
    twitter_thread.start()
    
    # Start the Flask app
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
