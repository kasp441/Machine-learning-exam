import speech_recognition as sr
import json
import autogen
import time
from autogen import register_function

recognizer = sr.Recognizer()

ASSISTANT_AGENT_MESSAGE = """
You are a helpful assistant. Provide precise and concise answers to user queries. Use tools when necessary and prompt the tool agent to use them and answer with the result.
"""

TOOL_AGENT_MESSAGE = """
You are a tool agent. Execute the functions requested by the assistant agent accurately and efficiently.
"""

config_list_llama = [
    {
        "model": "llama3.2:latest",
        "api_type": "ollama",
        "client_host": "http://localhost:11434/",
    }
]

# Actual LLM config file for the assistants
llm_config_llama = {
    "seed": 42,
    "config_list": config_list_llama,
    "temperature": 0,
}

config_list_qwen = [
    {
        "model": "qwen2.5-coder",
        "api_type": "ollama",
        "client_host": "http://localhost:11434/",
        }
        ]

# Actual LLM config file for the assistants
llm_config_qwen = {
    "seed": 22,
    "config_list": config_list_qwen,
    "temperature": 1,
}

def InitAgent():
    return autogen.ConversableAgent(
        system_message=ASSISTANT_AGENT_MESSAGE,
        llm_config=llm_config_qwen,
        name="assistant",
        max_consecutive_auto_reply=1,
        human_input_mode="NEVER",
    )

def initProxy():
    return autogen.UserProxyAgent(
        system_message="",
        name="User",
        human_input_mode="NEVER",
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config=False
    )

# Simple tool call to add integers
def add_integers(a: int, b: int) -> int:
    """
    Adds two integers together.

    Parameters:
    a (int): The first integer.
    b (int): The second integer.

    Returns:
    int: The sum of the two integers.

    Example:
    >>> add_integers(2, 3)
    5
    """
    return a + b

# Initialize agents
assistant_agent = InitAgent()
user_agent = initProxy()

register_function(
    add_integers,
    caller=assistant_agent,
    executor=user_agent,
    name="add_integers",
    description="Adds two integers together",
)

# Preload the Vosk model
print("Preloading Vosk model, please wait...")
recognizer.recognize_vosk(sr.AudioData(b'', 16000, 2))  # Dummy audio data to trigger model loading

def callback(recognizer, audio):
    try:
        result_json = recognizer.recognize_vosk(audio)
        result = json.loads(result_json)
        text = result.get("text", "")
        print(f"You: {text}")

        # Send user input to the assistant agent
        user_message = {"content": text, "role": "user"}
        print(f"Sending message to assistant: {user_message}")
        user_agent.send(user_message, assistant_agent, request_reply=True)

        # Generate assistant reply
        assistant_reply = assistant_agent.generate_reply(messages=[user_message], sender=user_agent)
        print(f"Assistant: {assistant_reply}")

    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Speech Recognition service; {e}")

# Initialize the microphone source
microphone = sr.Microphone()

print("Adjusting for ambient noise, please wait...")
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)

print("Listening...")
# Start listening in the background
stop_listening = recognizer.listen_in_background(microphone, callback)

# Keep the script running to continue listening
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Stopping the listener...")
    stop_listening(wait_for_stop=False)
