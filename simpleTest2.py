from autogen import AssistantAgent, UserProxyAgent

# Your configuration
config_list_llama = [{
    "model": "llama3.2:latest",
    "api_type": "ollama",
    "client_host": "http://localhost:11434/",
}]

# Actual LLM config file for the assistants
llm_config_llama = {
    "seed": 42,
    "config_list": config_list_llama,
    "temperature": 0,
}

# Create an AssistantAgent with the given configuration
assistant = AssistantAgent(
    "assistant", 
    llm_config=llm_config_llama,
    system_message="You are a helpful assistant. Provide precise and concise answers to user queries. never answer with code, just text.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1
)
    

# Create a UserProxyAgent to interact with the assistant
user_proxy = UserProxyAgent("user_proxy")

# Define a simple task for the assistant
user_input = "What is the capital of France?"

# Structure the message as a dictionary
user_message = {"content": user_input, "role": "user"}

# Send the task to the assistant
user_proxy.send(user_message, assistant)

# Generate a reply from the assistant
assistant_reply = assistant.generate_reply(messages=[user_message], sender=user_proxy)

# Print the message sent by the user
print("User Message:", user_message)

# Print the response from the assistant
print("Assistant Reply:", assistant_reply)
