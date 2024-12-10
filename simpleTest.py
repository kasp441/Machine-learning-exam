import autogen
import time

ASSISTANT_AGENT_MESSAGE = """
You are a helpful assistant. Provide precise and concise answers to user queries. Use tools when necessary and prompt the tool agent to use them.
"""

TOOL_AGENT_MESSAGE = """
You are a tool agent. Execute the functions requested by the assistant agent accurately and efficiently.
"""

config_list_llama = [{
    "model": "llama3.2:latest",
    "api_type": "ollama",
    "client_host": "http://localhost:11434/",
}]


# Actual LLM config file for the assistants
llm_config_llama= {
    "seed": 42,
    "config_list": config_list_llama,
    "temperature": 0,
}

def InitAgent():
    return autogen.ConversableAgent(
        system_message=ASSISTANT_AGENT_MESSAGE,
        llm_config=llm_config_llama,
        name="assistant",
        max_consecutive_auto_reply=0,
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


assistant_agent = InitAgent()
user_agent = initProxy()

def main():
    while True:
        user_input = input("Enter a query: ")
        if user_input == "TERMINATE":
            break
        user_message = {"content": user_input, "role": "user"}
        user_agent.send(recipient=assistant_agent, message=user_message)
        time.sleep(1)
        assistant_reply = assistant_agent.generate_reply(messages=[user_message], sender=user_agent)
        print(assistant_reply)
        time.sleep(1)

# maincheck
if __name__ == "__main__":
    main()