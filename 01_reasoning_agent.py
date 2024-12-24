import os

from autogen import (
    UserProxyAgent,
    ReasoningAgent
)
from dotenv import load_dotenv

load_dotenv()
# Configure the model
config_list = [{"model": "gpt-4", "api_key": os.environ.get("OPENAI_API_KEY")}]


class ThinkerAgent:

    def create_the_reasoner(self):
        # Create a reasoning agent with beam search
        reasoning_agent = ReasoningAgent(
            name="reason_agent",
            llm_config={"config_list": config_list},
            verbose=True,
            reason_config={
                "beam_size": 1,
                "max_depth": 3
            }
        )

        return reasoning_agent

    def create_the_user_proxy(self):
        # Create a user proxy agent
        user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            code_execution_config={"use_docker": False},
            max_consecutive_auto_reply=10,
        )

        return user_proxy

    def initiate_a_chat(self, question):
        # question = "How many r we have in strawberry?"
        user_proxy = self.create_the_user_proxy()
        reasoning_agent = self.create_the_reasoner()
        chat_result = user_proxy.initiate_chat(reasoning_agent, message=question)
        # visualize_tree(reasoning_agent._root)

        return chat_result


if __name__ == "__main__":
    ta = ThinkerAgent()
    question = "How many r we have in strawberry?"
    result = ta.initiate_a_chat(question)
    # print(result)
    chat_history = result.chat_history
    for history in chat_history:
        print("role ", history["role"])
        print("name ", history["name"])
        print("content ", history["content"])
        print("------------------------------")

    print(result.cost)

