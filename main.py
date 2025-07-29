from agents import Agent, Runner, AsyncOpenAI, trace, OpenAIChatCompletionsModel, TResponseInputItem
from agents.run import RunConfig
from dotenv import load_dotenv
import os
import asyncio, uuid

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is present
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Gemini configuration for OpenAI-compatible SDK
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
)

# Define translation agents
spanish_agent = Agent(
    name="spanish_agent",
    instructions="You are a Spanish speaker. Translate any task into Spanish.",
)

english_agent = Agent(
    name="english_agent",
    instructions="You are an English speaker. Translate any task into English.",
)

french_agent = Agent(
    name="french_agent",
    instructions="You are a French speaker. Translate any task into French.",
)

# Orchestrator agent
triage_agent = Agent(
    name="triage_agent",
    instructions="""
    You are an orchestrator agent that decides which agent should handle the task 
    based on the language. Choose from: spanish_agent, english_agent, or french_agent.
    """,
    handoffs=[
        spanish_agent,
        english_agent,
        french_agent
    ]
)

async def main():
    conversation_id = str(uuid.uuid4().hex[:16])

    msg = input("Hi! We speak French, Spanish and English. How can I help? ")
    agent = triage_agent
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    while True:
        with trace("Routing example", group_id=conversation_id):
            result = await Runner.run(
                agent,
                input=inputs,
                run_config=config
            )

        print(f"\nAssistant: ({result.last_agent.name})")
        print(f"Agent Output: {result.final_output}\n")

        user_msg = input("Enter a message (or 'exit' to quit): ")
        if user_msg.strip().lower() == "exit":
            break

        # Update conversation history
        inputs = result.to_input_list()
        inputs.append({"content": user_msg, "role": "user"})

        # You can choose to always re-triage (or stick with the last agent)
        agent = triage_agent  # Always re-evaluate language
        # agent = result.last_agent  # Uncomment this if you want to stick to the same agent

if __name__ == "__main__":
    asyncio.run(main())
