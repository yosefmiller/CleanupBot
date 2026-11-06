from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler  # Built into LangChain core
from rosa import ROSA

# 1. Create a tiny tracker to catch Ollama's speed stats as they come in
class TokenSpeedTracker(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        try:
            # Look at the final chunk of generation data returned by Ollama
            generation_info = response.generations[0][0].generation_info

            # Extract token counts and durations (Ollama returns duration in nanoseconds)
            output_tokens = generation_info.get("eval_count", 0)
            duration_ns = generation_info.get("eval_duration", 0)

            if duration_ns > 0:
                seconds = duration_ns / 1_000_000_000  # Convert nanoseconds to seconds
                tps = output_tokens / seconds
                print("\n=== OLLAMA SPEED STATS ===")
                print(f"Output Tokens: {output_tokens}")
                print(f"Generation Time: {seconds:.2f} seconds")
                print(f"Token Speed: {tps:.2f} tokens/second")
                print("==========================\n")
        except Exception:
            # Fail silently if metadata structural format differs
            pass

# 2. Configure the LLM and attach our tracker
tracker = TokenSpeedTracker()

llm = ChatOllama(
    model="qwen2.5:7b",
    # model="qwen2.5:3b",
    # model="ministral-3:3b",
    # model="llama3.2",
    temperature=0,
    base_url="http://localhost:11434",
    callbacks=[tracker]  # <-- This tells the LLM to run our speed math!
)

# 3. Create the ROSA agent
agent = ROSA(ros_version=2, llm=llm, verbose=True, max_iterations=5)

# 4. Send your command
agent_response = agent.invoke("List all active ROS 2 topics.")
print("\n--- Agent Response ---")
print(agent_response)

# We build a prompt using the specific topics found by the agent
large_paragraph_prompt = f"""
Using the context that the agent just discovered these ROS 2 topics ({agent_response}), 
write a few detailed paragraphs explaining how ROSA utilizes LangGraph to orchestrate 
these autonomous robotics tasks. 

Elongate this explanation by describing how LangGraph models the agent's decision-making 
process as a cyclical state graph using explicit nodes for LLM reasoning and tool execution, 
connected by conditional edges that evaluate the system's status. Explain how this architecture 
allows ROSA to maintain a persistent state memory, dynamically route intents when interacting 
with the ROS 2 environment, execute CLI tools, inspect active topics, and loop back to the LLM 
for evaluation until a complete final response can be synthesized for the user.
"""

# Call the LLM directly to generate the long text and observe the token speed
large_text_response = llm.invoke([HumanMessage(content=large_paragraph_prompt)])

print("--- Final Large Technical Paragraph ---")
print(large_text_response.content)