from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from rosa import ROSA

# 1. Configure the LLM — reaches Ollama on the host via Docker networking
llm = ChatOllama(
    model="llama3.2",       # must match a model you've pulled in the Ollama guide
    temperature=0,            # deterministic output for reproducibility
    base_url="http://localhost:11434",
    # reasoning=True,
)

# llm = ChatOpenAI(
#     base_url="http://host.docker.internal:1234/v1",
#     api_key="lm-studio",  # LM Studio does not require a real key, but a placeholder prevents authentication errors
#     model="qwen2.5-vl-7b-instruct",  # Match the exact name of the model loaded in LM Studio
#     temperature=0,
#     streaming=False,
# )

# 2. Create the ROSA agent (no custom tools yet)
agent = ROSA(ros_version=2, llm=llm, verbose=True)

# 3. Send a natural language command
response = agent.invoke("List all active ROS 2 topics.")
print(response)