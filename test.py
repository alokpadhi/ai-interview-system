from src.utils.llm_factory import get_llm, get_secondary_llm
from src.utils.config import get_settings

settings = get_settings()

# Test primary LLM
primary = get_llm()
response = primary.invoke("Create a json object with two variables having age and name")
print(f"Primary ({settings.ollama_model}): {response.content}")

# Test secondary LLM
secondary = get_secondary_llm()
response = secondary.invoke("Route to: evaluator or feedback? Pick only one")
print(f"Secondary ({settings.ollama_model_secondary}): {response.content}")