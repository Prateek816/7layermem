from src.memory import AgentMemory

memory = AgentMemory.from_config(data_dir="./testing/data")

memory.remember("Alice is a developer")
memory.remember("Bob is a designer")