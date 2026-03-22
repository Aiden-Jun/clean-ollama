# clean-ollama
A minimal, typed Python interface for Ollama that simplifies the existing ollama library 
## Installation
Install with pip
```bash
pip install clean-ollama
```
## Requirements
- Python 3.10+
- [Ollama](https://ollama.com) installed
## Usage
### Basic generation
```python
from clean_ollama import Client, Message, Role

client = Client("qwen3.5:4b")
messages = [
    Message(Role.USER, "What is the capital of France?")
]
thinking, response, tool_calls = client.generate(messages)
print(response)
```
### Streaming
```python
from clean_ollama import Client, Message, Role

client = Client("qwen3.5:4b")
messages = [
    Message(Role.USER, "Tell me a joke.")
]
for thinking, chunk in client.stream(messages):
    print(chunk, end="", flush=True)
```
### Tool use
```python
from clean_ollama import Client, Message, Role, Tool, Param, ParamType

get_weather = Tool(
    name="get_weather",
    description="Get the current weather for a location",
    params=[
        Param("location", "City name", ParamType.string),
        Param("units", "celsius or fahrenheit", ParamType.string, required=False),
    ]
)
messages = [
    Message(Role.USER, "What's the weather in Tokyo?")
]
thinking, response, tool_calls = client.generate(messages, tools=[get_weather])
print(tool_calls)
```
### What tool calls look like
`generate` returns a list of tool calls. Each call has a `function` with a `name` and `arguments` dict:
```python
thinking, response, tool_calls = client.generate(messages, tools=[get_weather])
for call in tool_calls:
    print(call.function.name) # "get_weather"
    print(call.function.arguments) # {"location": "Tokyo", "units": "celsius"}
```
If the model didn't call any tools, `tool_calls` is an empty list.
### Thinking (extended reasoning)
```python
from clean_ollama import Client, Message, Role

thinking, response, tool_calls = client.generate(messages, think=True)
print("Thinking:", thinking)
print("Response:", response)
```
### Loading and unloading models
```python
from clean_ollama import Client

client = Client("qwen3.5:4b")
client.load() # keeps the model in memory
client.unload() # frees the model from memory
```
## API Reference
### `Client(model: str)`
Main class for interacting with Ollama.
| Method | Description |
|---|---|
| `load()` | Loads the model into memory |
| `unload()` | Unloads the model from memory |
| `generate(messages, tools=None, think=False)` | Returns `(thinking, response, tool_calls)` |
| `stream(messages, think=False)` | Yields `(thinking, chunk)` pairs |
### `Message(role, content)`
Represents a chat message. `role` accepts a `Role` enum or a plain string.
### `Role`
Enum with values: `SYSTEM`, `USER`, `ASSISTANT`
### `Tool(name, description, params)`
Defines a callable tool the model can invoke.
### `Param(name, description, param_type, required=True)`
Defines a single tool parameter.
### `ParamType`
Enum for parameter types: `string`, `integer`, `float`, `boolean`, `json`
## License
MIT