from typing import Iterator, Tuple
from enum import Enum
import subprocess
import ollama


class ParamType(Enum):
    string = "string"
    integer = "integer"
    float = "number"
    boolean = "boolean"
    json = "object"


class Param:
    def __init__(self, name: str, description: str, param_type: ParamType, required: bool=True):
        if not isinstance(param_type, ParamType):
            raise ValueError(f"param_type must be a ParamType, got {param_type}")
        self.name = name
        self.description = description
        self.param_type = param_type
        self.required = required


class Tool:
    def __init__(self, name, description, params: list[Param]):
        self.name = name
        self.description = description
        self.params = params


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message:
    def __init__(self, role: Role, content: str):
        if isinstance(role, str):
            role = Role(role)
        self.role = role
        self.content = content


class Client:
    def __init__(self, model):
        self._model = model
        self._check_ollama_installed()

    @staticmethod
    def _check_ollama_installed():
        try:
            subprocess.run(
                ["ollama", "--version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "ollama is not installed or not found in PATH. "
                "Install it from https://www.ollama.com/"
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ollama check failed with exit code {e.returncode}")

    def load(self):
        ollama.chat(model=self._model, messages=[], keep_alive=-1)

    def unload(self):
        ollama.chat(model=self._model, messages=[], keep_alive=0)

    @staticmethod
    def tools_to_schema(tools: list) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            p.name: {
                                "type": p.param_type.value,
                                "description": p.description,
                            }
                            for p in t.params
                        },
                        "required": [p.name for p in t.params if p.required],
                    },
                },
            }
            for t in tools
        ]

    @staticmethod
    def messages_to_schema(messages: list) -> list[dict]:
        return [{"role": m.role.value, "content": m.content} for m in messages]

    def generate(self, messages: list[Message], tools: list[Tool]=None, think: bool=False) -> tuple[str, str, list]:
        if tools is None:
            tools = []
        response = ollama.chat(
            model=self._model,
            messages=self.messages_to_schema(messages),
            tools=self.tools_to_schema(tools),
            think=think,
        )

        tool_calls = response.message.tool_calls or []

        if think and response.message.thinking:
            lines = response.message.thinking.strip().splitlines()
            thinking = "\n".join(lines)
            return thinking, response.message.content.strip(), tool_calls
        return "", response.message.content or "", tool_calls

    def stream(self, messages: list[Message], tools: list[Tool]=None, think: bool=False) -> Iterator[Tuple[str, str]]:
        if tools is None:
            tools = []

        for chunk in ollama.chat(
            model=self._model,
            messages=self.messages_to_schema(messages),
            tools=self.tools_to_schema(tools),
            think=think,
            stream=True,
        ):
            msg = getattr(chunk, "message", {})
            content = getattr(msg, "content", "") or ""
            thinking = getattr(msg, "thinking", None)
            yield thinking or "", content
