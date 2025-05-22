from dotenv import load_dotenv
import httpx
import json
import os
from typing import AsyncGenerator, Optional, List


# Load environment variables from .env file
load_dotenv()

# Access the environment variables
ollama_url = os.getenv("OLLAMA_URL")
vllm_url = os.getenv("VLLM_URL")


class CancellationToken:
    def __init__(self):
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True


class EnvConfig:
    def __init__(self):
        # VLLM Configuration
        self.vllm_url = os.getenv("VLLM_URL")
        self.vllm_model = os.getenv("VLLM_MODEL")
        
        # Ollama Configuration
        self.ollama_url = os.getenv("OLLAMA_URL")
        self.ollama_model = os.getenv("OLLAMA_MODEL")
    
    def is_vllm_available(self) -> bool:
        """Check if VLLM is configured and available for specific model type"""
        return bool(self.vllm_url and self.vllm_model)
    
    def is_ollama_available(self) -> bool:
        """Check if Ollama is configured and available for specific model type"""
        return bool(self.ollama_url and self.ollama_model)
    
    def get_model_and_url(self) -> tuple[Optional[str], Optional[str]]:
        """Get the appropriate model and URL based on availability"""
        # First try VLLM
        if self.is_vllm_available():
            return self.vllm_model, self.vllm_url
        # Then try Ollama
        elif self.is_ollama_available():
            return self.ollama_model, self.ollama_url
        return None, None


async def invoke_llm(
    prompt: Optional[str] = None,
    messages: Optional[List] = None,
    config: Optional[EnvConfig] = None,
    stream: bool = False,
) -> dict | AsyncGenerator[str, None]:
    """
    Unified interface for invoking LLM models. Automatically chooses between VLLM and Ollama
    based on availability, with priority given to VLLM.
    """
    if stream:
        return stream_llm(prompt, messages, config)
    if config is None:
        config = EnvConfig()
    
    model, url = config.get_model_and_url()
    
    if not model or not url:
        return {"error": f"No LLM service available for model"}
    
    if config.is_vllm_available():
        return await invoke_llm_vllm(model, prompt=prompt, messages=messages)
    else:
        return await invoke_llm_ollama(model, prompt=prompt, messages=messages)
    
    
async def stream_llm(
    prompt: Optional[str] = None,
    messages: Optional[List] = None,
    config: Optional[EnvConfig] = None,
) -> AsyncGenerator[str, None]:
    """Unified streaming interface with cancellation support"""
    if config is None:
        config = EnvConfig()
    
    model, url = config.get_model_and_url()
    
    if not model or not url:
        yield f"Error: No LLM service available for model"
        return
    
    if config.is_vllm_available():
        async for chunk in stream_llm_vllm(model, prompt=prompt, messages=messages):
            yield chunk
    else:
        async for chunk in stream_llm_ollama(model, prompt=prompt, messages=messages):
            yield chunk

##############################################################################################################################
##############################################################################################################################
###############################################VLLM GENERATION FUNCTIONS START################################################
##############################################################################################################################


async def invoke_llm_vllm(
    model: str,
    prompt: Optional[str] = None,
    messages: Optional[List] = None,
    temperature: float = 0.0, 
    top_p: float = 0.1,
    top_k: int = 1,   
    seed: int = 42
) -> dict:
    """Invoke the LLM with specified sampling parameters and return the final non-streaming response."""
    
    # Define the payload for the request with sampling parameters
    payload = {
        "model": model,
        "messages": messages if messages else [
            {"role": "system", "content": prompt[0]},
            {"role": "user", "content": prompt[1]}
        ],
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "stream": False  # Set stream to False for non-streaming
    }
    
    try:
        async with httpx.AsyncClient() as client:
            # Use the provided VLLM URL
            response = await client.post(vllm_url, json=payload, timeout=None)
            
            if response.status_code == 200:
                response_data = json.loads(response.content)
                ai_msg = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
                return {"answer": ai_msg}
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return {"error": response.text}
    
    except httpx.TimeoutException:
        print("Request timed out.")
        return {"error": "Request timed out"}
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return {"error": str(e)}
    

async def stream_llm_vllm(ollama_model, prompt: Optional[str] = None, messages: Optional[List] = None,
    temperature: float = 0.0,
    top_p: float = 0.1,
    top_k: int = 1,   
    seed: int = 42
) -> AsyncGenerator[str, None]:
    """Stream responses from the LLM with cancellation support"""
    
    payload = {
        "model": ollama_model,
        "messages": messages if messages else [
            {"role": "system", "content": prompt[0]},
            {"role": "user", "content": prompt[1]}
        ],
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "stream": True
    }

    async with httpx.AsyncClient() as client:
        try:
            async with client.stream('POST', vllm_url, json=payload, timeout=None) as response:
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        # if cancellation_token.is_cancelled:
                        #     # Close the connection explicitly
                        #     await response.aclose()
                        #     break
                            
                        if line:
                            raw_line = line.lstrip("data: ").strip()
                            
                            if raw_line == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(raw_line)
                                content = data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue
                else:
                    print(f"Request failed with status code {response.status_code}")
        except Exception as e:
            print(f"Error during streaming: {str(e)}")
            raise

##############################################################################################################################
##############################################################################################################################
################################################VLLM GENERATION FUNCTIONS STOP################################################
##############################################################################################################################



##############################################################################################################################
##############################################################################################################################
################################################OLLAMA GENERATION FUNCTIONS START#############################################
##############################################################################################################################

# Use the global ollama_url directly inside the function
async def invoke_llm_ollama(ollama_model, prompt: Optional[str] = None, messages: Optional[List] = None):
    model = os.getenv("OLLAMA_MODEL")
    ollama_model = model
    payload = {
        "messages": messages,
        "prompt": prompt,
        "model": ollama_model,
        "options": {
            "top_k": 1, 
            "top_p": 0, 
            "temperature": 0,
            "seed": 100,
            "num_ctx": 4096
        },
        "stream": False
    }

    try:
        async with httpx.AsyncClient() as client:
            # Use the global ollama_url here
            response = await client.post(f"{ollama_url}/api/generate", json=payload, timeout=None)
            response_data = json.loads(response.content)

        if response.status_code == 200:
            ai_msg = response_data['response']
            return {"answer": ai_msg}
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return {"error": response.text}
    except httpx.TimeoutException:
        print("Request timed out. This should not happen with unlimited timeout.")
        return {"error": "Request timed out"}
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return {"error": str(e)}


async def stream_llm_ollama(ollama_model, prompt: Optional[str] = None, messages: Optional[List] = None) -> AsyncGenerator[str, None]:
    """Stream responses from Ollama with cancellation support"""
    payload = {
        "prompt": prompt,
        "messages": messages,
        "model": ollama_model,
        "options": {
            "top_k": 1,
            "top_p": 0,
            "temperature": 0,
            "seed": 100,
            "num_ctx": 4096
        },
        "stream": True
    }
    
    async with httpx.AsyncClient() as client:
        try:
            async with client.stream('POST', f"{ollama_url}/api/generate", json=payload, timeout=None) as response:
                async for line in response.aiter_lines():
                    # if cancellation_token.is_cancelled:
                    #     # Close the connection explicitly
                    #     await response.aclose()
                    #     break
                        
                    if line:
                        try:
                            data = json.loads(line)
                            if 'response' in data:
                                yield data['response']
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Error during streaming: {str(e)}")
            raise



##############################################################################################################################
##############################################################################################################################
################################################OLLAMA GENERATION FUNCTIONS STOP##############################################
##############################################################################################################################


from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from langchain_core.runnables import RunnableConfig

from typing import Optional, List, AsyncGenerator, Any


class SpandaLLM(BaseChatModel):
    """LangChain-compatible wrapper around custom async LLM interface (supports tools)."""
    config: Optional[EnvConfig] = Field(default_factory=EnvConfig)
    streaming: bool = False

    @property
    def _llm_type(self) -> str:
        return "spanda-llm"

    async def _astream(self, messages: List[Any], stop: Optional[List[str]] = None, **kwargs) -> AsyncGenerator[AIMessage, None]:
        """Stream responses from the underlying model."""
        content = ""
        async for chunk in stream_llm(messages=messages, config=self.config):
            content += chunk
            yield AIMessage(content=chunk)
        # Final full message
        yield AIMessage(content=content)

    async def _agenerate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResult:
        """Non-streaming response for a single generation."""
        result = await invoke_llm(messages=messages, config=self.config, stream=False)

        if "answer" in result:
            msg = AIMessage(content=result["answer"])
        else:
            msg = AIMessage(content=f"Error: {result.get('error')}")

        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def ainvoke(self, input: List[Any], config: Optional[RunnableConfig] = None, **kwargs) -> AIMessage:
        """LangChain-style invoke support"""
        result = await self._agenerate(input, **kwargs)
        return result.generations[0].message

    async def astream(self, input: List[Any], config: Optional[RunnableConfig] = None, **kwargs) -> AsyncGenerator[AIMessage, None]:
        """LangChain-style astream support"""
        async for msg in self._astream(input, **kwargs):
            yield msg
    
    def _generate(self, messages: List[Any], stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
        raise NotImplementedError("Synchronous _generate is not supported. Use async methods.")
