"""
Ollama Client
Connection management for local LLM inference via Ollama
"""

from typing import Optional, List, Dict, Any
import os
import json
import asyncio
from dataclasses import dataclass

try:
    import httpx
except ImportError:
    httpx = None


@dataclass
class OllamaResponse:
    """Response from Ollama API"""
    text: str
    model: str
    done: bool
    total_duration: Optional[int] = None
    eval_count: Optional[int] = None
    prompt_eval_count: Optional[int] = None


class OllamaClient:
    """
    Client for interacting with Ollama API.
    Supports text generation and embeddings.

    Ollama handles concurrency natively via environment variables:
    - OLLAMA_NUM_PARALLEL: Max parallel requests per model (default: auto 1-4)
    - OLLAMA_MAX_LOADED_MODELS: Max concurrent models (default: 3)
    - OLLAMA_MAX_QUEUE: Max queued requests before 503 (default: 512)

    For RTX 3060 Ti (12GB VRAM), recommended settings:
        export OLLAMA_NUM_PARALLEL=2
        export OLLAMA_MAX_QUEUE=100
    """

    # Default model: Qwen2.5-7B for reliable generation without "thinking" mode
    # Qwen3 uses thinking mode which complicates output parsing
    # Qwen2.5 provides direct responses and good tool calling support
    # Fits comfortably on 12GB VRAM (~5GB)
    DEFAULT_MODEL = "qwen2.5:7b"
    DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

    # Standard token count for generation
    DEFAULT_MAX_TOKENS = 500

    def __init__(
        self,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        timeout: float = 120.0,
    ):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama API URL (default: localhost:11434)
            default_model: Default model for generation (default: qwen3:8b)
            embedding_model: Model for embeddings (default: nomic-embed-text)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.default_model = default_model or os.getenv("OLLAMA_MODEL", self.DEFAULT_MODEL)
        self.embedding_model = embedding_model or os.getenv("OLLAMA_EMBED_MODEL", self.DEFAULT_EMBEDDING_MODEL)
        self.timeout = timeout

        if httpx is None:
            raise ImportError(
                "httpx package not installed. "
                "Install with: pip install httpx"
            )

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> OllamaResponse:
        """
        Generate text from prompt.

        Args:
            prompt: User prompt
            model: Model to use (defaults to default_model)
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (higher for thinking models)
            stop: Stop sequences

        Returns:
            OllamaResponse with generated text
        """
        model = model or self.default_model
        # Use higher default for Qwen3 thinking mode
        max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        if system:
            payload["system"] = system

        if stop:
            payload["options"]["stop"] = stop

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            data = response.json()

        # Qwen3 uses a "thinking" mode - extract response or fallback to thinking
        response_text = data.get("response", "")
        thinking_text = data.get("thinking", "")

        if not response_text.strip() and thinking_text:
            # Extract useful content from thinking if response is empty
            response_text = self._extract_from_thinking(thinking_text)

        return OllamaResponse(
            text=response_text,
            model=model,
            done=data.get("done", True),
            total_duration=data.get("total_duration"),
            eval_count=data.get("eval_count"),
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> OllamaResponse:
        """
        Chat-style generation with message history.

        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            OllamaResponse with generated text
        """
        model = model or self.default_model
        max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            data = response.json()

        message = data.get("message", {})
        response_text = message.get("content", "")
        thinking_text = message.get("thinking", "")

        # Handle Qwen3 thinking mode
        if not response_text.strip() and thinking_text:
            response_text = self._extract_from_thinking(thinking_text)

        return OllamaResponse(
            text=response_text,
            model=model,
            done=data.get("done", True),
            total_duration=data.get("total_duration"),
            eval_count=data.get("eval_count"),
        )

    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> List[float]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed
            model: Embedding model to use

        Returns:
            Embedding vector
        """
        model = model or self.embedding_model

        payload = {
            "model": model,
            "prompt": text,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json=payload
            )
            response.raise_for_status()
            data = response.json()

        return data.get("embedding", [])

    async def embed_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            model: Embedding model

        Returns:
            List of embedding vectors
        """
        # Process in parallel with concurrency limit
        semaphore = asyncio.Semaphore(5)

        async def embed_one(text: str) -> List[float]:
            async with semaphore:
                return await self.embed(text, model)

        tasks = [embed_one(text) for text in texts]
        return await asyncio.gather(*tasks)

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()

        return data.get("models", [])

    async def is_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> Dict[str, Any]:
        """
        Generate with tool/function calling support.

        Qwen3 has excellent tool calling (0.933 F1 on BFCL benchmark).

        Args:
            prompt: User prompt
            tools: List of tool definitions in OpenAI format
            model: Model to use
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Dict with 'text' and optional 'tool_calls'
        """
        model = model or self.default_model

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            data = response.json()

        message = data.get("message", {})
        return {
            "text": message.get("content", ""),
            "tool_calls": message.get("tool_calls", []),
            "model": model,
            "done": data.get("done", True),
        }

    async def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> Dict[str, Any]:
        """
        Multi-turn chat with tool calling support.

        Args:
            messages: Conversation history
            tools: Available tools
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Dict with response and tool calls
        """
        model = model or self.default_model

        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            data = response.json()

        message = data.get("message", {})
        return {
            "text": message.get("content", ""),
            "tool_calls": message.get("tool_calls", []),
            "model": model,
            "done": data.get("done", True),
        }

    def generate_sync(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> OllamaResponse:
        """
        Synchronous version of generate.
        """
        return asyncio.get_event_loop().run_until_complete(
            self.generate(prompt, model, system, temperature, max_tokens)
        )

    def embed_sync(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Synchronous version of embed.
        """
        return asyncio.get_event_loop().run_until_complete(
            self.embed(text, model)
        )

    def _extract_from_thinking(self, thinking: str) -> str:
        """
        Extract useful content from Qwen3's thinking output.

        Qwen3 outputs its reasoning to a 'thinking' field. This method
        attempts to extract actionable content when the main response is empty.

        For planning/action tasks, we try to find:
        1. Explicit conclusions or answers
        2. Action items or plans
        3. The most concrete/actionable statement
        """
        if not thinking:
            return ""

        import re

        lines = thinking.strip().split('\n')

        # Meta-commentary patterns to filter out
        meta_patterns = [
            r'^okay[,\.\s]', r'^alright[,\.\s]', r'^let me',
            r'^i need to', r'^the user', r'^they want',
            r'^thinking about', r'^considering', r'^first[,\s]i',
            r'^let\'s see', r'^so[,\s]the', r'^hmm',
            r'^wait[,\.\s]', r'^actually[,\.\s]',
        ]

        def is_meta(line: str) -> bool:
            """Check if line is meta-commentary"""
            line_lower = line.lower().strip()
            return any(re.match(p, line_lower) for p in meta_patterns)

        # 1. Look for explicit answer/action patterns
        answer_patterns = [
            r'(?:the )?answer (?:is|would be)[:\s]+["\']?(.+?)["\']?$',
            r'(?:the )?result (?:is|would be)[:\s]+["\']?(.+?)["\']?$',
            r'(?:elena |she )(?:would|should|will)[:\s]+(.+)',
            r'current action[:\s]+["\']?(.+?)["\']?$',
            r'response[:\s]+["\']?(.+?)["\']?$',
        ]

        for line in lines:
            if is_meta(line):
                continue
            line_lower = line.lower().strip()
            for pattern in answer_patterns:
                match = re.search(pattern, line_lower)
                if match:
                    extracted = match.group(1).strip()
                    if len(extracted) > 5 and not is_meta(extracted):
                        return extracted[0].upper() + extracted[1:]

        # 2. Look for quoted content (often the actual output)
        for line in lines:
            # Match content in quotes
            match = re.search(r'["\']([^"\']{10,})["\']', line)
            if match:
                quoted = match.group(1).strip()
                if not is_meta(quoted):
                    return quoted

        # 3. Look for action-oriented lines (for planning tasks)
        action_patterns = [
            r'^(?:elena |she )(?:is |would |will |should )(.+)',
            r'^(?:wake|eat|work|walk|go|talk|make|forge|hammer|light)(.+)',
        ]
        for line in reversed(lines):
            line = line.strip()
            if is_meta(line):
                continue
            line_lower = line.lower()
            for pattern in action_patterns:
                match = re.match(pattern, line_lower)
                if match:
                    return line[0].upper() + line[1:]

        # 4. Look for numbered list items (often contain the actual plan)
        for line in lines:
            line = line.strip()
            if re.match(r'^[\d]+[\.\)]\s*.+', line):
                cleaned = re.sub(r'^[\d]+[\.\)]\s*', '', line)
                if len(cleaned) > 10 and not is_meta(cleaned):
                    return cleaned

        # 5. Find the most concrete statement (avoiding meta-commentary)
        for line in reversed(lines):
            line = line.strip()
            if line and not is_meta(line) and len(line) > 20:
                # Also skip lines that talk about "the user" or are clearly reasoning
                if not any(kw in line.lower() for kw in ['the user', 'they want', 'i should', 'i need']):
                    return line

        # 6. Last resort: return a default action
        return "continuing with current activities"


# Global client instance
_ollama_client: Optional[OllamaClient] = None


def get_ollama_client(
    base_url: Optional[str] = None,
    force_new: bool = False
) -> OllamaClient:
    """
    Get or create Ollama client singleton.

    Args:
        base_url: Ollama URL (uses env if not provided)
        force_new: Force create new client

    Returns:
        OllamaClient instance
    """
    global _ollama_client

    if _ollama_client is None or force_new:
        _ollama_client = OllamaClient(base_url)

    return _ollama_client
