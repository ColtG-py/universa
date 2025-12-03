"""
LLM Integration Module
Provides LLM-based scoring, embedding, and generation functions.

Default model: Qwen3-8B (excellent tool calling, 0.933 F1 on BFCL)
Fits on RTX 3060 Ti (12GB VRAM) with Q4 quantization.

Configure Ollama for concurrent agent requests:
    export OLLAMA_NUM_PARALLEL=2
    export OLLAMA_MAX_QUEUE=100
"""

from agents.llm.ollama_client import OllamaClient, OllamaResponse, get_ollama_client
from agents.llm.importance_scorer import ImportanceScorer
from agents.llm.embedding_generator import EmbeddingGenerator
from agents.llm.prompts import PromptTemplates

__all__ = [
    "OllamaClient",
    "OllamaResponse",
    "get_ollama_client",
    "ImportanceScorer",
    "EmbeddingGenerator",
    "PromptTemplates",
]
