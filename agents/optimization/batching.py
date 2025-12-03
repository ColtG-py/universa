"""
Batching Systems
Batch LLM calls and embedding requests for efficiency.
"""

from typing import Optional, List, Dict, Any, Callable, Awaitable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4
import asyncio


@dataclass
class BatchRequest:
    """A single request in a batch"""
    request_id: UUID = field(default_factory=uuid4)
    data: Any = None
    future: asyncio.Future = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class LLMBatcher:
    """
    Batches LLM requests for more efficient processing.

    Collects requests and processes them together when:
    - Batch size reaches threshold, OR
    - Timeout is reached

    This reduces overhead and can improve throughput with
    Ollama's parallel processing.
    """

    def __init__(
        self,
        process_fn: Callable[[List[str]], Awaitable[List[str]]],
        batch_size: int = 5,
        timeout_ms: float = 100.0,
    ):
        """
        Initialize batcher.

        Args:
            process_fn: Function to process batch of prompts
            batch_size: Maximum batch size
            timeout_ms: Maximum wait time before processing
        """
        self.process_fn = process_fn
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms

        self._queue: List[BatchRequest] = []
        self._lock = asyncio.Lock()
        self._processing = False

        # Stats
        self._stats = {
            "total_requests": 0,
            "batches_processed": 0,
            "avg_batch_size": 0.0,
        }

    async def submit(self, prompt: str) -> str:
        """
        Submit a prompt for processing.

        Args:
            prompt: The prompt to process

        Returns:
            Generated response
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        request = BatchRequest(
            data=prompt,
            future=future,
        )

        async with self._lock:
            self._queue.append(request)
            self._stats["total_requests"] += 1

            # Check if we should process
            if len(self._queue) >= self.batch_size:
                asyncio.create_task(self._process_batch())
            elif len(self._queue) == 1:
                # Start timeout for first request
                asyncio.create_task(self._timeout_trigger())

        return await future

    async def _timeout_trigger(self) -> None:
        """Trigger batch processing after timeout"""
        await asyncio.sleep(self.timeout_ms / 1000.0)

        async with self._lock:
            if self._queue and not self._processing:
                asyncio.create_task(self._process_batch())

    async def _process_batch(self) -> None:
        """Process the current batch"""
        async with self._lock:
            if not self._queue or self._processing:
                return

            self._processing = True
            batch = self._queue[:self.batch_size]
            self._queue = self._queue[self.batch_size:]

        try:
            # Process all prompts
            prompts = [r.data for r in batch]
            results = await self.process_fn(prompts)

            # Distribute results
            for request, result in zip(batch, results):
                if not request.future.done():
                    request.future.set_result(result)

            # Update stats
            self._stats["batches_processed"] += 1
            total_batches = self._stats["batches_processed"]
            self._stats["avg_batch_size"] = (
                (self._stats["avg_batch_size"] * (total_batches - 1) + len(batch))
                / total_batches
            )

        except Exception as e:
            # Set exception on all futures
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)

        finally:
            async with self._lock:
                self._processing = False

                # Check if more requests waiting
                if self._queue:
                    asyncio.create_task(self._process_batch())

    def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics"""
        return {
            **self._stats,
            "queue_size": len(self._queue),
            "is_processing": self._processing,
        }


class EmbeddingBatcher:
    """
    Batches embedding requests for efficient processing.

    Embeddings are particularly well-suited for batching since
    they're independent and deterministic.
    """

    def __init__(
        self,
        embed_fn: Callable[[List[str]], Awaitable[List[List[float]]]],
        batch_size: int = 10,
        timeout_ms: float = 50.0,
    ):
        """
        Initialize embedding batcher.

        Args:
            embed_fn: Function to embed batch of texts
            batch_size: Maximum batch size
            timeout_ms: Maximum wait time
        """
        self.embed_fn = embed_fn
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms

        self._queue: List[BatchRequest] = []
        self._lock = asyncio.Lock()
        self._processing = False

        self._stats = {
            "total_requests": 0,
            "batches_processed": 0,
            "total_texts_embedded": 0,
        }

    async def embed(self, text: str) -> List[float]:
        """
        Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        request = BatchRequest(
            data=text,
            future=future,
        )

        async with self._lock:
            self._queue.append(request)
            self._stats["total_requests"] += 1

            if len(self._queue) >= self.batch_size:
                asyncio.create_task(self._process_batch())
            elif len(self._queue) == 1:
                asyncio.create_task(self._timeout_trigger())

        return await future

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts.

        Args:
            texts: Texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Submit all and gather results
        tasks = [self.embed(text) for text in texts]
        return await asyncio.gather(*tasks)

    async def _timeout_trigger(self) -> None:
        """Trigger batch processing after timeout"""
        await asyncio.sleep(self.timeout_ms / 1000.0)

        async with self._lock:
            if self._queue and not self._processing:
                asyncio.create_task(self._process_batch())

    async def _process_batch(self) -> None:
        """Process the current batch"""
        async with self._lock:
            if not self._queue or self._processing:
                return

            self._processing = True
            batch = self._queue[:self.batch_size]
            self._queue = self._queue[self.batch_size:]

        try:
            texts = [r.data for r in batch]
            embeddings = await self.embed_fn(texts)

            for request, embedding in zip(batch, embeddings):
                if not request.future.done():
                    request.future.set_result(embedding)

            self._stats["batches_processed"] += 1
            self._stats["total_texts_embedded"] += len(batch)

        except Exception as e:
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)

        finally:
            async with self._lock:
                self._processing = False

                if self._queue:
                    asyncio.create_task(self._process_batch())

    def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics"""
        return {
            **self._stats,
            "queue_size": len(self._queue),
            "avg_batch_size": (
                self._stats["total_texts_embedded"] / self._stats["batches_processed"]
                if self._stats["batches_processed"] > 0 else 0
            ),
        }


class RequestCoalescer:
    """
    Coalesces identical requests to avoid duplicate processing.

    If multiple requests for the same input arrive before processing,
    they share the same result.
    """

    def __init__(
        self,
        process_fn: Callable[[str], Awaitable[Any]],
    ):
        """
        Initialize coalescer.

        Args:
            process_fn: Function to process single request
        """
        self.process_fn = process_fn
        self._pending: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

        self._stats = {
            "total_requests": 0,
            "coalesced_requests": 0,
            "unique_processed": 0,
        }

    async def request(self, key: str, data: str) -> Any:
        """
        Make a request, coalescing with any pending identical request.

        Args:
            key: Unique key for this request type
            data: Request data

        Returns:
            Processing result
        """
        async with self._lock:
            self._stats["total_requests"] += 1

            # Check if identical request is pending
            if key in self._pending:
                self._stats["coalesced_requests"] += 1
                return await self._pending[key]

            # Create new future
            loop = asyncio.get_event_loop()
            future = loop.create_future()
            self._pending[key] = future

        try:
            # Process the request
            result = await self.process_fn(data)
            future.set_result(result)
            self._stats["unique_processed"] += 1
            return result

        except Exception as e:
            future.set_exception(e)
            raise

        finally:
            async with self._lock:
                self._pending.pop(key, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get coalescer statistics"""
        total = self._stats["total_requests"]
        coalesced = self._stats["coalesced_requests"]

        return {
            **self._stats,
            "pending_count": len(self._pending),
            "coalesce_rate": coalesced / total if total > 0 else 0.0,
        }
