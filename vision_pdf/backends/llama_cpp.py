"""
llama.cpp backend implementation for VisionPDF.

This module provides integration with llama.cpp for vision language models
through Python bindings or API server.
"""

import asyncio
import aiohttp
import base64
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from .base import VLMBackend, BackendType, ModelInfo, ModelStatus, ProcessingRequest, ProcessingResponse, ProcessingMode
from ..utils.logging_config import get_logger
from ..utils.exceptions import BackendConnectionError, ModelNotFoundError, InferenceError

logger = get_logger(__name__)


class LlamaCppBackend(VLMBackend):
    """
    llama.cpp backend for vision language models.

    This backend integrates with llama.cpp either through Python bindings
    or by communicating with a llama.cpp server instance.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize llama.cpp backend.

        Args:
            config: Backend configuration dictionary
        """
        super().__init__(config)

        # Configuration options
        self.server_url = config.get('server_url', 'http://localhost:8080')
        self.model_path = config.get('model_path')
        self.use_python_bindings = config.get('use_python_bindings', False)
        self.timeout = config.get('timeout', 120)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 2.0)
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 4096)
        self.context_size = config.get('context_size', 2048)
        self.gpu_layers = config.get('gpu_layers', 0)
        self.n_batch = config.get('n_batch', 512)

        # Validate configuration
        if self.use_python_bindings and not self.model_path:
            raise BackendConnectionError("model_path is required when using Python bindings", "llama_cpp")

        if not self.use_python_bindings and not self.server_url:
            raise BackendConnectionError("server_url is required when not using Python bindings", "llama_cpp")

        # Clean up server URL
        if self.server_url.endswith('/'):
            self.server_url = self.server_url[:-1]

        self._session: Optional[aiohttp.ClientSession] = None
        self._model: Optional[Any] = None
        self._available_models: List[str] = []

    def _get_required_config_keys(self) -> List[str]:
        """Get required configuration keys."""
        if self.use_python_bindings:
            return ['model_path']
        else:
            return ['server_url']

    async def initialize(self) -> None:
        """Initialize the llama.cpp backend."""
        try:
            if self.use_python_bindings:
                await self._init_python_bindings()
            else:
                await self._init_server_client()

            logger.info(f"llama.cpp backend initialized (bindings: {self.use_python_bindings})")

        except Exception as e:
            if self._session:
                await self._session.close()
                self._session = None
            raise BackendConnectionError(f"Failed to initialize llama.cpp backend: {e}", "llama_cpp")

    async def _init_python_bindings(self) -> None:
        """Initialize llama.cpp Python bindings."""
        try:
            import llama_cpp

            # Check if model file exists
            model_path = Path(self.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            # Initialize model
            self._model = llama_cpp.Llama(
                model_path=str(model_path),
                n_ctx=self.context_size,
                n_gpu_layers=self.gpu_layers,
                n_batch=self.n_batch,
                verbose=False
            )

            logger.info(f"Loaded llama.cpp model: {self.model_path}")

        except ImportError:
            raise BackendConnectionError(
                "llama-cpp-python package not found. Install with: pip install llama-cpp-python",
                "llama_cpp"
            )

    async def _init_server_client(self) -> None:
        """Initialize llama.cpp server client."""
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self._session = aiohttp.ClientSession(timeout=timeout)

        # Test connection
        if not await self.test_connection():
            raise BackendConnectionError(
                f"Cannot connect to llama.cpp server at {self.server_url}",
                "llama_cpp"
            )

        # Get server info
        await self._refresh_server_info()

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._session:
            await self._session.close()
            self._session = None

        if self._model:
            # Python bindings model cleanup
            self._model = None

        logger.info("llama.cpp backend cleaned up")

    async def test_connection(self) -> bool:
        """Test connection to llama.cpp."""
        if self.use_python_bindings:
            return self._model is not None
        else:
            try:
                if not self._session:
                    timeout = aiohttp.ClientTimeout(total=5)
                    self._session = aiohttp.ClientSession(timeout=timeout)

                # Check server health endpoint
                async with self._session.get(f"{self.server_url}/health") as response:
                    return response.status == 200

            except Exception as e:
                logger.debug(f"llama.cpp connection test failed: {e}")
                return False

    async def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models."""
        if self.use_python_bindings:
            return await self._get_local_models()
        else:
            return await self._get_server_models()

    async def _get_local_models(self) -> List[ModelInfo]:
        """Get available local models."""
        models = []

        if self.model_path:
            model_name = Path(self.model_path).name
            model_info = ModelInfo(
                name=model_name,
                backend_type=BackendType.LLAMA_CPP,
                status=ModelStatus.AVAILABLE,
                description=f"Local llama.cpp model: {model_name}",
                supported_modes=[ProcessingMode.VISION_ONLY, ProcessingMode.HYBRID],
                requires_api_key=False
            )
            models.append(model_info)

        return models

    async def _get_server_models(self) -> List[ModelInfo]:
        """Get available server models."""
        try:
            if not self._session:
                await self.initialize()

            # Get model info from server
            async with self._session.get(f"{self.server_url}/props") as response:
                if response.status != 200:
                    raise BackendConnectionError(f"Failed to get model info: {response.status}", "llama_cpp")

                props = await response.json()
                model_name = props.get('model_name', 'unknown')

                # Check if it's a vision model (llama.cpp with vision support)
                supports_vision = props.get('vision', False) or 'vision' in model_name.lower()

                if supports_vision:
                    model_info = ModelInfo(
                        name=model_name,
                        backend_type=BackendType.LLAMA_CPP,
                        status=ModelStatus.AVAILABLE,
                        description=f"llama.cpp server model: {model_name}",
                        supported_modes=[ProcessingMode.VISION_ONLY, ProcessingMode.HYBRID],
                        requires_api_key=False
                    )
                    return [model_info]
                else:
                    logger.warning(f"Model {model_name} does not support vision")
                    return []

        except Exception as e:
            logger.error(f"Failed to get llama.cpp server models: {e}")
            return []

    async def process_page(
        self,
        request: ProcessingRequest,
        progress_callback: Optional[callable] = None
    ) -> ProcessingResponse:
        """
        Process a page using llama.cpp.

        Args:
            request: ProcessingRequest with page data
            progress_callback: Optional progress callback

        Returns:
            ProcessingResponse with markdown content
        """
        try:
            if self.use_python_bindings:
                return await self._process_with_bindings(request)
            else:
                return await self._process_with_server(request)

        except Exception as e:
            logger.error(f"llama.cpp processing failed: {e}")
            return ProcessingResponse(
                markdown="",
                confidence=0.0,
                processing_time=0.0,
                model_used="llama_cpp",
                error_message=str(e),
                elements_detected=[]
            )

    async def _process_with_bindings(self, request: ProcessingRequest) -> ProcessingResponse:
        """Process page using Python bindings."""
        import time
        start_time = time.time()

        try:
            # Prepare prompt
            prompt = self._prepare_prompt(request)

            # Note: This is a simplified implementation
            # In practice, you'd need to handle vision input properly
            # llama.cpp vision support is still evolving

            # Generate text
            output = self._model(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["\n\n\n", "###", "##"],
                echo=False
            )

            generated_text = output['choices'][0]['text'].strip()
            processing_time = time.time() - start_time

            # Estimate confidence and detect elements
            confidence = self._estimate_confidence(generated_text)
            elements_detected = self._detect_elements(generated_text)

            return ProcessingResponse(
                markdown=generated_text,
                confidence=confidence,
                processing_time=processing_time,
                model_used="llama_cpp_bindings",
                elements_detected=elements_detected,
                metadata={
                    'input_tokens': output.get('usage', {}).get('prompt_tokens', 0),
                    'output_tokens': output.get('usage', {}).get('completion_tokens', 0)
                }
            )

        except Exception as e:
            raise InferenceError(f"llama.cpp inference failed: {e}", model_name="llama_cpp")

    async def _process_with_server(self, request: ProcessingRequest) -> ProcessingResponse:
        """Process page using server API."""
        import time
        start_time = time.time()

        try:
            # Prepare request
            server_request = await self._prepare_server_request(request)

            # Call server
            async with self._session.post(
                f"{self.server_url}/completion",
                json=server_request
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise InferenceError(
                        f"llama.cpp server error: {response.status} - {error_text}",
                        model_name="llama_cpp_server"
                    )

                result = await response.json()

            processing_time = time.time() - start_time
            generated_text = result.get('content', '').strip()

            # Estimate confidence and detect elements
            confidence = self._estimate_confidence(generated_text)
            elements_detected = self._detect_elements(generated_text)

            return ProcessingResponse(
                markdown=generated_text,
                confidence=confidence,
                processing_time=processing_time,
                model_used="llama_cpp_server",
                elements_detected=elements_detected,
                metadata={
                    'input_tokens': result.get('prompt_eval_count', 0),
                    'output_tokens': result.get('eval_count', 0),
                    'model': result.get('model', 'unknown')
                }
            )

        except Exception as e:
            raise InferenceError(f"llama.cpp server inference failed: {e}", model_name="llama_cpp_server")

    async def _prepare_server_request(self, request: ProcessingRequest) -> Dict[str, Any]:
        """Prepare request for llama.cpp server."""
        # Handle image if present
        image_data = None
        if request.image_path and request.image_path.exists():
            # For now, we'll just read the image as base64
            # In practice, you'd need to implement proper image preprocessing
            with open(request.image_path, 'rb') as f:
                image_b64 = base64.b64encode(f.read()).decode('utf-8')

        prompt = self._prepare_prompt(request)

        server_request = {
            "prompt": prompt,
            "temperature": self.temperature,
            "n_predict": self.max_tokens,
            "stop": ["\n\n\n", "###", "##"],
            "stream": False
        }

        # Add image if vision is supported
        if image_data:
            # This depends on the specific llama.cpp server implementation
            server_request["image_data"] = image_data

        return server_request

    def _prepare_prompt(self, request: ProcessingRequest) -> str:
        """Prepare prompt for processing."""
        prompt = request.prompt or "Convert this PDF page to well-formatted markdown."

        if request.text_content and request.processing_mode.value == 'hybrid':
            prompt = f"Extracted text: {request.text_content}\n\n{prompt}"

        # Add formatting instructions
        prompt += "\n\nInstructions:"
        prompt += "- Use proper markdown formatting"
        prompt += "- Preserve table structures"
        prompt += "- Convert math to LaTeX format ($...$)"
        prompt += "- Format code blocks with ```language```"

        return prompt

    def _estimate_confidence(self, response_text: str) -> float:
        """Estimate confidence in response quality."""
        if not response_text or not response_text.strip():
            return 0.0

        confidence = 0.5

        # Check for markdown elements
        if '##' in response_text:
            confidence += 0.1
        if '|' in response_text and '\n|' in response_text:
            confidence += 0.15
        if '```' in response_text:
            confidence += 0.1
        if '$' in response_text and response_text.count('$') >= 2:
            confidence += 0.1

        # Length considerations
        if len(response_text) > 100:
            confidence += 0.1
        elif len(response_text) < 20:
            confidence -= 0.2

        return min(1.0, max(0.0, confidence))

    def _detect_elements(self, text: str) -> List[str]:
        """Detect different elements in response."""
        elements = []

        if '##' in text or '###' in text:
            elements.append('headings')
        if '|' in text and '\n|' in text:
            elements.append('tables')
        if '```' in text:
            elements.append('code')
        if '$' in text and text.count('$') >= 2:
            elements.append('math')
        if '*' in text or '**' in text:
            elements.append('formatting')

        return elements

    async def _refresh_server_info(self) -> None:
        """Refresh server information."""
        try:
            if not self._session:
                return

            async with self._session.get(f"{self.server_url}/props") as response:
                if response.status == 200:
                    props = await response.json()
                    model_name = props.get('model_name', 'unknown')
                    self._available_models = [model_name]
                    logger.info(f"llama.cpp server model: {model_name}")
                else:
                    logger.warning(f"Failed to get server info: {response.status}")

        except Exception as e:
            logger.warning(f"Failed to refresh server info: {e}")
            self._available_models = []

    def __repr__(self) -> str:
        """String representation of the backend."""
        if self.use_python_bindings:
            return f"LlamaCppBackend(model_path='{self.model_path}')"
        else:
            return f"LlamaCppBackend(server_url='{self.server_url}')"