"""
Ollama backend implementation for VisionPDF.

This module provides integration with Ollama's vision language models
for PDF to markdown conversion.
"""

import asyncio
import aiohttp
import base64
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from .base import VLMBackend, BackendType, ModelInfo, ModelStatus, ProcessingRequest, ProcessingResponse
from ..utils.logging_config import get_logger
from ..utils.exceptions import BackendConnectionError, ModelNotFoundError, InferenceError

logger = get_logger(__name__)


class OllamaBackend(VLMBackend):
    """
    Ollama backend for vision language models.

    This backend integrates with Ollama's REST API to provide
    vision-based PDF processing capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Ollama backend.

        Args:
            config: Backend configuration dictionary
        """
        super().__init__(config)

        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.model = config.get('model', 'llava')
        self.timeout = config.get('timeout', 60)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 4096)

        # Validate configuration
        if not self.base_url:
            raise BackendConnectionError("Ollama base URL is required", "ollama")

        # Ensure URL doesn't end with /
        if self.base_url.endswith('/'):
            self.base_url = self.base_url[:-1]

        self._session: Optional[aiohttp.ClientSession] = None
        self._available_models: List[str] = []

    def _get_required_config_keys(self) -> List[str]:
        """Get required configuration keys."""
        return ['base_url']

    async def initialize(self) -> None:
        """Initialize the Ollama backend."""
        try:
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)

            # Test connection
            if not await self.test_connection():
                raise BackendConnectionError(f"Cannot connect to Ollama at {self.base_url}", "ollama")

            # Load available models
            await self._refresh_available_models()

            # Check if the requested model is available
            if self.model not in self._available_models:
                logger.warning(f"Model '{self.model}' not found in available models: {self._available_models}")

            logger.info(f"Ollama backend initialized with model: {self.model}")

        except Exception as e:
            if self._session:
                await self._session.close()
                self._session = None
            raise BackendConnectionError(f"Failed to initialize Ollama backend: {e}", "ollama")

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("Ollama backend cleaned up")

    async def test_connection(self) -> bool:
        """Test connection to Ollama."""
        try:
            if not self._session:
                timeout = aiohttp.ClientTimeout(total=5)
                self._session = aiohttp.ClientSession(timeout=timeout)

            async with self._session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200

        except Exception as e:
            logger.debug(f"Ollama connection test failed: {e}")
            return False

    async def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models from Ollama."""
        try:
            if not self._session:
                await self.initialize()

            # Get models from Ollama API
            async with self._session.get(f"{self.base_url}/api/tags") as response:
                if response.status != 200:
                    raise BackendConnectionError(f"Failed to get models: {response.status}", "ollama")

                data = await response.json()
                models = []

                for model_data in data.get('models', []):
                    model_name = model_data['name']
                    size = model_data.get('size', 0)
                    modified_at = model_data.get('modified_at', '')

                    # Check if model supports vision
                    supports_vision = self._model_supports_vision(model_name)

                    if supports_vision:
                        model_info = ModelInfo(
                            name=model_name,
                            backend_type=BackendType.OLLAMA,
                            status=ModelStatus.AVAILABLE,
                            description=f"Ollama model: {model_name} (Size: {size:,} bytes)",
                            supported_modes=[ProcessingMode.VISION_ONLY, ProcessingMode.HYBRID],
                            requires_api_key=False
                        )
                        models.append(model_info)

                return models

        except Exception as e:
            logger.error(f"Failed to get Ollama models: {e}")
            raise BackendConnectionError(f"Failed to retrieve models: {e}", "ollama")

    async def process_page(
        self,
        request: ProcessingRequest,
        progress_callback: Optional[callable] = None
    ) -> ProcessingResponse:
        """
        Process a page using Ollama.

        Args:
            request: ProcessingRequest with page data
            progress_callback: Optional progress callback

        Returns:
            ProcessingResponse with markdown content
        """
        try:
            if not self._session:
                await self.initialize()

            # Prepare the request
            ollama_request = self._prepare_ollama_request(request)

            # Call Ollama API
            response = await self._call_ollama_with_retry(ollama_request)

            # Parse response
            return self._parse_ollama_response(response, request)

        except Exception as e:
            logger.error(f"Ollama processing failed: {e}")
            return ProcessingResponse(
                markdown="",
                confidence=0.0,
                processing_time=0.0,
                model_used=self.model,
                error_message=str(e),
                elements_detected=[]
            )

    def _model_supports_vision(self, model_name: str) -> bool:
        """Check if a model supports vision capabilities."""
        # Known vision models
        vision_models = [
            'llava',
            'llava-llama3',
            'llava-phi3',
            'minicpm-v',
            'moondream',
            'cogvlm',
            'qwen2-vl',
            'qwen2.5-vl'
        ]

        model_lower = model_name.lower()
        return any(vision_model in model_lower for vision_model in vision_models)

    def _prepare_ollama_request(self, request: ProcessingRequest) -> Dict[str, Any]:
        """Prepare request for Ollama API."""
        # Read and encode image
        if not request.image_path.exists():
            raise FileNotFoundError(f"Image file not found: {request.image_path}")

        with open(request.image_path, 'rb') as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')

        # Prepare prompt
        prompt = self._enhance_prompt(request.prompt, request)

        # Build Ollama request
        ollama_request = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens or 4096
            }
        }

        return ollama_request

    def _enhance_prompt(self, base_prompt: str, request: ProcessingRequest) -> str:
        """Enhance the prompt with additional context."""
        enhanced_prompt = base_prompt

        # Add text content if available for hybrid processing
        if request.text_content and request.processing_mode.value == 'hybrid':
            enhanced_prompt = f"Extracted text: {request.text_content}\n\n{enhanced_prompt}"

        # Add formatting instructions
        enhanced_prompt += "\n\nAdditional instructions:\n"
        enhanced_prompt += "- Use proper markdown formatting (# for headings, * for emphasis)\n"
        enhanced_prompt += "- Preserve table structures using markdown table syntax\n"
        enhanced_prompt += "- Convert mathematical expressions to LaTeX format ($...$)\n"
        enhanced_prompt += "- Identify and format code blocks with ```language ...```\n"
        enhanced_prompt += "- Maintain the original document structure and hierarchy"

        return enhanced_prompt

    async def _call_ollama_with_retry(self, ollama_request: Dict[str, Any]) -> Dict[str, Any]:
        """Call Ollama API with retry logic."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                async with self._session.post(
                    f"{self.base_url}/api/generate",
                    json=ollama_request
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise InferenceError(
                            f"Ollama API error: {response.status} - {error_text}",
                            model_name=self.model
                        )

            except asyncio.TimeoutError:
                last_exception = InferenceError("Ollama request timed out", model_name=self.model)
                logger.warning(f"Ollama request timeout (attempt {attempt + 1}/{self.max_retries + 1})")

            except Exception as e:
                last_exception = e
                logger.warning(f"Ollama request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")

            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff

        raise last_exception or InferenceError("All Ollama retry attempts failed", model_name=self.model)

    def _parse_ollama_response(self, ollama_response: Dict[str, Any], request: ProcessingRequest) -> ProcessingResponse:
        """Parse Ollama response into ProcessingResponse."""
        try:
            # Extract response data
            response_text = ollama_response.get('response', '')
            total_duration = ollama_response.get('total_duration', 0) / 1e9  # Convert nanoseconds to seconds
            prompt_eval_count = ollama_response.get('prompt_eval_count', 0)
            eval_count = ollama_response.get('eval_count', 0)

            # Estimate confidence based on response quality
            confidence = self._estimate_confidence(response_text, request)

            # Detect elements in response
            elements_detected = self._detect_elements(response_text)

            # If no content was generated, try fallback
            if not response_text.strip():
                response_text = self._generate_fallback_content(request)
                confidence = 0.3

            return ProcessingResponse(
                markdown=response_text,
                confidence=confidence,
                processing_time=total_duration,
                model_used=self.model,
                elements_detected=elements_detected,
                metadata={
                    'prompt_tokens': prompt_eval_count,
                    'response_tokens': eval_count,
                    'total_tokens': prompt_eval_count + eval_count
                }
            )

        except Exception as e:
            logger.error(f"Failed to parse Ollama response: {e}")
            raise InferenceError(f"Response parsing failed: {e}", model_name=self.model)

    def _estimate_confidence(self, response_text: str, request: ProcessingRequest) -> float:
        """Estimate confidence in the response quality."""
        if not response_text or not response_text.strip():
            return 0.0

        confidence = 0.5  # Base confidence

        # Check for proper markdown formatting
        if '##' in response_text or '###' in response_text:
            confidence += 0.1
        if '|' in response_text and '\n|' in response_text:  # Likely a table
            confidence += 0.15
        if '```' in response_text:  # Code blocks
            confidence += 0.1
        if '$' in response_text and response_text.count('$') >= 2:  # Math expressions
            confidence += 0.1

        # Check length
        if len(response_text) > 100:
            confidence += 0.1
        elif len(response_text) < 20:
            confidence -= 0.2

        # Check for common indicators of good responses
        if response_text.strip().endswith(('.', '!', '?')):
            confidence += 0.05

        return min(1.0, max(0.0, confidence))

    def _detect_elements(self, text: str) -> List[str]:
        """Detect different elements in the response text."""
        elements = []

        if '##' in text or '###' in text or '####' in text:
            elements.append('headings')
        if '|' in text and '\n|' in text:
            elements.append('tables')
        if '```' in text:
            elements.append('code')
        if '$' in text and text.count('$') >= 2:
            elements.append('math')
        if '*' in text or '**' in text:
            elements.append('formatting')
        if '-' in text and '\n-' in text:
            elements.append('lists')

        return elements

    def _generate_fallback_content(self, request: ProcessingRequest) -> str:
        """Generate fallback content when VLM fails."""
        content = f"# Page {request.image_path.stem if hasattr(request.image_path, 'stem') else 'Unknown'}\n\n"

        if request.text_content:
            content += "## Extracted Text\n\n"
            content += request.text_content
        else:
            content += "*Unable to process this page. The content could not be extracted or analyzed.*"

        return content

    async def _refresh_available_models(self) -> None:
        """Refresh the list of available models."""
        try:
            models_info = await self.get_available_models()
            self._available_models = [model.name for model in models_info]
            logger.info(f"Available Ollama models: {self._available_models}")
        except Exception as e:
            logger.warning(f"Failed to refresh available models: {e}")
            self._available_models = []

    async def pull_model(self, model_name: str, progress_callback: Optional[callable] = None) -> bool:
        """
        Pull a model from Ollama registry.

        Args:
            model_name: Name of the model to pull
            progress_callback: Optional progress callback

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._session:
                await self.initialize()

            pull_request = {"name": model_name}

            async with self._session.post(
                f"{self.base_url}/api/pull",
                json=pull_request
            ) as response:
                if response.status == 200:
                    logger.info(f"Successfully pulled model: {model_name}")
                    await self._refresh_available_models()
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to pull model {model_name}: {error_text}")
                    return False

        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False

    def __repr__(self) -> str:
        """String representation of the backend."""
        return f"OllamaBackend(base_url='{self.base_url}', model='{self.model}')"