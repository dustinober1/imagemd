"""
Custom API backend implementation for VisionPDF.

This module provides integration with custom vision language model APIs,
supporting various authentication methods and API formats.
"""

import asyncio
import aiohttp
import base64
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import logging

from .base import VLMBackend, BackendType, ModelInfo, ModelStatus, ProcessingRequest, ProcessingResponse, ProcessingMode
from ..utils.logging_config import get_logger
from ..utils.exceptions import BackendConnectionError, ModelNotFoundError, InferenceError

logger = get_logger(__name__)


class CustomAPIBackend(VLMBackend):
    """
    Custom API backend for vision language models.

    This backend provides flexible integration with custom VLM APIs,
    supporting various authentication methods, request/response formats,
    and API conventions.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize custom API backend.

        Args:
            config: Backend configuration dictionary
        """
        super().__init__(config)

        # API configuration
        self.base_url = config.get('base_url', '')
        self.api_key = config.get('api_key', '')
        self.auth_type = config.get('auth_type', 'bearer')  # bearer, api_key, basic, custom
        self.headers = config.get('headers', {})
        self.timeout = config.get('timeout', 120)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 2.0)

        # Model configuration
        self.model_name = config.get('model_name', 'custom-vision-model')
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 4096)
        self.top_p = config.get('top_p', 0.9)
        self.top_k = config.get('top_k', 40)

        # Request/Response format configuration
        self.request_format = config.get('request_format', 'openai')  # openai, anthropic, custom
        self.response_format = config.get('response_format', 'auto')  # auto, openai, custom
        self.custom_request_handler = config.get('custom_request_handler')
        self.custom_response_handler = config.get('custom_response_handler')

        # Image handling
        self.image_format = config.get('image_format', 'base64')  # base64, url, multipart
        self.supported_image_formats = config.get('supported_image_formats', ['png', 'jpg', 'jpeg', 'webp'])

        # Validate configuration
        if not self.base_url:
            raise BackendConnectionError("base_url is required for custom API backend", "custom_api")

        # Clean up base URL
        if self.base_url.endswith('/'):
            self.base_url = self.base_url[:-1]

        self._session: Optional[aiohttp.ClientSession] = None
        self._model_info: Optional[ModelInfo] = None

    def _get_required_config_keys(self) -> List[str]:
        """Get required configuration keys."""
        return ['base_url']

    async def initialize(self) -> None:
        """Initialize the custom API backend."""
        try:
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)

            # Test connection
            if not await self.test_connection():
                raise BackendConnectionError(
                    f"Cannot connect to custom API at {self.base_url}",
                    "custom_api"
                )

            # Get model information
            await self._get_model_info()

            logger.info(f"Custom API backend initialized: {self.base_url}")

        except Exception as e:
            if self._session:
                await self._session.close()
                self._session = None
            raise BackendConnectionError(f"Failed to initialize custom API backend: {e}", "custom_api")

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("Custom API backend cleaned up")

    async def test_connection(self) -> bool:
        """Test connection to the custom API."""
        try:
            if not self._session:
                timeout = aiohttp.ClientTimeout(total=10)
                self._session = aiohttp.ClientSession(timeout=timeout)

            # Try to make a simple request to test connectivity
            test_url = f"{self.base_url}/health"
            headers = self._prepare_headers()

            async with self._session.get(test_url, headers=headers) as response:
                return response.status in [200, 404]  # 404 is ok, health endpoint might not exist

        except Exception as e:
            logger.debug(f"Custom API connection test failed: {e}")
            return False

    async def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models."""
        if self._model_info:
            return [self._model_info]
        else:
            # Create default model info
            model_info = ModelInfo(
                name=self.model_name,
                backend_type=BackendType.CUSTOM_API,
                status=ModelStatus.AVAILABLE,
                description=f"Custom API model at {self.base_url}",
                supported_modes=[ProcessingMode.VISION_ONLY, ProcessingMode.HYBRID],
                requires_api_key=bool(self.api_key),
                max_image_size=self.config.get('max_image_size', 1024 * 1024)  # 1MB default
            )
            return [model_info]

    async def _get_model_info(self) -> None:
        """Get detailed model information from API."""
        try:
            # Try common endpoints for model info
            info_endpoints = ['/models', '/v1/models', '/info', '/model']

            for endpoint in info_endpoints:
                try:
                    url = f"{self.base_url}{endpoint}"
                    headers = self._prepare_headers()

                    async with self._session.get(url, headers=headers) as response:
                        if response.status == 200:
                            model_data = await response.json()
                            self._parse_model_info(model_data)
                            return

                except Exception:
                    continue

            # If we can't get model info, create default
            self._model_info = ModelInfo(
                name=self.model_name,
                backend_type=BackendType.CUSTOM_API,
                status=ModelStatus.AVAILABLE,
                description=f"Custom API model at {self.base_url}",
                supported_modes=[ProcessingMode.VISION_ONLY, ProcessingMode.HYBRID],
                requires_api_key=bool(self.api_key)
            )

        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")
            # Create default model info
            self._model_info = ModelInfo(
                name=self.model_name,
                backend_type=BackendType.CUSTOM_API,
                status=ModelStatus.AVAILABLE,
                description=f"Custom API model at {self.base_url}",
                supported_modes=[ProcessingMode.VISION_ONLY, ProcessingMode.HYBRID],
                requires_api_key=bool(self.api_key)
            )

    def _parse_model_info(self, model_data: Dict[str, Any]) -> None:
        """Parse model information from API response."""
        try:
            # Handle different API response formats
            if 'data' in model_data and isinstance(model_data['data'], list):
                # OpenAI-like format
                if model_data['data']:
                    model_info = model_data['data'][0]
                    self.model_name = model_info.get('id', self.model_name)
                    description = model_info.get('object', 'model')
            elif 'model' in model_data:
                # Simple format
                self.model_name = model_data['model']
                description = model_data.get('description', 'Custom vision model')
            else:
                description = 'Custom vision model'

            self._model_info = ModelInfo(
                name=self.model_name,
                backend_type=BackendType.CUSTOM_API,
                status=ModelStatus.AVAILABLE,
                description=description,
                supported_modes=[ProcessingMode.VISION_ONLY, ProcessingMode.HYBRID],
                requires_api_key=bool(self.api_key)
            )

        except Exception as e:
            logger.warning(f"Failed to parse model info: {e}")
            # Fall back to default model info

    async def process_page(
        self,
        request: ProcessingRequest,
        progress_callback: Optional[callable] = None
    ) -> ProcessingResponse:
        """
        Process a page using the custom API.

        Args:
            request: ProcessingRequest with page data
            progress_callback: Optional progress callback

        Returns:
            ProcessingResponse with markdown content
        """
        try:
            if not self._session:
                await self.initialize()

            # Prepare request
            api_request = await self._prepare_api_request(request)

            # Call API with retry logic
            api_response = await self._call_api_with_retry(api_request)

            # Parse response
            response = await self._parse_api_response(api_response, request)

            return response

        except Exception as e:
            logger.error(f"Custom API processing failed: {e}")
            return ProcessingResponse(
                markdown="",
                confidence=0.0,
                processing_time=0.0,
                model_used=self.model_name,
                error_message=str(e),
                elements_detected=[]
            )

    async def _prepare_api_request(self, request: ProcessingRequest) -> Dict[str, Any]:
        """Prepare request for the custom API."""
        if self.custom_request_handler:
            # Use custom request handler
            if callable(self.custom_request_handler):
                return await self.custom_request_handler(request, self.config)
            else:
                # Load custom handler module
                module_path, function_name = self.custom_request_handler.rsplit('.', 1)
                module = __import__(module_path, fromlist=[function_name])
                handler = getattr(module, function_name)
                return await handler(request, self.config)

        # Default request preparation based on format
        if self.request_format == 'openai':
            return await self._prepare_openai_request(request)
        elif self.request_format == 'anthropic':
            return await self._prepare_anthropic_request(request)
        else:
            return await self._prepare_generic_request(request)

    async def _prepare_openai_request(self, request: ProcessingRequest) -> Dict[str, Any]:
        """Prepare OpenAI-compatible request."""
        # Prepare messages
        messages = []

        # System message
        messages.append({
            "role": "system",
            "content": request.prompt or "Convert this PDF page to well-formatted markdown."
        })

        # User message with image
        user_content = []

        # Add text content if available
        if request.text_content:
            user_content.append({
                "type": "text",
                "text": f"Extracted text: {request.text_content}"
            })

        # Add image
        if request.image_path and request.image_path.exists():
            image_data = await self._encode_image(request.image_path)
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                }
            })

        messages.append({
            "role": "user",
            "content": user_content
        })

        return {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p
        }

    async def _prepare_anthropic_request(self, request: ProcessingRequest) -> Dict[str, Any]:
        """Prepare Anthropic-compatible request."""
        # Prepare messages
        messages = []

        # User message with image
        user_content = []

        # Add text
        text_content = request.prompt or "Convert this PDF page to well-formatted markdown."
        if request.text_content:
            text_content = f"Extracted text: {request.text_content}\n\n{text_content}"

        user_content.append({
            "type": "text",
            "text": text_content
        })

        # Add image
        if request.image_path and request.image_path.exists():
            image_data = await self._encode_image(request.image_path)
            user_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_data
                }
            })

        messages.append({
            "role": "user",
            "content": user_content
        })

        return {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p
        }

    async def _prepare_generic_request(self, request: ProcessingRequest) -> Dict[str, Any]:
        """Prepare generic request."""
        request_data = {
            "model": self.model_name,
            "prompt": request.prompt or "Convert this PDF page to well-formatted markdown.",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p
        }

        # Add text content
        if request.text_content:
            request_data["text_content"] = request.text_content

        # Add image
        if request.image_path and request.image_path.exists():
            if self.image_format == 'base64':
                image_data = await self._encode_image(request.image_path)
                request_data["image"] = f"data:image/jpeg;base64,{image_data}"
            else:
                request_data["image_path"] = str(request.image_path)

        return request_data

    async def _encode_image(self, image_path: Path) -> str:
        """Encode image as base64."""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    async def _call_api_with_retry(self, api_request: Dict[str, Any]) -> Dict[str, Any]:
        """Call API with retry logic."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                headers = self._prepare_headers()

                # Determine endpoint and method
                if self.request_format == 'openai':
                    endpoint = '/v1/chat/completions'
                elif self.request_format == 'anthropic':
                    endpoint = '/v1/messages'
                else:
                    endpoint = self.config.get('endpoint', '/process')

                url = f"{self.base_url}{endpoint}"

                # Make request
                async with self._session.post(url, json=api_request, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise InferenceError(
                            f"Custom API error: {response.status} - {error_text}",
                            model_name=self.model_name
                        )

            except asyncio.TimeoutError:
                last_exception = InferenceError("Custom API request timed out", model_name=self.model_name)
                logger.warning(f"Custom API request timeout (attempt {attempt + 1}/{self.max_retries + 1})")

            except Exception as e:
                last_exception = e
                logger.warning(f"Custom API request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")

            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff

        raise last_exception or InferenceError("All custom API retry attempts failed", model_name=self.model_name)

    async def _parse_api_response(self, api_response: Dict[str, Any], request: ProcessingRequest) -> ProcessingResponse:
        """Parse API response into ProcessingResponse."""
        if self.custom_response_handler:
            # Use custom response handler
            if callable(self.custom_response_handler):
                return await self.custom_response_handler(api_response, request)
            else:
                # Load custom handler module
                module_path, function_name = self.custom_response_handler.rsplit('.', 1)
                module = __import__(module_path, fromlist=[function_name])
                handler = getattr(module, function_name)
                return await handler(api_response, request)

        # Default response parsing based on format
        if self.response_format == 'openai' or 'choices' in api_response:
            return self._parse_openai_response(api_response, request)
        elif self.response_format == 'anthropic' or 'content' in api_response:
            return self._parse_anthropic_response(api_response, request)
        else:
            return self._parse_generic_response(api_response, request)

    def _parse_openai_response(self, api_response: Dict[str, Any], request: ProcessingRequest) -> ProcessingResponse:
        """Parse OpenAI-compatible response."""
        try:
            choice = api_response['choices'][0]
            content = choice['message']['content'].strip()

            # Extract metadata
            usage = api_response.get('usage', {})
            processing_time = api_response.get('created', 0)  # OpenAI doesn't provide timing

            # Estimate confidence and detect elements
            confidence = self._estimate_confidence(content)
            elements_detected = self._detect_elements(content)

            return ProcessingResponse(
                markdown=content,
                confidence=confidence,
                processing_time=processing_time,
                model_used=self.model_name,
                elements_detected=elements_detected,
                metadata={
                    'input_tokens': usage.get('prompt_tokens', 0),
                    'output_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0),
                    'finish_reason': choice.get('finish_reason')
                }
            )

        except Exception as e:
            raise InferenceError(f"Failed to parse OpenAI response: {e}", model_name=self.model_name)

    def _parse_anthropic_response(self, api_response: Dict[str, Any], request: ProcessingRequest) -> ProcessingResponse:
        """Parse Anthropic-compatible response."""
        try:
            content_block = api_response['content'][0]
            content = content_block['text'].strip()

            # Extract metadata
            usage = api_response.get('usage', {})

            # Estimate confidence and detect elements
            confidence = self._estimate_confidence(content)
            elements_detected = self._detect_elements(content)

            return ProcessingResponse(
                markdown=content,
                confidence=confidence,
                processing_time=0.0,  # Anthropic doesn't provide timing
                model_used=self.model_name,
                elements_detected=elements_detected,
                metadata={
                    'input_tokens': usage.get('input_tokens', 0),
                    'output_tokens': usage.get('output_tokens', 0),
                    'stop_reason': api_response.get('stop_reason')
                }
            )

        except Exception as e:
            raise InferenceError(f"Failed to parse Anthropic response: {e}", model_name=self.model_name)

    def _parse_generic_response(self, api_response: Dict[str, Any], request: ProcessingRequest) -> ProcessingResponse:
        """Parse generic response."""
        try:
            # Try to extract content from common fields
            content = (
                api_response.get('content') or
                api_response.get('text') or
                api_response.get('response') or
                api_response.get('output') or
                str(api_response)
            )

            content = str(content).strip()

            # Estimate confidence and detect elements
            confidence = self._estimate_confidence(content)
            elements_detected = self._detect_elements(content)

            return ProcessingResponse(
                markdown=content,
                confidence=confidence,
                processing_time=0.0,
                model_used=self.model_name,
                elements_detected=elements_detected,
                metadata={
                    'raw_response_keys': list(api_response.keys())
                }
            )

        except Exception as e:
            raise InferenceError(f"Failed to parse generic response: {e}", model_name=self.model_name)

    def _prepare_headers(self) -> Dict[str, str]:
        """Prepare HTTP headers for API requests."""
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'VisionPDF/1.0'
        }

        # Add custom headers
        headers.update(self.headers)

        # Add authentication
        if self.api_key:
            if self.auth_type == 'bearer':
                headers['Authorization'] = f'Bearer {self.api_key}'
            elif self.auth_type == 'api_key':
                headers['X-API-Key'] = self.api_key
            elif self.auth_type == 'basic':
                import base64
                auth_string = base64.b64encode(f':{self.api_key}'.encode()).decode()
                headers['Authorization'] = f'Basic {auth_string}'

        return headers

    def _estimate_confidence(self, response_text: str) -> float:
        """Estimate confidence in response quality."""
        if not response_text or not response_text.strip():
            return 0.0

        confidence = 0.5

        # Check for markdown elements
        if '##' in response_text or '###' in response_text:
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

    def __repr__(self) -> str:
        """String representation of the backend."""
        return f"CustomAPIBackend(base_url='{self.base_url}', model='{self.model_name}')"