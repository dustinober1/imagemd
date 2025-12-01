# Air-Gapped Environment Setup

VisionPDF is designed to work seamlessly in air-gapped environments where internet access is restricted. This guide explains how to configure VisionPDF to use a custom API endpoint (e.g., a locally hosted LLM) and your own API key.

## Prerequisites

1.  **Local Vision Language Model**: You must have a vision language model hosted within your network. This could be:
    *   A local deployment of [Ollama](https://github.com/ollama/ollama).
    *   A local deployment of [llama.cpp](https://github.com/ggerganov/llama.cpp) server.
    *   A custom inference server (e.g., vLLM, TGI) exposing an OpenAI-compatible API.

2.  **VisionPDF Package**: Ensure `vision-pdf` is installed in your environment. You may need to transfer the wheel file or install from a local PyPI mirror.

## Configuration

You can configure VisionPDF to use your custom endpoint using Python code or environment variables.

### Method 1: Python Configuration

Use the `CustomAPIBackend` by specifying `BackendType.CUSTOM_API` and providing the necessary configuration in `backend_config`.

```python
import asyncio
from vision_pdf import VisionPDF, BackendType, ProcessingMode

async def main():
    # Configuration for the custom backend
    backend_config = {
        # The URL of your local model server
        "base_url": "http://your-internal-server:8080",

        # Your API key (if required by your server)
        "api_key": "your-internal-api-key",

        # The name of the model to use
        "model_name": "your-custom-vision-model",

        # Request/Response format (usually 'openai' for compatible servers)
        "request_format": "openai",
        "response_format": "openai"
    }

    # Initialize VisionPDF
    converter = VisionPDF(
        backend_type=BackendType.CUSTOM_API,
        backend_config=backend_config,
        # 'vision_only' relies purely on the model, 'hybrid' uses OCR/Text extraction + Model
        processing_mode=ProcessingMode.VISION_ONLY
    )

    # Convert PDF
    print("Converting PDF...")
    markdown = await converter.convert_pdf("document.pdf")

    # Save to file
    with open("output.md", "w") as f:
        f.write(markdown)
    print("Conversion complete!")

if __name__ == "__main__":
    asyncio.run(main())
```

### Method 2: Environment Variables

You can also set the configuration via environment variables, which is useful for containerized deployments.

```bash
export VISIONPDF_DEFAULT_BACKEND=custom_api
export VISIONPDF_CUSTOM_API_BASE_URL=http://your-internal-server:8080
export VISIONPDF_CUSTOM_API_KEY=your-internal-api-key
# Note: Additional custom config might need to be passed via code or config file
```

## Using Config File

You can also use a YAML configuration file:

```yaml
processing:
  mode: vision_only

backends:
  custom_api:
    enabled: true
    config:
      base_url: "http://your-internal-server:8080"
      api_key: "your-internal-api-key"
      model_name: "your-custom-vision-model"
      request_format: "openai"
      response_format: "openai"

default_backend: custom_api
```

Load it in your code:

```python
from vision_pdf import VisionPDF, VisionPDFConfig

config = VisionPDFConfig.load_from_file("config.yaml")
converter = VisionPDF(config=config)
# ...
```

## Troubleshooting

*   **Connection Error**: Ensure your internal server is reachable from the machine running VisionPDF. Verify the `base_url` includes the protocol (http/https) and port.
*   **Model Not Found**: Check that `model_name` matches a model available on your server.
*   **Format Issues**: If your server doesn't support OpenAI-compatible API, you may need to implement a custom request/response handler or use `request_format="custom"`.

## Security

*   In an air-gapped environment, ensure that any sensitive data in PDFs is handled according to your organization's security policies.
*   VisionPDF processes data locally or sends it only to the configured `base_url`. It does not make unauthorized external calls.
