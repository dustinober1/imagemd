"""
Command-line interface for VisionPDF.

This module provides a comprehensive CLI for converting PDF documents
to markdown using various backends and processing options.
"""

import asyncio
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

import click

from ..core.processor import VisionPDF
from ..config.settings import VisionPDFConfig, BackendType, ProcessingMode
from ..utils.logging_config import setup_cli_logging
from ..utils.exceptions import VisionPDFError, ValidationError, BackendError

# Setup basic logging for CLI
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output except errors')
@click.option('--log-file', type=click.Path(), help='Log to file instead of console')
@click.pass_context
def cli(ctx, verbose, quiet, log_file):
    """
    VisionPDF: Convert PDF documents to well-formatted markdown using vision language models.

    This tool supports multiple VLM backends including Ollama, llama.cpp, and custom APIs.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Setup logging based on verbosity
    if quiet:
        log_level = logging.ERROR
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    setup_cli_logging(level=log_level, log_file=log_file)

    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet


@cli.command()
@click.argument('input_file', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--output', '-o',
    type=click.Path(),
    help='Output file path (default: input_file.md)'
)
@click.option(
    '--backend', '-b',
    type=click.Choice(['ollama', 'llama_cpp', 'custom_api'], case_sensitive=False),
    default='ollama',
    help='VLM backend to use'
)
@click.option(
    '--model',
    help='Model name to use with the backend'
)
@click.option(
    '--mode', '-m',
    type=click.Choice(['vision_only', 'hybrid', 'text_only'], case_sensitive=False),
    default='hybrid',
    help='Processing mode'
)
@click.option(
    '--base-url',
    help='Base URL for backend API'
)
@click.option(
    '--api-key',
    help='API key for backend authentication'
)
@click.option(
    '--dpi',
    type=int,
    default=300,
    help='DPI for PDF rendering (default: 300)'
)
@click.option(
    '--parallel',
    type=int,
    default=1,
    help='Number of parallel processing workers'
)
@click.option(
    '--preserve-tables/--no-preserve-tables',
    default=True,
    help='Enable table preservation'
)
@click.option(
    '--preserve-math/--no-preserve-math',
    default=True,
    help='Enable mathematical expression preservation'
)
@click.option(
    '--preserve-code/--no-preserve-code',
    default=True,
    help='Enable code block preservation'
)
@click.option(
    '--timeout',
    type=int,
    default=300,
    help='Processing timeout in seconds'
)
@click.option(
    '--config',
    type=click.Path(exists=True, dir_okay=False),
    help='Path to configuration file'
)
@click.option(
    '--progress/--no-progress',
    default=True,
    help='Show progress bar'
)
@click.pass_context
def convert(
    ctx,
    input_file: str,
    output: Optional[str],
    backend: str,
    model: Optional[str],
    mode: str,
    base_url: Optional[str],
    api_key: Optional[str],
    dpi: int,
    parallel: int,
    preserve_tables: bool,
    preserve_math: bool,
    preserve_code: bool,
    timeout: int,
    config: Optional[str],
    progress: bool
):
    """
    Convert a PDF file to markdown.

    INPUT_FILE: Path to the PDF file to convert
    """
    input_path = Path(input_file)

    # Determine output path
    if output is None:
        output_path = input_path.with_suffix('.md')
    else:
        output_path = Path(output)

    try:
        # Run conversion
        asyncio.run(_convert_single(
            input_path=input_path,
            output_path=output_path,
            backend_type=BackendType(backend.lower()),
            model=model,
            processing_mode=ProcessingMode(mode.lower()),
            base_url=base_url,
            api_key=api_key,
            dpi=dpi,
            parallel_workers=parallel,
            preserve_tables=preserve_tables,
            preserve_math=preserve_math,
            preserve_code=preserve_code,
            timeout=timeout,
            config_file=config,
            show_progress=progress and not ctx.obj.get('quiet', False)
        ))

        if not ctx.obj.get('quiet', False):
            click.echo(f"✅ Successfully converted {input_file} to {output_path}")

    except Exception as e:
        click.echo(f"❌ Error converting {input_file}: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('output_dir', type=click.Path())
@click.option(
    '--pattern',
    default='*.pdf',
    help='File pattern to match (default: *.pdf)'
)
@click.option(
    '--recursive', '-r',
    is_flag=True,
    help='Search recursively in subdirectories'
)
@click.option(
    '--backend', '-b',
    type=click.Choice(['ollama', 'llama_cpp', 'custom_api'], case_sensitive=False),
    default='ollama',
    help='VLM backend to use'
)
@click.option(
    '--parallel-files',
    type=int,
    default=1,
    help='Number of files to process in parallel'
)
@click.option(
    '--parallel-pages',
    type=int,
    default=1,
    help='Number of pages to process in parallel within each file'
)
@click.option(
    '--continue-on-error',
    is_flag=True,
    help='Continue processing other files if one fails'
)
@click.option(
    '--config',
    type=click.Path(exists=True, dir_okay=False),
    help='Path to configuration file'
)
@click.pass_context
def batch(
    ctx,
    input_dir: str,
    output_dir: str,
    pattern: str,
    recursive: bool,
    backend: str,
    parallel_files: int,
    parallel_pages: int,
    continue_on_error: bool,
    config: Optional[str]
):
    """
    Convert multiple PDF files to markdown.

    INPUT_DIR: Directory containing PDF files to convert
    OUTPUT_DIR: Directory to save converted markdown files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    try:
        # Find PDF files
        pdf_files = []
        if recursive:
            pdf_files = list(input_path.rglob(pattern))
        else:
            pdf_files = list(input_path.glob(pattern))

        if not pdf_files:
            click.echo(f"No PDF files found matching '{pattern}' in {input_dir}")
            return

        click.echo(f"Found {len(pdf_files)} PDF files to convert")

        # Run batch conversion
        results = asyncio.run(_convert_batch(
            pdf_files=pdf_files,
            output_dir=output_path,
            backend_type=BackendType(backend.lower()),
            parallel_files=parallel_files,
            parallel_pages=parallel_pages,
            config_file=config,
            continue_on_error=continue_on_error,
            show_progress=not ctx.obj.get('quiet', False)
        ))

        # Report results
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful

        if not ctx.obj.get('quiet', False):
            click.echo(f"\n📊 Batch conversion complete:")
            click.echo(f"   ✅ Successful: {successful}")
            click.echo(f"   ❌ Failed: {failed}")

        if failed > 0 and not continue_on_error:
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Batch conversion failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('pdf_file', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--backend', '-b',
    type=click.Choice(['ollama', 'llama_cpp', 'custom_api'], case_sensitive=False),
    default='ollama',
    help='Backend to test'
)
@click.option(
    '--base-url',
    help='Base URL for backend API'
)
@click.option(
    '--api-key',
    help='API key for backend authentication'
)
@click.option(
    '--config',
    type=click.Path(exists=True, dir_okay=False),
    help='Path to configuration file'
)
def test(
    pdf_file: str,
    backend: str,
    base_url: Optional[str],
    api_key: Optional[str],
    config: Optional[str]
):
    """
    Test backend connectivity and basic functionality.

    PDF_FILE: PDF file to use for testing
    """
    try:
        asyncio.run(_test_backend(
            pdf_path=Path(pdf_file),
            backend_type=BackendType(backend.lower()),
            base_url=base_url,
            api_key=api_key,
            config_file=config
        ))

    except Exception as e:
        click.echo(f"❌ Backend test failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    '--backend', '-b',
    type=click.Choice(['ollama', 'llama_cpp', 'custom_api'], case_sensitive=False),
    help='Backend to list models for'
)
@click.option(
    '--base-url',
    help='Base URL for backend API'
)
@click.option(
    '--api-key',
    help='API key for backend authentication'
)
@click.option(
    '--config',
    type=click.Path(exists=True, dir_okay=False),
    help='Path to configuration file'
)
def models(
    backend: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    config: Optional[str]
):
    """
    List available models from backend(s).
    """
    try:
        asyncio.run(_list_models(
            backend_type=BackendType(backend.lower()) if backend else None,
            base_url=base_url,
            api_key=api_key,
            config_file=config
        ))

    except Exception as e:
        click.echo(f"❌ Failed to list models: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    '--stats/--no-stats',
    default=True,
    help='Include cache statistics'
)
@click.option(
    '--config',
    type=click.Path(exists=True, dir_okay=False),
    help='Path to configuration file'
)
def info(stats: bool, config: Optional[str]):
    """Show system and configuration information."""
    try:
        _show_info(stats, config)
    except Exception as e:
        click.echo(f"❌ Failed to get system info: {e}", err=True)
        sys.exit(1)


async def _convert_single(
    input_path: Path,
    output_path: Path,
    backend_type: BackendType,
    model: Optional[str],
    processing_mode: ProcessingMode,
    base_url: Optional[str],
    api_key: Optional[str],
    dpi: int,
    parallel_workers: int,
    preserve_tables: bool,
    preserve_math: bool,
    preserve_code: bool,
    timeout: int,
    config_file: Optional[str],
    show_progress: bool
):
    """Convert a single PDF file."""
    # Load configuration
    config = _load_config(config_file) if config_file else VisionPDFConfig()

    # Configure backend
    backend_config = {}
    if base_url:
        backend_config['base_url'] = base_url
    if api_key:
        backend_config['api_key'] = api_key
    if model:
        backend_config['model'] = model

    # Update config
    config.default_backend = backend_type
    if backend_type.value in config.backends:
        config.backends[backend_type.value].config.update(backend_config)

    # Update processing config
    config.processing.dpi = dpi
    config.processing.parallel_workers = parallel_workers
    config.processing.preserve_tables = preserve_tables
    config.processing.preserve_math = preserve_math
    config.processing.preserve_code = preserve_code
    config.processing.timeout = timeout

    # Create converter
    async with VisionPDF(
        config=config,
        backend_type=backend_type,
        backend_config=backend_config,
        processing_mode=processing_mode
    ) as converter:
        # Progress callback
        def progress_callback(current: int, total: int):
            if show_progress:
                percentage = (current / total) * 100
                click.echo(f"Progress: {current}/{total} pages ({percentage:.1f}%)", nl=False)
                click.echo('\r', nl=False)

        # Convert PDF
        await converter.convert_pdf_to_file(input_path, output_path, progress_callback)

        if show_progress:
            click.echo()  # New line after progress


async def _convert_batch(
    pdf_files: List[Path],
    output_dir: Path,
    backend_type: BackendType,
    parallel_files: int,
    parallel_pages: int,
    config_file: Optional[str],
    continue_on_error: bool,
    show_progress: bool
) -> List[Dict[str, Any]]:
    """Convert multiple PDF files."""
    # Load configuration
    config = _load_config(config_file) if config_file else VisionPDFConfig()
    config.processing.parallel_workers = parallel_pages

    # Create converter
    async with VisionPDF(config=config, backend_type=backend_type) as converter:
        # Process files
        results = []

        with click.progressbar(
            pdf_files,
            label='Converting files',
            show_eta=True,
            show_percent=True
        ) as bar:
            for pdf_file in bar:
                try:
                    # Generate output path
                    output_path = output_dir / f"{pdf_file.stem}.md"

                    # Convert file
                    await converter.convert_pdf_to_file(pdf_file, output_path)

                    results.append({
                        'file': str(pdf_file),
                        'success': True,
                        'output': str(output_path)
                    })

                except Exception as e:
                    results.append({
                        'file': str(pdf_file),
                        'success': False,
                        'error': str(e)
                    })

                    if not continue_on_error:
                        raise e

        return results


async def _test_backend(
    pdf_path: Path,
    backend_type: BackendType,
    base_url: Optional[str],
    api_key: Optional[str],
    config_file: Optional[str]
):
    """Test backend connectivity."""
    click.echo(f"🧪 Testing {backend_type.value} backend...")

    # Load configuration
    config = _load_config(config_file) if config_file else VisionPDFConfig()

    # Configure backend
    backend_config = {}
    if base_url:
        backend_config['base_url'] = base_url
    if api_key:
        backend_config['api_key'] = api_key

    # Test connection
    async with VisionPDF(
        config=config,
        backend_type=backend_type,
        backend_config=backend_config
    ) as converter:
        # Test backend connection
        connection_ok = await converter.test_backend_connection()

        if connection_ok:
            click.echo("✅ Backend connection successful")

            # List available models
            models = await converter.get_available_models()
            if models:
                click.echo(f"📋 Available models: {', '.join(models)}")
            else:
                click.echo("⚠️  No models found")

            # Test conversion with first page only
            click.echo("🔄 Testing conversion...")
            try:
                # Create temporary output
                test_output = pdf_path.parent / f"{pdf_path.stem}_test.md"
                await converter.convert_pdf_to_file(pdf_path, test_output)

                # Check if file was created and has content
                if test_output.exists() and test_output.stat().st_size > 0:
                    click.echo("✅ Test conversion successful")
                    test_output.unlink()  # Clean up
                else:
                    click.echo("❌ Test conversion failed - no output generated")

            except Exception as e:
                click.echo(f"❌ Test conversion failed: {e}")

        else:
            click.echo("❌ Backend connection failed")
            sys.exit(1)


async def _list_models(
    backend_type: Optional[BackendType],
    base_url: Optional[str],
    api_key: Optional[str],
    config_file: Optional[str]
):
    """List available models from backends."""
    # Load configuration
    config = _load_config(config_file) if config_file else VisionPDFConfig()

    # Configure backend
    backend_config = {}
    if base_url:
        backend_config['base_url'] = base_url
    if api_key:
        backend_config['api_key'] = api_key

    backends_to_test = [backend_type] if backend_type else list(BackendType)

    for bt in backends_to_test:
        try:
            click.echo(f"\n🔍 Checking {bt.value} backend...")

            async with VisionPDF(
                config=config,
                backend_type=bt,
                backend_config=backend_config
            ) as converter:
                models = await converter.get_available_models()

                if models:
                    click.echo(f"📋 Available models for {bt.value}:")
                    for model in models:
                        click.echo(f"   • {model}")
                else:
                    click.echo(f"⚠️  No models found for {bt.value}")

        except Exception as e:
            click.echo(f"❌ Failed to get models from {bt.value}: {e}")


def _show_info(include_stats: bool, config_file: Optional[str]):
    """Show system information."""
    click.echo("🔧 VisionPDF System Information")
    click.echo("=" * 40)

    # Basic info
    click.echo(f"📅 Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    click.echo(f"🐍 Python version: {sys.version}")

    # Package info
    try:
        import vision_pdf
        click.echo(f"📦 VisionPDF version: {vision_pdf.__version__}")
    except ImportError:
        click.echo("📦 VisionPDF version: unknown")

    # Configuration
    if config_file:
        click.echo(f"⚙️  Configuration file: {config_file}")
        try:
            config = _load_config(config_file)
            click.echo(f"🎯 Default backend: {config.default_backend.value}")
            click.echo(f"📄 Default DPI: {config.processing.dpi}")
        except Exception as e:
            click.echo(f"❌ Failed to load config: {e}")

    # Cache stats
    if include_stats:
        try:
            config = _load_config(config_file) if config_file else VisionPDFConfig()
            if config.cache.enabled:
                from ..utils.cache import PDFCache
                cache = PDFCache(config)
                stats = cache.get_cache_stats()

                click.echo(f"\n💾 Cache Statistics:")
                click.echo(f"   Enabled: {stats['enabled']}")
                if stats['enabled']:
                    if 'file_cache' in stats:
                        fc = stats['file_cache']
                        click.echo(f"   File cache: {fc['total_entries']} entries, "
                                  f"{fc['total_size_mb']:.1f} MB ({fc['utilization_percent']:.1f}% used)")
                    if 'memory_cache' in stats:
                        mc = stats['memory_cache']
                        click.echo(f"   Memory cache: {mc['current_entries']}/{mc['max_entries']} entries")
            else:
                click.echo(f"\n💾 Cache: Disabled")

        except Exception as e:
            click.echo(f"\n💾 Cache stats unavailable: {e}")


def _load_config(config_file: str) -> VisionPDFConfig:
    """Load configuration from file."""
    config_path = Path(config_file)

    if config_path.suffix.lower() in ['.yaml', '.yml']:
        return VisionPDFConfig.from_yaml_file(config_path)
    elif config_path.suffix.lower() == '.json':
        return VisionPDFConfig.from_json_file(config_path)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n🛑 Operation cancelled by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"💥 Unexpected error: {e}", err=True)
        if '--verbose' in sys.argv or '-v' in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()