"""
Sphinx configuration file for VisionPDF documentation.

This file contains the Sphinx configuration for building the
VisionPDF documentation.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path for autodoc
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# -- Project information -----------------------------------------------------
project = 'VisionPDF'
copyright = '2024, VisionPDF Team'
author = 'VisionPDF Team'
release = '0.1.0'
version = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'myst_parser'  # For Markdown support
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': None,
    '.md': None,
}

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Custom CSS files
html_css_files = [
    'css/custom.css',
]

# Custom JavaScript files
html_js_files = []

# Output file base name for HTML help builder.
htmlhelp_basename = 'VisionPDFdoc'

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'fncychap': '\\usepackage[Bjornstrup]{fncychap}',
    'printindex': '\\footnotesize\\raggedright\\printindex',
}

# Grouping the document tree into LaTeX files. List of tuples
latex_documents = [
    (master_doc, 'VisionPDF.tex', 'VisionPDF Documentation',
     'VisionPDF Team', 'manual'),
]

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pymupdf': ('https://pymupdf.readthedocs.io/en/latest', None),
    'pdfplumber': ('https://pdfplumber.readthedocs.io/en/latest', None),
    'pillow': ('https://pillow.readthedocs.io/en/stable', None),
    'pydantic': ('https://pydantic-docs.helpmanual.io', None),
}

# Coverage settings
coverage_show_missing_items = True

# Doctest settings
doctest_global_setup = '''
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
'''

# -- Custom processing -------------------------------------------------------

def setup(app):
    """Custom Sphinx setup."""
    # Add custom CSS
    app.add_css_file('css/custom.css')


# -- Project-specific configuration -----------------------------------------

# If the documentation is being built for a specific version
# include version-specific information
if os.environ.get('READTHEDOCS') or os.environ.get('BUILDING_DOCS'):
    # Add readthedocs specific settings
    html_context = {
        'display_github': True,
        'github_user': 'visionpdf',
        'github_repo': 'vision-pdf',
        'github_version': 'main/docs/',
        'conf_py_path': '/docs/source/',
    }

# -- Source file exclusions -------------------------------------------------

# Exclude certain patterns from documentation build
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**.ipynb_checkpoints'
]

# -- Numbers and figures ----------------------------------------------------

numfig = True
numfig_secnum_depth = 1

# -- Language and locale ----------------------------------------------------
language = 'en'

# -- Miscellaneous settings -------------------------------------------------
show_authors = True
pygments_style = 'sphinx'