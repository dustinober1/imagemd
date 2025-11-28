"""
Sphinx configuration file for VisionPDF documentation.

This file configures the Sphinx documentation build process,
including extensions, theme settings, and project metadata.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path for autodoc
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# -- Project information -------------------------------------------------
project = 'VisionPDF'
copyright = '2023, VisionPDF Team'
author = 'VisionPDF Team'
release = '1.0.0'
version = '1.0.0'

# -- General configuration ------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',  # Google/NumPy style docstrings
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx_rtd_theme',  # Read the Docs theme
    'myst_parser',  # Markdown support
]

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

# autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# autosummary settings
autosummary_generate = True
autosummary_imported_members = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# MyST parser settings for Markdown
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# -- Options for HTML output ---------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']

# Theme options
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# HTML output settings
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# -- Options for LaTeX output --------------------------------------------
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': r'''
\usepackage{hyperref}
\usepackage{alphabeta}
''',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'VisionPDF.tex', 'VisionPDF Documentation',
     'VisionPDF Team', 'manual'),
]

# -- Options for manual page output ---------------------------------------
man_pages = [
    (master_doc, 'visionpdf', 'VisionPDF Documentation',
     [author], 1)
]

# -- Options for Texinfo output ------------------------------------------
texinfo_documents = [
    (master_doc, 'VisionPDF', 'VisionPDF Documentation',
     author, 'VisionPDF', 'One line description of project.',
     'Miscellaneous'),
]

# -- Extension configuration -----------------------------------------------
# autoclass content
autoclass_content = 'both'

# autodoc type hints
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# Napoleon settings for better API documentation
napoleon_use_ivar = False
napoleon_use_rtype = True

# Coverage settings
coverage_show_missing_items = True

# Doctest settings
doctest_global_setup = '''
import sys
sys.path.insert(0, '.')
'''

# -- Custom setup ---------------------------------------------------------
def setup(app):
    """Custom Sphinx app setup."""
    app.add_css_file('custom.css')