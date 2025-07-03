# docs/conf.py

"""
Configuration file for the Sphinx documentation builder.

This file is execfile()d with the current directory set to its containing dir.
"""

import os
import sys
# sys.path.insert(0, os.path.abspath('.')) # If extensions are in this dir
sys.path.insert(0, os.path.abspath('..')) # To find the pyvoi package

# -- Project information -----------------------------------------------------
project = 'pyVOI'
copyright = '2024, Your Name / Organization Here' # Update with actual
author = 'Your Name / Organization Here' # Update with actual

# The full version, including alpha/beta/rc tags
# Attempt to get version from pyvoi package itself
try:
    from pyvoi import __version__ as release
except ImportError:
    release = '0.1.0' # Fallback, update as needed

version = '.'.join(release.split('.')[:2]) # The short X.Y version


# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',      # Include documentation from docstrings
    'sphinx.ext.autosummary',  # Create summary tables from autodoc
    'sphinx.ext.napoleon',     # Support for NumPy and Google style docstrings
    'sphinx.ext.intersphinx',  # Link to other projects' documentation
    'sphinx.ext.viewcode',     # Add links to source code
    'sphinx.ext.githubpages',  # Helps with GitHub Pages deployment
    'sphinx_rtd_theme',      # ReadTheDocs theme
    'sphinx.ext.mathjax',      # Render math equations (optional)
    # 'nbsphinx',              # For including Jupyter notebooks (optional, needs separate install)
    # 'myst_parser',           # For Markdown support (optional, needs separate install)
]

# Autodoc settings
autodoc_member_order = 'bysource' # Order members by source order
autosummary_generate = True       # Turn on remote generation for autosummary

# Napoleon settings (for NumPy and Google docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True # Include __init__ docstrings
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False # Requires sphinx-autodoc-typehints if True
napoleon_type_aliases = None
napoleon_attr_annotations = True


# Intersphinx mapping (example: link to Python, NumPy, SciPy, Pandas docs)
intersphinx_mapping = {
    'python': ('https.docs.python.org/3', None),
    'numpy': ('https.numpy.org/doc/stable/', None),
    'scipy': ('https.docs.scipy.org/doc/scipy/', None),
    'pandas': ('https.pandas.pydata.org/pandas-docs/stable/', None),
    'matplotlib': ('https.matplotlib.org/stable/', None),
    'sklearn': ('https.scikit-learn.org/stable/', None),
    'pymc': ('https.www.pymc.io/projects/docs/en/stable/', None),
    # 'jax': ('https.jax.readthedocs.io/en/latest/', None), # If JAX docs are Sphinx
}

templates_path = ['_templates']    # Directory for custom templates
source_suffix = '.rst'             # Default source file type (.md also possible with myst_parser)
master_doc = 'index'               # The master toctree document. (ชื่อเดิมคือ master_doc, เปลี่ยนเป็น root_doc ใน Sphinx 4.0+)
# root_doc = 'index' # For Sphinx 4.0+

language = 'en'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store'] # Patterns to exclude

pygments_style = 'sphinx'          # Syntax highlighting style


# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'    # Popular ReadTheDocs theme
# html_logo = "_static/logo.png"   # If you have a logo
html_static_path = ['_static']     # Directory for static files (CSS, images)
# html_css_files = ['custom.css']  # If you have custom CSS

# Theme options are theme-specific
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    # 'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}


# -- Options for HTMLHelp output ---------------------------------------------
htmlhelp_basename = 'pyVOIdoc'


# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    # 'papersize': 'letterpaper',
    # 'pointsize': '10pt',
    # 'preamble': '',
    # 'figure_align': 'htbp',
}
latex_documents = [
    (master_doc, 'pyVOI.tex', 'pyVOI Documentation',
     author, 'manual'),
]

# -- Options for manual page output ------------------------------------------
man_pages = [
    (master_doc, 'pyvoi', 'pyVOI Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (master_doc, 'pyVOI', 'pyVOI Documentation',
     author, 'pyVOI', 'One line description of project.',
     'Miscellaneous'),
]

# -- Options for Epub output -------------------------------------------------
epub_title = project
epub_exclude_files = ['search.html']


# -- Extension configuration -------------------------------------------------
# Example: nbsphinx configuration (if using Jupyter notebooks)
# nbsphinx_execute = 'never' # 'auto', 'always', 'never'
# nbsphinx_allow_errors = True

# If using myst_parser for Markdown
# source_suffix = {
#     '.rst': 'restructuredtext',
#     '.txt': 'markdown',
#     '.md': 'markdown',
# }

# This function can be used to dynamically skip members during autodoc
# def autodoc_skip_member(app, what, name, obj, skip, options):
#     # Example: skip private members not starting with an underscore (if any)
#     # if what == "method" and not name.startswith("_") and name.startswith("__"):
#     #     return True
#     return skip

# def setup(app):
#     app.connect("autodoc-skip-member", autodoc_skip_member)

# Add path for custom Sphinx extensions if any
# sys.path.append(os.path.abspath('_extensions'))
# extensions.append('my_custom_extension')

print("Sphinx conf.py loaded.")
