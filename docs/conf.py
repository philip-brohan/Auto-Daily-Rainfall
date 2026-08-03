project = "Auto Daily Rainfall"
copyright = "2026, Philip Brohan"
author = "Philip Brohan"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.githubpages",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_heading_anchors = 3

master_doc = "index"
exclude_patterns = ["_build"]

html_theme = "sphinxdoc"
html_title = "Auto Daily Rainfall"
html_short_title = "ADR"
html_static_path = ["_static"]
html_sidebars = {"**": ["globaltoc.html", "sourcelink.html"]}
html_use_index = False
html_show_sphinx = False
html_show_copyright = False
html_logo = "logo/ADR_logo.png"
