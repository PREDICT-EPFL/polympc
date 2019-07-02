# Build documentation using MkDocs

This project uses [MkDocs](https://www.mkdocs.org) for building the documentation site.

## Installation

```bash
pip install mkdocs python-markdown-math
```

## Build and serve locally

To build and run the site locally run
```bash
mkdocs build
mkdocs serve
```

## Deploy on GitHub Pages

**WARNING: DO NOT DEPLOY FROM MAIN COPY. DATA CAN BE LOST!**

First make a copy of the repository from where to build and deploy the documentation.

```bash
cd polympc-copy
mkdocs gh-deploy
```

This will create a branch `gh-pages`, build the site, commit the result and push to github.
