# nlp-link
A python package to semantically link two lists of texts.


## Set-up

In setting up this project we ran:
```
conda create --name nlp-link pip python=3.9
conda activate nlp-link
pip install poetry
```

```
poetry init

```

```
poetry install

```

## Tests

To run tests:

```
poetry run pytest tests/
```

## Documentation

Docs for this repo are automatically published to gh-pages branch via. Github actions after a PR is merged into dev. We use Material for MkDocs for these. Nothing needs to be done to update these.

However, if you are editing the docs you can test them out locally by running

```
cd guidelines
pip install -r docs/requirements.txt  
mkdocs serve
```
