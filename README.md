# nlp-link

A python package to semantically link two lists of texts.

## Set-up

In setting up this project we ran:

```
conda create --name nlp-link pip python=3.9
conda activate nlp-link
pip install poetry
pip install pre-commit black
pre-commit install
```

```
poetry init

```

```
poetry install

```

## Usage

```
from nlp_link.linker import NLPLinker

nlp_link = NLPLinker()

# dict inputs
comparison_data = {'a': 'cats', 'b': 'dogs', 'd': 'rats', 'e': 'birds'}
input_data = {'x': 'owls', 'y': 'feline', 'z': 'doggies', 'za': 'dogs', 'zb': 'chair'}
nlp_link.load(comparison_data)
matches = nlp_link.link_dataset(input_data)
# Top match output
print(matches)

# list inputs
comparison_data = ['cats', 'dogs', 'rats', 'birds']
input_data = ['owls', 'feline', 'doggies', 'dogs','chair']
nlp_link.load(comparison_data)
matches = nlp_link.link_dataset(input_data)
# Top match output
print(matches)
```

## Tests

To run tests:

```
poetry run pytest tests/
```

## Documentation

Docs for this repo are automatically published to gh-pages branch via. Github actions after a PR is merged into main. We use Material for MkDocs for these. Nothing needs to be done to update these.

However, if you are editing the docs you can test them out locally by running

```
cd docs
<!-- pip install -r docs/requirements.txt -->
mkdocs serve
```
