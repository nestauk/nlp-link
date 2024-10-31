# 🖇️ NLP Link

NLP Link finds the most similar word (or words) in a reference list to an inputted word. For example, if you are trying to find which word is most similar to 'puppies' from a reference list of `['cats', 'dogs', 'rats', 'birds']`, nlp-link will return 'dogs'.

# 🗺️ SOC Mapper

Another functionality of this package is using the linking methodology to find the [Standard Occupation Classification (SOC)](https://www.ons.gov.uk/methodology/classificationsandstandards/standardoccupationalclassificationsoc) code most similar to an inputted job title. More on this [here](https://github.com/nestauk/nlp-link/blob/main/docs/page1.md).

## 🔨 Usage

Install the package using pip:

```bash
pip install nlp-link
```

### Basic usage

Match two lists in python:

```python

from nlp_link.linker import NLPLinker

nlp_link = NLPLinker()

# list inputs
comparison_data = ['cats', 'dogs', 'rats', 'birds']
input_data = ['owls', 'feline', 'doggies', 'dogs','chair']
nlp_link.load(comparison_data)
matches = nlp_link.link_dataset(input_data)
# Top match output
print(matches)

```

Which outputs:

```
   input_id input_text  link_id link_text  similarity
0         0       owls        3     birds    0.613577
1         1     feline        0      cats    0.669633
2         2    doggies        1      dogs    0.757443
3         3       dogs        1      dogs    1.000000
4         4      chair        0      cats    0.331178

```

### SOC Mapping

Match a list of job titles to SOC codes:

```
from nlp_link.soc_mapper.soc_map import SOCMapper

soc_mapper = SOCMapper()
soc_mapper.load()
job_titles=["data scientist", "Assistant nurse", "Senior financial consultant - London"]

soc_mapper.get_soc(job_titles, return_soc_name=True)
```

Which will output

```
[((('2433/04', 'Statistical data scientists'), ('2433', 'Actuaries, economists and statisticians'), '2425'), 'Data scientist'), ((('6131/99', 'Nursing auxiliaries and assistants n.e.c.'), ('6131', 'Nursing auxiliaries and assistants'), '6141'), 'Assistant nurse'), ((('2422/02', 'Financial advisers and planners'), ('2422', 'Finance and investment analysts and advisers'), '3534'), 'Financial consultant')]
```

## Contributing

The instructions here are for those contrbuting to the repo.

### Set-up

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

### Tests

To run tests:

```
poetry run pytest tests/
```

### Documentation

Docs for this repo are automatically published to gh-pages branch via. Github actions after a PR is merged into main. We use Material for MkDocs for these. Nothing needs to be done to update these.

However, if you are editing the docs you can test them out locally by running

```
cd docs
<!-- pip install -r docs/requirements.txt -->
mkdocs serve
```
