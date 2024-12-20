# 🖇️ NLP Link

NLP Link finds the most similar word (or sentences) in a reference list to an inputted word. For example, if you are trying to find which word is most similar to 'puppies' from a reference list of `['cats', 'dogs', 'rats', 'birds']`, nlp-link will return 'dogs'.

# 🗺️ SOC Mapper

Another functionality of this package is using the linking methodology to find the [Standard Occupation Classification (SOC)](https://www.ons.gov.uk/methodology/classificationsandstandards/standardoccupationalclassificationsoc) code most similar to an inputted job title. More on this [here](https://github.com/nestauk/nlp-link/blob/main/nlp_link/soc_mapper/README.md).

## 🔨 Usage

Install the package using pip:

```bash
pip install nlp-link
```

### Basic usage

> ⏳ **NOTE:** The first time you import `NLPLinker` in your environment it will take some time (around a minute) to load.

Match two lists of words or sentences in python:

```python

from nlp_link.linker import NLPLinker

nlp_link = NLPLinker()

# list inputs
input_data = ['owls', 'feline', 'doggies', 'dogs','chair']
reference_data = ['cats', 'dogs', 'rats', 'birds']
nlp_link.load(reference_data)
matches = nlp_link.link_dataset(input_data)
# Top match output
print(matches)

```

Which outputs:

```
   input_id input_text  reference_id reference_text  similarity
0         0       owls        3     birds    0.613577
1         1     feline        0      cats    0.669633
2         2    doggies        1      dogs    0.757443
3         3       dogs        1      dogs    1.000000
4         4      chair        0      cats    0.331178

```

These results show the most similar word from the `reference_data` list to each word in the `input_data` list. The word 'dogs' was found across both lists, so it had a similarity score of 1, 'doggies' was matched to 'dogs' since these words are very similar. The inputted word 'chair' had no words that were very similar - the most similar was 'cats' with a low similarity score.

> 🔍 **INFO:** Semantic similarity scores are between 0 and 1, with 0 being very unsimilar, and 1 being exactly the same. This value is calculated by utilising [a large model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) trained on data sets of sentence pairs from various websites (including Reddit comments and WikiHow). The model learns the semantic rules which link the pairs of sentences - e.g. it will learn synonyms. In the above example the reason 'chair' matches most similarly to 'cats' might be because the model learned that "cats" are often mentioned in relation to "chairs" (e.g. sitting on them) compared to dogs, rats, or birds.

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

This nested list gives information about the most similar SOC codes for each of the three inputted job titles. The most similar extended SOC for "data scientist" was 'Statistical data scientists - 2433/04'.

More about this output format is explained in the [SOCMapper page](https://github.com/nestauk/nlp-link/blob/main/nlp_link/soc_mapper/README.md#soc_output).

## Contributing

The instructions here are for those contributing to the repo.

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

## References

https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

https://www.ons.gov.uk/methodology/classificationsandstandards/standardoccupationalclassificationsoc

https://www.ons.gov.uk/methodology/classificationsandstandards/standardoccupationalclassificationsoc/soc2020/soc2020volume2codingrulesandconventions
