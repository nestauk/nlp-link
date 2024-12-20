# 🖇️ NLP Link

NLP Link finds the most similar word (or sentences) in a reference list to an inputted word. For example, if you are trying to find which word is most similar to 'puppies' from a reference list of `['cats', 'dogs', 'rats', 'birds']`, nlp-link will return 'dogs'.

Another functionality of this package is using the linking methodology to find the [SOC](https://www.ons.gov.uk/methodology/classificationsandstandards/standardoccupationalclassificationsoc) code most similar to an inputted job title. More on this [here](./page1.md).

## 🔨 Usage

Install the package using pip:

```bash
pip install nlp-link
```

### Basic usage

> ⏳ **NOTE:** The first time you import `NLPLinker` in your environment it will take some time (around a minute) to load.

Match two lists in python:

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

### Extended usage

Match using dictionary inputs (where the key is a unique ID):

```python

from nlp_link.linker import NLPLinker

nlp_link = NLPLinker()

# dict inputs
reference_data = {'a': 'cats', 'b': 'dogs', 'd': 'rats', 'e': 'birds'}
input_data = {'x': 'owls', 'y': 'feline', 'z': 'doggies', 'za': 'dogs', 'zb': 'chair'}
nlp_link.load(reference_data)
matches = nlp_link.link_dataset(input_data)
# Top match output
print(matches)

```

Which outputs:

```
  input_id input_text reference_id reference_text  similarity
0        x       owls       e     birds    0.613577
1        y     feline       a      cats    0.669633
2        z    doggies       b      dogs    0.757443
3       za       dogs       b      dogs    1.000000
4       zb      chair       a      cats    0.331178

```

Output the top n most similar reference word matches using the `top_n` argument (`format_output` needs to be set to False for this):

```python

from nlp_link.linker import NLPLinker

nlp_link = NLPLinker()

reference_data = {'a': 'cats', 'b': 'dogs', 'c': 'kittens', 'd': 'rats', 'e': 'birds'}
input_data = {'x': 'pets', 'y': 'feline'}
nlp_link.load(reference_data)
matches = nlp_link.link_dataset(input_data, top_n=2, format_output=False)
# Top match output
print(matches)
# Format output for ease of reading
print({input_data[k]: [reference_data[r] for r, _ in v] for k,v in matches.items()})
```

Which will output:

```
{'x': [['b', 0.8171109], ['a', 0.7650396]], 'y': [['a', 0.6696329], ['c', 0.5778763]]}
{'pets': ['dogs', 'cats'], 'feline': ['cats', 'kittens']}
```

The `drop_most_similar` argument can be set to True if you don't want to output the most similar match - this might be the case if you were comparing a list with itself. For this you would run `nlp_link.link_dataset(input_data, drop_most_similar=True)`.

## References

https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

https://www.ons.gov.uk/methodology/classificationsandstandards/standardoccupationalclassificationsoc

https://www.ons.gov.uk/methodology/classificationsandstandards/standardoccupationalclassificationsoc/soc2020/soc2020volume2codingrulesandconventions
