# 🗺️ SOC Mapper

The SOC mapper relies on the [SOC coding index](https://www.ons.gov.uk/methodology/classificationsandstandards/standardoccupationalclassificationsoc/soc2020/soc2020volume2codingrulesandconventions) released by the ONS. This dataset contains over 30,000 job titles with the SOC code.

The `SOCMapper` class in `soc_map.py` maps job title(s) to SOC(s).

## 🔨 Core functionality

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

## 📖 Read more

Read more about the methods and evaluation of the SOCMapper [here](https://github.com/nestauk/nlp-link/blob/main/nlp_link/soc_mapper/README.md).
