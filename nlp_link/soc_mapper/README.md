# 🗺️ SOCMapper

Key files and folders in this directory are:

1. [soc_map.py](https://github.com/nestauk/nlp-link/blob/main/nlp_link/soc_mapper/soc_map.py): The script containing the `SOCMapper` class.
2. [soc_map_utils.py](https://github.com/nestauk/nlp-link/blob/main/nlp_link/soc_mapper/soc_map_utils.py): Functions for loading data and cleaning job titles for the `SOCMapper` class.
3. [config.yaml](https://github.com/nestauk/nlp-link/blob/main/nlp_link/soc_mapper/config.yaml): The default arguments for the `SOCMapper` class.

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

### Modifications

If you want to match job titles to a different, locally saved, version of a SOC coding index file, you can do this with:

```
from nlp_link.soc_mapper.soc_map import SOCMapper

soc_mapper = SOCMapper(soc_dir = LOCAL_DIR_OF_SOC_CODING_INDEX)
soc_mapper.load(save_embeds = True)
```

Where `LOCAL_DIR_OF_SOC_CODING_INDEX` is the location of your locally saved version of the SOC coding index xlsx file, e.g. `data/soc2020volume2thecodingindexexcel22022024.xlsx`. If `save_embeds = True` then the embeddings will be saved in this same directory.

If this file has different column names from what is outlined in `nlp_link/soc_mapper/config.yaml`, then you can edit them individually by running:

```
from nlp_link.soc_mapper.soc_map import SOCMapper

soc_mapper = SOCMapper(soc_dir = LOCAL_DIR_OF_SOC_CODING_INDEX)
soc_mapper.soc_mapper_config['sheet_name'] = 'Name of new sheet name'
soc_mapper.load(save_embeds = True)

```

<a name="soc_output"></a>

## 📤 Output

The output for one job title is in the format

```
(((SOC 2020 Extension code, SOC 2020 Extension name), (SOC 2020 4-digit code, SOC 2020 4-digit name), SOC 2010 code), job title matched to in SOC data)
```

for example

```
((('2422/02', 'Financial advisors and planners'), ('2422', 'Finance and investment analysts and advisers'), '3534'), 'financial consultant')
```

If the names of the SOC codes aren't needed then you can set `return_soc_name=False`. The variables `soc_mapper.soc_2020_6_dict` and `soc_mapper.soc_2020_4_dict` give the names of each SOC 2020 6 and 4 digit codes.

The following table gives the results of using the SOCMapper function on the job titles in the "Input job title" column.

| Input job title                      | SOC 2020 EXT code | SOC 2020 sub-unit group                   | SOC 2020 unit group                          | SOC 2010 code | SOC data job title   |
| ------------------------------------ | ----------------- | ----------------------------------------- | -------------------------------------------- | ------------- | -------------------- |
| data scientist                       | 2433/04           | Statistical data scientists               | Actuaries, economists and statisticians      | 2425          | Data scientist       |
| Assistant nurse                      | 6131/99           | Nursing auxiliaries and assistants n.e.c. | Nursing auxiliaries and assistants           | 6141          | Assistant nurse      |
| Senior financial consultant - London | 2422/02           | Financial advisors and planners           | Finance and investment analysts and advisers | 3534          | Financial consultant |

## 🖊️ Methodology

The SOCMapper works by finding the semantically closest job titles between the inputed job titles and the job titles in the ONS SOC dataset. An overview of the methodology is in the diagram below.

<p align="center">
  <img src="SOCMapper_overview.jpg" />
</p>

**Step 1:** We clean the inputted job title. This cleaning involves removing words which describe the job conditions but not the job title; e.g. removing common placenames or words like "part-time".

For example, if our inputted job adverts were

```
["Data Scientist - London", "Electric motor assembler - part time", "Data visualisation developer £30k per annum"]
```

these would be cleaned to

```
['Data Scientist', 'Electric motor assembler', 'Data visualisation developer']
```

**Step 2:** We process the [ONS SOC coding index](https://www.ons.gov.uk/methodology/classificationsandstandards/standardoccupationalclassificationsoc/standardoccupationalclassificationsocextensionproject). An example of this dataset is:

| SOC 2010 | SOC 2020 | SOC 2020 Ext Code | INDEXOCC           | ADD       | IND         | INDEXOCC NATURAL WORD ORDER | SOC 2020 UNIT GROUP DESCRIPTIONS                   | SUB-UNIT GROUP DESCRIPTIONS                               |
| -------- | -------- | ----------------- | ------------------ | --------- | ----------- | --------------------------- | -------------------------------------------------- | --------------------------------------------------------- |
| 2425     | 2433     | 2433/04           | Scientist, data    |           |             | Data scientist              | Actuaries, economists and statisticians            | Statistical data scientists                               |
| 2136     | 2134     | 2134/99           | Analyst, data      | computing |             | data analyst                | Programmers and software development professionals | Programmers and software development professionals n.e.c. |
| 3539     | 3544     | 3544/00           | Analyst, data      |           |             | data analyst                | Data analysts                                      | Data analysts                                             |
| 2136     | 2134     | 2134/03           | Developer, analyst |           |             | analyst developer           | Programmers and software development professionals | Software developers                                       |
| 8139     | 8149     | 8149/00           | Assembler, meter   |           |             | meter assembler             | Assemblers and routine operatives n.e.c.           | Assemblers and routine operatives n.e.c.                  |
| 8131     | 8141     | 8141/00           | Assembler, motor   | electric  |             | motor assembler             | Assemblers (electrical and electronic products)    | Assemblers (electrical and electronic products)           |
| 8132     | 8142     | 8142/02           | Assembler, motor   |           | engineering | motor assembler             | Assemblers (vehicles and metal goods)              | Vehicle and vehicle part assemblers                       |

We can combine the `INDEXOCC NATURAL WORD ORDER`, `ADD` and `IND` columns to create unique job titles. The dictionary of unique job titles to SOC information would be:

```
{"data scientist": ("2433/04", "2433", "2425"), "data analyst computing": ("2134/99", "2134", "2136"), "data analyst": ("3544/00", "3544", "3539"), "analyst developer": ("2134/03", "2134", "2136"), "meter assembler": ("8149/00", "8149", "8139"), "motor assembler electric": ("8141/00", "8141", "8131"), "motor assembler engineering": ("8142/02", "8142", "8132")}
```

**Step 3:** We embed these unique ONS job titles and the input job title using the `all-MiniLM-L6-v2` Sentence Tranformers pretrained model.

**Step 4:** We then calculate the cosine similarity between the embedded input job title and all the embedded ONS job titles.

In our example, the cosine similarity scores for each input job title (row) and each ONS SOC data job title (columns) are:

|                              | data scientist | data analyst computing | data analyst | analyst developer | meter assembler | motor assembler electric | motor assembler engineering |
| ---------------------------- | -------------- | ---------------------- | ------------ | ----------------- | --------------- | ------------------------ | --------------------------- |
| Data Scientist               | **1.0**        | 0.69                   | 0.81         | 0.56              | 0.20            | 0.07                     | 0.15                        |
| Electric motor assembler     | 0.07           | 0.20                   | 0.11         | 0.15              | 0.52            | **0.81**                 | 0.80                        |
| Data visualisation developer | 0.53           | 0.57                   | 0.59         | 0.47              | 0.22            | 0.04                     | 0.17                        |

**Step 5:** Finally, we find the SOC information for the ONS job title with the highest similarity as long as it is over a certain threshold (default is 0.67). If there is no single job title with a particularly high similiarity, then we use a consensus approach at the SOC 2020 4-digit level (default to having over 3 matches with over 0.5 similarity score).

With the default values, the final matches for each inputted job title would be:

|                              | ONS job title matched to | SOC     | SOC description                                 |
| ---------------------------- | ------------------------ | ------- | ----------------------------------------------- |
| Data Scientist               | data scientist           | 2433/04 | Statistical data scientists                     |
| Electric motor assembler     | motor assembler electric | 8141/00 | Assemblers (electrical and electronic products) |
| Data visualisation developer | None                     | None    | None                                            |

However, if we had set slightly different conditions for the consensus approach, another outcome could be that the "Data visualisation developer" job title was mapped to the SOC "2134 - Programmers and software professionals" since 2 out of the 4 matches with over 0.45 similarity were from this 4-digit SOC.

## 🤔 Evaluation

To get the evaluation sample we found the most common job titles in Nesta's job advert dataset, and a random sample of job titles.

These datasets were manually labelled with how well we thought the job title was matched to a SOC code. We chose 3 categories - excellent, good or poor.

### From a **random sample** of 200 job titles:

- 59.6% had a SOC 6-digit code matched
- 5% were only able to have a SOC 4-digit code matched
- 35.5% had no SOC matched

Using 118 job titles of the random sample with SOC 6-digit codes found:

- 66% had excellent quality SOC matches
- 23% had good quality SOC matches
- 11% had poor quality SOC matches

From the 5% (10 job titles) of the random sample with SOC 4-digit codes found:

- 80% had excellent quality SOC matches
- 10% had good quality SOC matches
- 10% had poor quality SOC matches

### We also labelled 300 of the **most commonly occuring** job titles in our dataset with quality measures.

- 89% had a SOC 6-digit code matched
- 4% were only able to have a SOC 4-digit code matched
- 7% had no SOC matched

Using 255 job titles of the most commonly occuring job titles with SOC 6-digit codes found:

- 82% had excellent quality SOC matches
- 10% had good quality SOC matches
- 8% had poor quality SOC matches

From the 20 job titles of the most commonly occuring job titles with SOC 4-digit codes found:

- 95% had excellent quality SOC matches
- 5% had good quality SOC matches

We note that the results from the most commonly occuring job titles are probably better since the job title tends to be more clean and standardised.

### Examples of **excellent** matches:

| job_title                                 | num_job_ads | prop_job_ads | soc_2020_6_name                          | occ_matched              | match_prob |
| ----------------------------------------- | ----------- | ------------ | ---------------------------------------- | ------------------------ | ---------- |
| Care Assistant - Bank - Care Home         | 22444       | 0.0031       | Domiciliary care workers                 | home care assistant      | 0.78       |
| Pastry Demi Chef de Partie                | 1           | 0.0000       | Chefs                                    | chef de partie           | 0.79       |
| Forklift Driver                           | 2922        | 0.0004       | Fork-lift truck drivers                  | fork lift truck driver   | 0.88       |
| Finance ManagerRB                         | 1           | 0.0000       | Company secretaries and finance managers | finance manager          | 0.85       |
| Service Engineer Carpentry and Decorating | 1           | 0.0000       | Painters and decorators                  | decorating contractor    | 0.72       |
| Senior Software Engineer                  | 2681        | 0.0004       | Software developers                      | senior software engineer | 1.00       |
| Change Business Analyst - FMCG experience | 1           | 0.0000       | Business analysts                        | business change analyst  | 0.69       |
| Private Client Solicitor                  | 5338        | 0.0007       | Solicitors and lawyers n.e.c.            | solicitor                | 0.75       |
| Internal Sales Executive                  | 2281        | 0.0003       | Business sales executives                | sales executive          | 0.85       |
| HR Advisor                                | 10386       | 0.0014       | Human resources advisors                 | human resources adviser  | 0.85       |

### Examples of **good** matches:

| job_title                                | num_job_ads | prop_job_ads | soc_2020_6_name                                       | occ_matched                     | match_prob |
| ---------------------------------------- | ----------- | ------------ | ----------------------------------------------------- | ------------------------------- | ---------- |
| Domestic Assistant                       | 2934        | 0.0004       | Commercial cleaners                                   | domestic assistant              | 1          |
| Holiday Club Admin Manager               | 1           | 0.0000       | Hotel and accommodation managers and proprietors      | holiday centre manager          | 0.79       |
| Training and Support Manager             | 1           | 0.0000       | Education managers                                    | learning support manager        | 0.80       |
| Digital Marketing Executive              | 4554        | 0.0006       | Marketing consultants                                 | digital marketing executive     | 1          |
| Operations Manager -Commercial Insurance | 1           | 0.0000       | Financial managers and directors n.e.c.               | insurance company manager       | 0.79       |
| Field Service Engineer                   | 7272        | 0.0010       | Telecoms and related network installers and repairers | home service field engineer     | 0.87       |
| Tutor                                    | 3370        | 0.0005       | Higher education teaching professionals n.e.c.        | course tutor                    | 0.92       |
| Assistant Manager - Truro                | 2           | 0.0000       | Other administrative occupations n.e.c.               | manager's assistant             | 0.78       |
| Marketing Executive                      | 8363        | 0.0012       | Marketing consultants                                 | marketing executive             | 1          |
| Chartered Financial Advisor - Berkshire  | 2           | 0.0000       | Financial accountants                                 | chartered management accountant | 0.70       |

### Examples of **bad** matches:

| job_title                                                         | num_job_ads | prop_job_ads | soc_2020_6_name                                         | occ_matched                       | match_prob |
| ----------------------------------------------------------------- | ----------- | ------------ | ------------------------------------------------------- | --------------------------------- | ---------- |
| Academic Mentor                                                   | 2847        | 0.000        | Learning and behaviour mentors                          | learning mentor                   | 0.85       |
| Electronics Assembly Technician - Oxford - &#163;30,000 per annum | 2           | 0.000        | Metal working production and maintenance fitters n.e.c. | assembly engineer                 | 0.67       |
| Senior Administrator                                              | 2315        | 0.000        | Registrars                                              | senior registration administrator | 0.84       |
| Census officer                                                    | 3547        | 0.000        | Office managers                                         | census district manager           | 0.80       |
| Operative                                                         | 2201        | 0.000        | Textile process operatives n.e.c.                       | general operative                 | 0.77       |
| Business Case Manager Business                                    | 1           | 0.000        | National government administrative occupations n.e.c.   | case manager                      | 0.76       |
| Production Operative                                              | 16113       | 0.002        | Printing machine assistants                             | finishing operative               | 0.76       |
| Night Care Assistant                                              | 11302       | 0.002        | Shelf fillers                                           | night assistant                   | 0.86       |
| Supply Teacher                                                    | 7316        | 0.001        | Inventory and stock controllers                         | supplies superintendent           | 0.72       |
| Carpenter - Timber Frame                                          | 1           | 0.000        | Agricultural and fishing trades n.e.c.                  | timber contractor                 | 0.72       |

### Observations

A random sample of the job titles that don't match to a SOC revealed that extra cleaning may help them match to SOC codes. Some of the job titles that didn't match include:

['Disability Assessor Homebase / Front Office', 'Clinical Fellow ST3 Stroke medicine', 'IT Engineer - &#163;25-&#163;30k - Normanton / Hybrid', 'Entry Level Graduate Scheme', 'Operatives Needed', 'Waiting Staff 1', 'Staff Nurse, General Surgery - Band 5', 'PHP Developer Laravel Vue.js', 'Bike Courier parttime Liverpool', 'E&amp;I Technician Days', '1-1 Tutor Required - Wakefield', 'Flexcube Analyst permanent', 'Infection Prevention Control Nurse - Band 8a', 'Blinds and Curtains Installer', 'Senior Community Host - Woking', 'Data Architect, Microsoft Stack, Remote', 'Factory Cleaning Operative - &#163;1000 sign on bonus!', 'Retail Customer Service CSM 30hrs - Multi-site', 'Retail Customer Service CSA 30hrs', 'Driver weekend Liverpool']

## Future work

It'd be good to compare our SOCmapper performance to other mappers out there, for example [this python package that maps to 3-digit SOC](https://github.com/aeturrell/occupationcoder) or [the online tool from Cascot](https://cascotweb.warwick.ac.uk/#/classification/soc2020).
