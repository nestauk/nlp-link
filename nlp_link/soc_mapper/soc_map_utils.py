import pandas as pd

import re
import os

from nlp_link import soc_mapper_config
from nlp_link.utils.utils import get_df_from_excel_s3_path


def load_job_title_soc(soc_mapper_config: dict = soc_mapper_config) -> pd.DataFrame():
    """
    Load the ONS dataset which gives SOC codes for thousands of job titles
    """

    soc_dir = soc_mapper_config["soc_data"]["soc_dir"]
    dir_split = soc_dir.split("s3://")[1].split("/")

    s3_bucket_name = dir_split[0]
    s3_key = os.path.join("", *dir_split[1:])

    jobtitle_soc_data = get_df_from_excel_s3_path(
        bucket_name=s3_bucket_name,
        key=s3_key,
        sheet_name=soc_mapper_config["soc_data"]["sheet_name"],
        converters={
            soc_mapper_config["soc_data"]["soc_2020_ext_col"]: str,
            soc_mapper_config["soc_data"]["soc_2020_col"]: str,
            soc_mapper_config["soc_data"]["soc_2010_col"]: str,
        },
    )

    return jobtitle_soc_data


def process_job_title_soc(
    jobtitle_soc_data: pd.DataFrame(), soc_mapper_config: dict = soc_mapper_config
) -> pd.DataFrame():
    """Standardise the column names for use in soc_map.py
    Args:
        jobtitle_soc_data (pd.DataFrame): the raw ONS SOC coding index dataset
    Returns:
        pd.DataFrame: the cleaned ONS SOC coding index dataset
    """

    jobtitle_soc_data = jobtitle_soc_data.rename(
        columns={
            soc_mapper_config["soc_data"]["soc_2020_ext_col"]: "SOC_2020_EXT",
            soc_mapper_config["soc_data"]["soc_2020_col"]: "SOC_2020",
            soc_mapper_config["soc_data"]["soc_2010_col"]: "SOC_2010",
            soc_mapper_config["soc_data"][
                "natural_order_col"
            ]: "INDEXOCC NATURAL WORD ORDER",
            soc_mapper_config["soc_data"]["sug_col"]: "SUB-UNIT GROUP DESCRIPTIONS",
            soc_mapper_config["soc_data"]["ug_col"]: "SOC 2020 UNIT GROUP DESCRIPTIONS",
            soc_mapper_config["soc_data"]["add_col"]: "ADD",
            soc_mapper_config["soc_data"]["ind_col"]: "IND",
        }
    )

    # Clean
    jobtitle_soc_data = jobtitle_soc_data[jobtitle_soc_data["SOC_2020"] != "}}}}"]

    return jobtitle_soc_data


def unique_soc_job_titles(jobtitle_soc_data: pd.DataFrame()) -> dict:
    """
    Taking the dataset of job titles and which SOC they belong to - create a unique
    dictionary where each key is a job title and the value is the SOC code.
    There are additional words to include in the job title if at first
    it is not unique.

    Args:
        jobtitle_soc_data (pd.DataFrame): the cleaned ONS SOC coding index dataset.

    Returns:
        dict: A dictionary where each key is a job title and the value is the SOC code.

    """

    col_name_0 = "INDEXOCC NATURAL WORD ORDER"
    col_name_1 = "ADD"
    col_name_2 = "IND"

    jobtitle_soc_data[f"{col_name_0} and {col_name_1}"] = jobtitle_soc_data.apply(
        lambda x: (
            x[col_name_0] + " " + x[col_name_1]
            if pd.notnull(x[col_name_1])
            else x[col_name_0]
        ),
        axis=1,
    )
    jobtitle_soc_data[
        f"{col_name_0} and {col_name_1} and {col_name_2}"
    ] = jobtitle_soc_data.apply(
        lambda x: (
            x[f"{col_name_0} and {col_name_1}"] + " " + x[col_name_2]
            if pd.notnull(x[col_name_2])
            else x[f"{col_name_0} and {col_name_1}"]
        ),
        axis=1,
    )

    # Try to find a unique job title to SOC 2020 4 or 6 code mapping
    job_title_2_soc6_4 = {}
    for job_title, grouped_soc_data in jobtitle_soc_data.groupby(col_name_0):
        if grouped_soc_data["SOC_2020_EXT"].nunique() == 1:
            job_title_2_soc6_4[job_title] = (
                grouped_soc_data["SOC_2020_EXT"].unique()[0],
                grouped_soc_data["SOC_2020"].unique()[0],
                grouped_soc_data["SOC_2010"].unique()[0],
            )
        else:
            for job_title_1, grouped_soc_data_1 in grouped_soc_data.groupby(
                f"{col_name_0} and {col_name_1}"
            ):
                if grouped_soc_data_1["SOC_2020_EXT"].nunique() == 1:
                    job_title_2_soc6_4[job_title_1] = (
                        grouped_soc_data_1["SOC_2020_EXT"].unique()[0],
                        grouped_soc_data_1["SOC_2020"].unique()[0],
                        grouped_soc_data_1["SOC_2010"].unique()[0],
                    )
                else:
                    for (
                        job_title_2,
                        grouped_soc_data_2,
                    ) in grouped_soc_data_1.groupby(
                        f"{col_name_0} and {col_name_1} and {col_name_2}"
                    ):
                        if grouped_soc_data_2["SOC_2020_EXT"].nunique() == 1:
                            job_title_2_soc6_4[job_title_2] = (
                                grouped_soc_data_2["SOC_2020_EXT"].unique()[0],
                                grouped_soc_data_2["SOC_2020"].unique()[0],
                                grouped_soc_data_2["SOC_2010"].unique()[0],
                            )

    return job_title_2_soc6_4


def unique_soc_descriptions(soc_data: pd.DataFrame()) -> dict:
    """
    Taking the dataset of SOC and their descriptions - create a unique
    dictionary where each key is a description and the value is the SOC code.

    Args:
        soc_data (pd.DataFrame): the cleaned ONS SOC coding index dataset.

    Returns:
        dict: A dictionary where each key is a SOC description and the value is the SOC code.

    """
    soc_data["SUB-UNIT GROUP DESCRIPTIONS"] = soc_data[
        "SUB-UNIT GROUP DESCRIPTIONS"
    ].apply(lambda x: x.replace(" n.e.c.", "").replace(" n.e.c", ""))

    dd = soc_data[
        ["SUB-UNIT GROUP DESCRIPTIONS", "SOC_2020_EXT", "SOC_2020", "SOC_2010"]
    ].drop_duplicates()

    # There can be multiple 2010 codes for each 6 digit, so just output the most common
    soc_desc_2_code = {}
    for description, soc_info in dd.groupby("SUB-UNIT GROUP DESCRIPTIONS"):
        soc_2020_6 = soc_info["SOC_2020_EXT"].value_counts().index[0]
        soc_2020_4 = soc_info["SOC_2020"].value_counts().index[0]
        soc_2010 = list(soc_info["SOC_2010"].unique())
        soc_desc_2_code[description] = (soc_2020_6, soc_2020_4, soc_2010)

    return soc_desc_2_code


major_places = [
    "Central London",
    "Midlands",
    "London",
    "Birmingham",
    "Leeds",
    "Glasgow",
    "Sheffield",
    "Bradford",
    "Manchester",
    "Edinburgh",
    "Liverpool",
    "Bristol",
    "Cardiff",
    "Coventry",
    "Nottingham",
    "Leicester",
    "Sunderland",
    "Belfast",
    "Newcastle",
    "Brighton",
    "Hull",
    "Plymouth",
    "Carlisle",
    "Berkshire",
    "Doncaster",
    "Bedford",
    "Chichester",
    "Wakefield",
]
lower_case_end_words = [
    "nights",
    "part time",
    "full time",
    "hybrid",
    "maternity cover",
    "remote",
    "self employed",
    "work from home",
    "benefits",
    "flexible",
    "Office Based",
]

lower_case_all_end_words = [
    word.lower() for word in major_places + lower_case_end_words
]


def job_title_cleaner(
    text: str, lower_case_all_end_words: list = lower_case_all_end_words
) -> str:
    """
    Will apply a bunch of cleaning to a job title
    - removing certain things (locations or work type after a "-")
    - fixes some unicode &#163; -> £
    - Removes text after "£""

    Assumption: weird bad stuff comes after dashes or £ signs.
    So this won't work well for e.g "£30k Data Scientist" or "Remote - London Data Scientist"

    This isn't perfect, but should hopefully help quite a few examples

    Examples:
    'Part Home Based Block Manager - Chichester' -> 'Part Home Based Block Manager'
    'Employment Solicitor - Claimant - Leeds' -> 'Employment Solicitor - Claimant'
    'Retail Customer Service CSM 16hrs' -> 'Retail Customer Service CSM'
    'Bike Delivery Driver - London' -> 'Bike Delivery Driver'
    'Fulfillment Associate - &#163;1000 Sign on Bonus!' -> 'Fulfillment Associate'

    Args:
        text (str): the text of the job title you want to clean
        lower_case_all_end_words (list): a list of all the words to clean out
        if they are at the end of the job title.
    Returns:
        str: the cleaned job title

    """
    if text:
        text = str(text)

        findreplace = {
            "&amp;": " and ",
            "&#160;": " ",
            "&#163;": "£",
            "(part time)": " ",
        }
        for f, r in findreplace.items():
            text = text.replace(f, r)
        # Get rid of any double + spaces
        text = re.sub(r"\s{2,}", " ", text).strip()

        # Remove mentions of hours e.g. Customer Service 30hrs -> Customer Service
        text = re.sub(r"\d+\s*hrs", "", text).strip()

        # If there is a "£" remove everything after it (e.g. £30k per annum)
        # Unless it occurs very early in the text
        if "£" in text:
            matches = re.findall(r"£", text)
            index_found = text.index(matches[0])
            if index_found > 4:
                text = " ".join(text.split("£")[0:-1]).strip() if "£" in text else text

        # Remove certain things after the last dash
        if " - " in text:
            last_bit = text.split(" - ")[-1].strip().lower()
            # if any of the target words are in this, then remove everything after the dash
            # e.g "Data Scientist - remote, London" -> "Data Scientist"
            found = False
            for word in lower_case_all_end_words:
                if word in last_bit:
                    found = True
                    break
            if last_bit == "":  # This may happen if a £ was found
                found = True
            if found:
                # Remove everything after the lastedash
                text = " - ".join(text.split(" - ")[0:-1]).strip()

        if text:  # The cleaning may make it so we are left with nothing
            if text[-1] == "-":
                text = text[0:-1].strip()

    return text
