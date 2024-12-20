"""
A class to map inputted job titles to their most likely SOC 2020 4-digit codes.

Usage:

from soc_mapper.soc_map import SOCMapper

soc_mapper = SOCMapper()
soc_mapper.load()
job_titles=["data scientist", "Assistant nurse", "Senior financial consultant - London"]

soc_mapper.get_soc(job_titles, return_soc_name=True)

"""

from collections import Counter
import os
from typing import List, Union

import pandas as pd
from tqdm import tqdm
import numpy as np

from nlp_link import soc_mapper_config

from nlp_link.soc_mapper.soc_map_utils import (
    load_job_title_soc,
    process_job_title_soc,
    job_title_cleaner,
    unique_soc_job_titles,
)
from nlp_link.linker import NLPLinker

from nlp_link.linker_utils import load_bert

from nlp_link.utils.utils import (
    load_s3_json,
    load_local_json,
    save_to_s3,
    save_json_dict,
)

from wasabi import msg, Printer

msg_print = Printer()


class SOCMapper(object):
    """Class for linking job titles to SOC codes.

    The input job title is matched to a dataset of job titles with their 2020 SOC.
    - If the most similar job title is very similar, then the corresponding 6-digit SOC is outputted.
    - Otherwise, we look at a group of the most similar job titles, and if they all have the same 4-digit SOC, then this is outputted.

    ----------

    Args:
        soc_dir (str): The directory of the SOC coding index xlsx file.
        match_top_n (int): The number of most similar SOC matches to consider when calculating the final SOC and outputing
        sim_threshold (float): The similarity threshold for outputting the most similar SOC match.
        top_n_sim_threshold (float): The similarity threshold for a match being added to a group of SOC matches.
        minimum_n (int): The minimum size of a group of SOC matches.
        minimum_prop (float): If a group of SOC matches have a high proportion (>= minimum_prop) of the same SOC being matched, then use this SOC.

    ----------
    Methods
    ----------

    load_process_soc_data():
        Load the SOC data
    load(reset_embeddings=False, save_embeds=False):
            Load everything to use this class, recalculate SOC embeddings and save if desired
    find_most_similar_matches(job_titles, job_title_embeddings):
            Using the inputted job title embeddings and the SOC embeddings, find the full information about the most similar SOC job titles
    find_most_likely_soc(match_row):
            For the full match information for one job title, find the most likely SOC (via top match, or group of top matches)
    get_soc(job_titles, additional_info=False):
            (main function) For inputted job titles, output the best SOC match, add extra information about matches using the additional_info argument

    ----------
    Usage
    ----------
        from soc_mapper.soc_map import SOCMapper
        soc_mapper = SOCMapper()
        soc_mapper.load()
        matches = soc_mapper.get_soc(job_titles=["data scientist", "Assistant nurse", "Senior financial consultant - London"])
        >>> [(('2433/02', '2433', '2425'), 'data scientist'), (('6131/99', '6131', '6141'), 'assistant nurse'), (('2422/02', '2422', '3534'), 'financial consultant')]
    """

    def __init__(
        self,
        soc_dir: str = soc_mapper_config["soc_data"]["soc_dir"],
        match_top_n: int = soc_mapper_config["soc_mapper"]["match_top_n"],
        sim_threshold: float = soc_mapper_config["soc_mapper"]["sim_threshold"],
        top_n_sim_threshold: float = soc_mapper_config["soc_mapper"][
            "top_n_sim_threshold"
        ],
        minimum_n: int = soc_mapper_config["soc_mapper"]["minimum_n"],
        minimum_prop: float = soc_mapper_config["soc_mapper"]["minimum_prop"],
    ):
        self.soc_dir = soc_dir
        self.match_top_n = match_top_n
        self.sim_threshold = sim_threshold
        self.top_n_sim_threshold = top_n_sim_threshold
        self.minimum_n = minimum_n
        self.minimum_prop = minimum_prop

        self.soc_mapper_config = soc_mapper_config  # This is so a user could change the soc_data values easily if needed

    def load_process_soc_data(self):
        """
        Load the job titles to SOC codes dataset as found on the ONS website.
        A small amount of processing.
        """

        jobtitle_soc_data = process_job_title_soc(
            load_job_title_soc(soc_mapper_config=self.soc_mapper_config),
            soc_mapper_config=self.soc_mapper_config,
        )

        return jobtitle_soc_data

    def load(
        self,
        reset_embeddings: bool = soc_mapper_config["soc_mapper"]["reset_embeddings"],
        save_embeds: bool = False,
    ):
        """
        Load the BERT model, SOC coding index data, and load or calculate embeddings for the job titles in this dataset.
        Args:
            reset_embeddings (bool): Whether to re-calculate and save soc coding index embeddings or not. Will be done anyway if
                an embeddings file isn't found.
            save_embeds (bool): Whether to save the out the embeddings or not (only used if the embeddings weren't loaded)
        """

        self.nlp_link = NLPLinker()
        self.nlp_link.bert_model = load_bert()

        self.jobtitle_soc_data = self.load_process_soc_data()

        self.soc_2020_6_dict = dict(
            zip(
                self.jobtitle_soc_data["SOC_2020_EXT"],
                self.jobtitle_soc_data["SUB-UNIT GROUP DESCRIPTIONS"],
            )
        )
        self.soc_2020_4_dict = dict(
            zip(
                self.jobtitle_soc_data["SOC_2020"],
                self.jobtitle_soc_data["SOC 2020 UNIT GROUP DESCRIPTIONS"],
            )
        )
        self.job_title_2_soc6_4 = unique_soc_job_titles(self.jobtitle_soc_data)

        embeddings_output_dir = os.path.dirname(self.soc_dir)

        if "s3://" in embeddings_output_dir:
            s3_bucket_name = embeddings_output_dir.split("s3://")[1].split("/")[0]
            embeddings_output_s3_folder = "/".join(
                embeddings_output_dir.split("s3://")[1].split("/")[1:]
            )
            embeddings_path = os.path.join(
                embeddings_output_s3_folder, "soc_job_embeddings.json"
            )
            job_titles_path = os.path.join(
                embeddings_output_s3_folder, "soc_job_embeddings_titles.json"
            )
        else:
            s3_bucket_name = None
            embeddings_path = os.path.join(
                embeddings_output_dir, "soc_job_embeddings.json"
            )
            job_titles_path = os.path.join(
                embeddings_output_dir, "soc_job_embeddings_titles.json"
            )

        try:
            if not reset_embeddings:
                try:
                    if s3_bucket_name:
                        with msg_print.loading(
                            f"Loading SOC job title embeddings from S3 ..."
                        ):
                            self.all_soc_embeddings = load_s3_json(
                                s3_bucket_name, embeddings_path
                            )
                            self.soc_job_titles = load_s3_json(
                                s3_bucket_name, job_titles_path
                            )
                    else:
                        with msg_print.loading(
                            f"Loading SOC job title embeddings locally ..."
                        ):
                            self.all_soc_embeddings = load_local_json(embeddings_path)
                            self.soc_job_titles = load_local_json(job_titles_path)
                    msg.good("SOC job title embeddings loaded.")
                except:
                    msg.warn(f"SOC job title embeddings not found.")
                    raise
            else:
                raise
        except:
            msg.info(f"Calculating SOC job title embeddings ...")

            # Embed the SOC job titles
            self.soc_job_titles = list(self.job_title_2_soc6_4.keys())

            self.all_soc_embeddings = self.nlp_link._get_embeddings(self.soc_job_titles)

            if save_embeds:
                msg.info(f"Saving SOC job title embeddings")
                if s3_bucket_name:
                    save_to_s3(s3_bucket_name, self.all_soc_embeddings, embeddings_path)
                    save_to_s3(s3_bucket_name, self.soc_job_titles, job_titles_path)
                    msg.good(f"Saved to s3://{s3_bucket_name} + {embeddings_path} ...")
                else:
                    save_json_dict(self.all_soc_embeddings, embeddings_path)
                    save_json_dict(self.soc_job_titles, job_titles_path)
                    msg.good(f"Saved to {embeddings_path} ...")
            else:
                msg.warn(
                    f"Newly calculated SOC job title embeddings were not saved, set save_embeds=True if you'd like to save them to speed up future use."
                )

    def find_most_similar_matches(
        self,
        job_titles: Union[str, List[str]],
        job_title_embeddings: np.array(object),
    ) -> list:
        """
        Using the job title embeddings and the SOC job title embeddings,
        find the top n SOC job titles which are most similar to each input job title.

        Args:
            job_titles (str or list of strings): One or a list of inputted job titles.
            job_title_embeddings (np.array()): The embeddings for the inputted job titles.

        Outputs:
            list: A list of the most similar SOC data for each inputted job title.
        """

        matches_topn_dict = self.nlp_link.get_matches(
            input_data_ids=list(range(len(job_titles))),
            input_embeddings=job_title_embeddings,
            reference_data_ids=list(range(len(self.all_soc_embeddings))),
            reference_embeddings=self.all_soc_embeddings,
            top_n=self.match_top_n,
        )

        job_top_soc_matches = []
        for k, v in matches_topn_dict.items():
            top_soc_matches = []
            for top_match in v:
                soc_ix = top_match[0]
                similarity = top_match[1]
                soc_text = self.soc_job_titles[soc_ix]
                top_soc_matches.append(
                    [
                        soc_text,
                        self.job_title_2_soc6_4[soc_text][0],  # 6 digit
                        self.job_title_2_soc6_4[soc_text][1],  # 4 digit
                        self.job_title_2_soc6_4[soc_text][2],  # 2010 4 digit
                        similarity,
                    ]
                )
            job_top_soc_matches.append(
                {
                    "job_title": job_titles[k],
                    "top_soc_matches": top_soc_matches,
                }
            )

        return job_top_soc_matches

    def find_most_likely_soc(
        self,
        match_row: dict,
    ) -> tuple:
        """
        For a single job title and the details of the most similar SOC matches, find a single most likely SOC
        1. If the top match has a really high similarity score (>sim_threshold) at the 6-digit level then use this.
                This will return (soc, job_title)
        2. Get the 4-digit SOCs of the good (>top_n_sim_threshold) matches in the top n most similar.
        3. If there are a few of these (>=minimum_n) and over a certain proportion (>minimum_prop) of these are the same at the 4 digit level - use this as the SOC.
                This will return (soc, the job titles given for this same soc)

        Returns data in the format ((soc_2020_6, soc_2020_4, soc_2010), job_title) or None
        If pathway 1. (above) isn't true then the output will be ((None, soc_2020_4, None), job_title) and job_title will be a set of multiple

        Args:
            match_row: One element from the list outputted in find_most_similar_matches.
            e.g. {"job_title": 'principal data scientist', "top_soc_matches": [["data scientist", 6digit SOC, 4 digit SOC, 4 digit 2010 SOC, similarity_score], ...]}

        Output:
            tuple, None: Details of the most likely SOC match for this job title.
        """

        top_soc_match = match_row["top_soc_matches"][0][0]
        top_soc_match_code = (
            match_row["top_soc_matches"][0][1],
            match_row["top_soc_matches"][0][2],
            match_row["top_soc_matches"][0][3],
        )  # 6 digit, 4 digit, 4 2010
        top_soc_match_score = match_row["top_soc_matches"][0][4]  # The score

        if top_soc_match_score > self.sim_threshold:
            return (top_soc_match_code, top_soc_match)
        else:
            all_good_socs = [
                t[2]  # 4 digit 2020 SOC
                for t in match_row["top_soc_matches"]
                if t[4] > self.top_n_sim_threshold
            ]
            if len(all_good_socs) >= self.minimum_n:
                common_soc, num_common_soc = Counter(all_good_socs).most_common(1)[0]
                prop_most_common_soc = num_common_soc / len(all_good_socs)
                if prop_most_common_soc > self.minimum_prop:
                    return (
                        (None, common_soc, None),
                        set(
                            [
                                t[0]
                                for t in match_row["top_soc_matches"]
                                if (
                                    (t[4] > self.top_n_sim_threshold)
                                    and (t[2] == common_soc)
                                )
                            ]
                        ),
                    )
                else:
                    return None
            else:
                return None

    def get_soc(
        self,
        job_titles: Union[str, List[str]],
        additional_info: bool = False,
        return_soc_name: bool = False,
        clean_job_title: bool = True,
    ) -> list:
        """
        Get the most likely SOC for each inputted job title

        Args:
            job_titles (str, list): A single job title or a list of raw job titles
            additional_info (bool): Whether to provide additional information about the matches.
                        Return just the most likely soc match (False) or the top soc matches (True)
            return_soc_name (bool): Whether to output the SOC names of the most likely SOC (or just the codes).
                        When applied to lots of data this might not be as desirable.
            clean_job_title (bool): Whether to apply the cleaning function to the job title.

        Output:
            list: A list of the top matches for each job title inputted

        """

        if isinstance(job_titles, str):
            job_titles = [job_titles]

        # Clean the job titles
        if clean_job_title:
            job_titles = [job_title_cleaner(job_title) for job_title in job_titles]

        # Embed the input job titles
        job_title_embeddings = self.nlp_link._get_embeddings(job_titles)

        top_soc_matches = self.find_most_similar_matches(
            job_titles, job_title_embeddings
        )

        msg.info(f"Finding most likely SOC")
        found_count = 0
        for job_matches in top_soc_matches:
            most_likely_soc = self.find_most_likely_soc(job_matches)
            if most_likely_soc:
                ((soc_2020_6, soc_2020_4, soc_2010_4), job_title) = most_likely_soc
                if return_soc_name:
                    job_matches["most_likely_soc"] = (
                        (
                            (soc_2020_6, self.soc_2020_6_dict.get(soc_2020_6)),
                            (soc_2020_4, self.soc_2020_4_dict.get(soc_2020_4)),
                            soc_2010_4,
                        ),
                        job_title,
                    )
                else:
                    job_matches["most_likely_soc"] = (
                        (soc_2020_6, soc_2020_4, soc_2010_4),
                        job_title,
                    )
            else:
                job_matches["most_likely_soc"] = None
            if most_likely_soc:
                found_count += 1

        msg.good(
            f"Found SOCs for {found_count*100/len(top_soc_matches)}% of the job titles"
        )

        if additional_info:
            return top_soc_matches
        else:
            return [
                job_matches.get("most_likely_soc") for job_matches in top_soc_matches
            ]
