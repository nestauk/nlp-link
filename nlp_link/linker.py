"""
Class to link two datasets.

Example usage:

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

"""

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

from typing import Union, Optional
import logging

from nlp_link.utils import chunk_list

logger = logging.getLogger(__name__)

# TO DO: cosine or euclidean?


class NLPLinker(object):
    """docstring for NLPLinker"""

    def __init__(self, batch_size=32, embed_chunk_size=500, match_chunk_size=10000):
        super(NLPLinker, self).__init__()
        self.batch_size = batch_size
        self.embed_chunk_size = embed_chunk_size
        self.match_chunk_size = match_chunk_size
        ## Cleaning?

    def _process_dataset(
        self,
        input_data: Union[list, dict, pd.DataFrame],
        id_column: Optional[str] = None,
        text_column: Optional[str] = None,
    ) -> dict:
        """Check and process a dataset according to the input type
        Args:
            input_data (Union[list, dict, pd.DataFrame])
                A list of texts or a dictionary of texts where the key is the unique id.
                If a list is given then a unique id will be assigned with the index order.

        Returns:
            dict: key is the id and the value is the text
        """

        if isinstance(input_data, list):
            return {ix: text for ix, text in enumerate(input_data)}
        elif isinstance(input_data, dict):
            return input_data
        elif isinstance(input_data, pd.DataFrame):
            try:
                return dict(zip(input_data[id_column], input_data[text_column]))
            except:
                logger.warning(
                    "Input is a dataframe, please specify id_column and text_column"
                )
        else:
            logger.warning(
                "The input_data input must be a dictionary, a list or pandas dataframe"
            )

        if not isinstance(input_data[0], str):
            logger.warning(
                "The input_data input must be a list of texts, or a dictionary where the values are texts"
            )

    def load(
        self,
        comparison_data: Union[list, dict],
    ):
        """
        Load the embedding model and embed the comparison dataset
        Args:
            comparison_data (Union[list, dict]): The comparison texts to find links to.
                A list of texts or a dictionary of texts where the key is the unique id.
                If a list is given then a unique id will be assigned with the index order.
        """
        logger.info("Loading model")
        self.bert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.bert_model.max_seq_length = 512

        self.comparison_data = self._process_dataset(comparison_data)
        self.comparison_data_texts = list(self.comparison_data.values())
        self.comparison_data_ids = list(self.comparison_data.keys())

        self.comparison_embeddings = self._get_embeddings(self.comparison_data_texts)

    def _get_embeddings(self, text_list: list) -> np.array:
        """
        Get embeddings for a list of texts

        Args:
            text_list (list): A lists of texts
        Returns:
            np.array: The embeddings for the input list of texts
        """

        logger.info(
            f"Finding embeddings for {len(text_list)} texts chunked into {round(len(text_list)/self.embed_chunk_size)} chunks"
        )
        all_embeddings = []
        for batch_texts in tqdm(chunk_list(text_list, self.embed_chunk_size)):
            all_embeddings.append(
                self.bert_model.encode(
                    np.array(batch_texts), batch_size=self.batch_size
                )
            )
        all_embeddings = np.concatenate(all_embeddings)

        return all_embeddings

    def get_matches(
        self,
        input_data_ids: list,
        input_embeddings: np.array,
        comparison_data_ids: list,
        comparison_embeddings: np.array,
        top_n: int,
        drop_most_similar: bool = False,
    ) -> dict:
        """
        Find top matches across two datasets using their embeddings.

        Args:
            input_data_ids (list): The ids of the input texts.
            input_embeddings (np.array): Embeddings for the input texts.
            comparison_data_ids (list): The ids of the comparison texts.
            comparison_embeddings (np.array): Embeddings for the comparison texts.
            top_n (int): The number of top links to return in the output.
            drop_most_similar (bool, default = False): Whether to not output the most similar match, this would be set to True if you are matching a list with itself.

        Returns:
            dict: The top matches for each input id.
        """

        logger.info(
            f"Finding the top dataset matches for {len(input_data_ids)} input texts chunked into {round(len(input_data_ids)/self.match_chunk_size)}"
        )

        if drop_most_similar:
            top_n = top_n + 1
            start_n = 1
        else:
            start_n = 0

        # We chunk up comparisons otherwise it can crash
        matches_topn = {}
        for batch_indices in tqdm(
            chunk_list(range(len(input_data_ids)), n_chunks=self.match_chunk_size)
        ):
            batch_input_ids = [input_data_ids[i] for i in batch_indices]
            batch_input_embeddings = [input_embeddings[i] for i in batch_indices]

            batch_similarities = cosine_similarity(
                batch_input_embeddings, comparison_embeddings
            )

            # Top links for each input text
            for input_ix, similarities in enumerate(batch_similarities):
                top_links = []
                for comparison_ix in np.flip(np.argsort(similarities))[start_n:top_n]:
                    # comparison data id + cosine similarity score
                    top_links.append(
                        [
                            comparison_data_ids[comparison_ix],
                            similarities[comparison_ix],
                        ]
                    )
                matches_topn[batch_input_ids[input_ix]] = top_links
        return matches_topn

    def link_dataset(
        self,
        input_data: Union[list, dict],
        top_n: int = 3,
        format_output: bool = True,
        drop_most_similar: bool = False,
    ) -> dict:
        """
        Link a dataset to the comparison dataset.

        Args:
            input_data (Union[list, dict]): The main dictionary to be linked to texts in the loaded comparison_data.
                A list of texts or a dictionary of texts where the key is the unique id.
                If a list is given then a unique id will be assigned with the index order.
            top_n (int, default = 3): The number of top links to return in the output.
            format_output (bool, default = True): If you'd like the output to be formatted to include the texts of
                the matched datasets or not (will just give the indices).
            drop_most_similar (bool, default = False): Whether to not output the most similar match, this would be set to True if you are matching a list with itself.
        Returns:
            dict: The keys are the ids of the input_data and the values are a list of lists of the top_n most similar
                ids from the comparison_data and a probability score.
                e.g. {'x': [['a', 0.75], ['c', 0.7]], 'y': [...]}
        """

        try:
            logger.info(
                f"Comparing {len(input_data)} input texts to {len(self.comparison_embeddings)} comparison texts"
            )
        except:
            logger.warning(
                "self.comparison_embeddings does not exist - you may have not run load()"
            )

        input_data = self._process_dataset(input_data)
        input_data_texts = list(input_data.values())
        input_data_ids = list(input_data.keys())

        input_embeddings = self._get_embeddings(input_data_texts)

        self.matches_topn = self.get_matches(
            input_data_ids,
            input_embeddings,
            self.comparison_data_ids,
            self.comparison_embeddings,
            top_n,
            drop_most_similar,
        )

        if format_output:
            # Format the output into a user friendly pandas format with the top link only
            df_output = pd.DataFrame(
                [
                    {
                        "input_id": input_id,
                        "input_text": input_data[input_id],
                        "link_id": link_data[0][0],
                        "link_text": self.comparison_data[link_data[0][0]],
                        "similarity": link_data[0][1],
                    }
                    for input_id, link_data in self.matches_topn.items()
                ]
            )
            return df_output
        else:
            return self.matches_topn
