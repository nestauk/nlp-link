"""
Class to link two datasets.

Example usage:

from nlp_link.linker import NLPLinker

nlp_link = NLPLinker()

# dict inputs
reference_data = {'a': 'cats', 'b': 'dogs', 'd': 'rats', 'e': 'birds'}
input_data = {'x': 'owls', 'y': 'feline', 'z': 'doggies', 'za': 'dogs', 'zb': 'chair'}
nlp_link.load(reference_data)
matches = nlp_link.link_dataset(input_data)
# Top match output
print(matches)

# list inputs
reference_data = ['cats', 'dogs', 'rats', 'birds']
input_data = ['owls', 'feline', 'doggies', 'dogs','chair']
nlp_link.load(reference_data)
matches = nlp_link.link_dataset(input_data)
# Top match output
print(matches)

"""

from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

from typing import Union, Optional

from nlp_link.linker_utils import chunk_list, get_embeddings, load_bert

from wasabi import msg, Printer

msg_print = Printer()

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
                msg.warn(
                    "Input is a dataframe, please specify id_column and text_column"
                )
        else:
            msg.warn(
                "The input_data input must be a dictionary, a list or pandas dataframe"
            )

        if not isinstance(input_data[0], str):
            msg.warn(
                "The input_data input must be a list of texts, or a dictionary where the values are texts"
            )

    def load(
        self,
        reference_data: Union[list, dict],
    ):
        """
        Load the embedding model and embed the reference dataset
        Args:
            reference_data (Union[list, dict]): The reference texts to find links to.
                A list of texts or a dictionary of texts where the key is the unique id.
                If a list is given then a unique id will be assigned with the index order.
        """
        self.bert_model = load_bert()

        self.reference_data = self._process_dataset(reference_data)
        self.reference_data_texts = list(self.reference_data.values())
        self.reference_data_ids = list(self.reference_data.keys())

        self.reference_embeddings = self._get_embeddings(self.reference_data_texts)

    def _get_embeddings(self, text_list: list) -> np.array:
        """
        Get embeddings for a list of texts

        Args:
            text_list (list): A lists of texts
        Returns:
            np.array: The embeddings for the input list of texts
        """

        return get_embeddings(
            text_list=text_list,
            embed_chunk_size=self.embed_chunk_size,
            batch_size=self.batch_size,
            bert_model=self.bert_model,
        )

    def get_matches(
        self,
        input_data_ids: list,
        input_embeddings: np.array,
        reference_data_ids: list,
        reference_embeddings: np.array,
        top_n: int,
        drop_most_similar: bool = False,
    ) -> dict:
        """
        Find top matches across two datasets using their embeddings.

        Args:
            input_data_ids (list): The ids of the input texts.
            input_embeddings (np.array): Embeddings for the input texts.
            reference_data_ids (list): The ids of the reference texts.
            reference_embeddings (np.array): Embeddings for the reference texts.
            top_n (int): The number of top links to return in the output.
            drop_most_similar (bool, default = False): Whether to not output the most similar match, this would be set to True if you are matching a list with itself.

        Returns:
            dict: The top matches for each input id.
        """

        msg.info(
            f"Finding the top dataset matches for {len(input_data_ids)} input texts chunked into {round(len(input_data_ids)/self.match_chunk_size)}"
        )

        if drop_most_similar:
            top_n = top_n + 1
            start_n = 1
        else:
            start_n = 0

        # We chunk up reference list otherwise it can crash
        matches_topn = {}
        for batch_indices in tqdm(
            chunk_list(range(len(input_data_ids)), n_chunks=self.match_chunk_size)
        ):
            batch_input_ids = [input_data_ids[i] for i in batch_indices]
            batch_input_embeddings = [input_embeddings[i] for i in batch_indices]

            batch_similarities = cosine_similarity(
                batch_input_embeddings, reference_embeddings
            )

            # Top links for each input text
            for input_ix, similarities in enumerate(batch_similarities):
                top_links = []
                for reference_ix in np.flip(np.argsort(similarities))[start_n:top_n]:
                    # reference data id + cosine similarity score
                    top_links.append(
                        [
                            reference_data_ids[reference_ix],
                            similarities[reference_ix],
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
        Link a dataset to the reference dataset.

        Args:
            input_data (Union[list, dict]): The main dictionary to be linked to texts in the loaded reference_data.
                A list of texts or a dictionary of texts where the key is the unique id.
                If a list is given then a unique id will be assigned with the index order.
            top_n (int, default = 3): The number of top links to return in the output.
            format_output (bool, default = True): If you'd like the output to be formatted to include the texts of
                the matched datasets or not (will just give the indices).
            drop_most_similar (bool, default = False): Whether to not output the most similar match, this would be set to True if you are matching a list with itself.
        Returns:
            dict: The keys are the ids of the input_data and the values are a list of lists of the top_n most similar
                ids from the reference_data and a probability score.
                e.g. {'x': [['a', 0.75], ['c', 0.7]], 'y': [...]}
        """

        try:
            msg.info(
                f"Comparing {len(input_data)} input texts to {len(self.reference_embeddings)} reference texts"
            )
        except:
            msg.warning(
                "self.reference_embeddings does not exist - you may have not run load()"
            )

        input_data = self._process_dataset(input_data)
        input_data_texts = list(input_data.values())
        input_data_ids = list(input_data.keys())

        input_embeddings = self._get_embeddings(input_data_texts)

        self.matches_topn = self.get_matches(
            input_data_ids,
            input_embeddings,
            self.reference_data_ids,
            self.reference_embeddings,
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
                        "reference_id": link_data[0][0],
                        "reference_text": self.reference_data[link_data[0][0]],
                        "similarity": link_data[0][1],
                    }
                    for input_id, link_data in self.matches_topn.items()
                ]
            )
            return df_output
        else:
            return self.matches_topn
