# Needed for Github Actions to not fail (see torch bug https://github.com/pytorch/pytorch/issues/121101)
import torch

torch.set_num_threads(1)

from nlp_link.linker import NLPLinker

import numpy as np


def test_NLPLinker_dict_input():

    nlp_link = NLPLinker()

    reference_data = {"a": "cats", "b": "dogs", "c": "rats", "d": "birds"}
    input_data = {
        "x": "owls",
        "y": "feline",
        "z": "doggies",
        "za": "dogs",
        "zb": "chair",
    }
    nlp_link.load(reference_data)
    matches = nlp_link.link_dataset(input_data)

    assert len(matches) == len(input_data)
    assert len(set(matches["reference_id"]).difference(set(reference_data.keys()))) == 0


def test_NLPLinker_list_input():

    nlp_link = NLPLinker()

    reference_data = ["cats", "dogs", "rats", "birds"]
    input_data = ["owls", "feline", "doggies", "dogs", "chair"]
    nlp_link.load(reference_data)
    matches = nlp_link.link_dataset(input_data)

    assert len(matches) == len(input_data)
    assert (
        len(set(matches["reference_id"]).difference(set(range(len(reference_data)))))
        == 0
    )


def test_get_matches():

    nlp_link = NLPLinker()

    matches_topn = nlp_link.get_matches(
        input_data_ids=["x", "y", "z"],
        input_embeddings=np.array(
            [[0.1, 0.13, 0.14], [0.12, 0.18, 0.15], [0.5, 0.9, 0.91]]
        ),
        reference_data_ids=["a", "b"],
        reference_embeddings=np.array([[0.51, 0.99, 0.9], [0.1, 0.13, 0.14]]),
        top_n=1,
    )

    assert matches_topn["x"][0][0] == "b"
    assert matches_topn["y"][0][0] == "b"
    assert matches_topn["z"][0][0] == "a"


def test_same_input():

    nlp_link = NLPLinker()

    reference_data = {"a": "cats", "b": "dogs", "c": "rats", "d": "birds"}
    input_data = reference_data
    nlp_link.load(reference_data)
    matches = nlp_link.link_dataset(input_data, drop_most_similar=False)

    assert all(matches["input_id"] == matches["reference_id"])

    matches = nlp_link.link_dataset(input_data, drop_most_similar=True)

    assert all(matches["input_id"] != matches["reference_id"])
