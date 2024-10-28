# Needed for Github Actions to not fail (see torch bug https://github.com/pytorch/pytorch/issues/121101)
import torch

torch.set_num_threads(1)

from nlp_link.soc_mapper.soc_map import SOCMapper


def test_find_most_likely_soc():

    # Made up top match data
    match_row = {
        "job_title": "data scientist",
        "top_soc_matches": [
            ["Data scientist", "2433/04", "2433", "2425", 0.95],
            ["Data scientist computing", "2133/99", "2133", "2135", 0.9],
            ["Data engineer", "2133/03", "2133", "2135", 0.8],
            ["Data analyst", "3544/00", "3544", "3539", 0.8],
            ["Computer scientist", "2133/01", "2133", "2135", 0.7],
            ["Finance adviser", "2422/02", "2422", "3534", 0.5],
        ],
    }

    # When there is a top match over the sim_threshold
    soc_mapper = SOCMapper(
        sim_threshold=0.91,
    )

    result = soc_mapper.find_most_likely_soc(match_row)
    assert result[1] == "Data scientist"

    # When there is no top match over the sim_threshold but at least minimum_n and over minimum_prop of
    # the matches over top_n_sim_threshold similarity are the same
    soc_mapper = SOCMapper(
        sim_threshold=0.98, top_n_sim_threshold=0.65, minimum_n=3, minimum_prop=0.5
    )

    result = soc_mapper.find_most_likely_soc(match_row)
    assert result[0][1] == "2133"

    # When there is no top match over the sim_threshold and there are some
    # matches over top_n_sim_threshold similarity but not enough (<minimum_n)
    soc_mapper = SOCMapper(
        sim_threshold=0.98,
        top_n_sim_threshold=0.65,
        minimum_n=6,
    )

    result = soc_mapper.find_most_likely_soc(match_row)

    assert result == None

    # When there is no top match over the sim_threshold and there are >=minimum_n
    # matches over top_n_sim_threshold similarity but not enough of them have the same 4 digit soc
    soc_mapper = SOCMapper(
        sim_threshold=0.98, top_n_sim_threshold=0.75, minimum_n=3, minimum_prop=0.75
    )

    result = soc_mapper.find_most_likely_soc(match_row)

    assert result == None

    # When none of the matches are good enough, even for a consensus approach
    soc_mapper = SOCMapper(
        sim_threshold=0.98,
        top_n_sim_threshold=0.98,
    )

    result = soc_mapper.find_most_likely_soc(match_row)

    assert result == None
