from nlp_link.linker import link_lists


def test_link_lists():

    list_1 = ["dog", "cat"]
    list_2 = ["kitten", "puppy"]
    linked = link_lists(list_1, list_2)

    assert len(linked) == len(list_1)
