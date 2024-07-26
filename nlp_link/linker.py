import pandas as pd
import random


def link_lists(list_1, list_2):
    """
    Mock linker
    """
    list_1_index = list(range(len(list_1)))
    list_2_index = list(range(len(list_2)))

    return [(i, random.choice(list_1_index)) for i in list_2_index]
