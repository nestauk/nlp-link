from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from wasabi import msg, Printer

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

msg_print = Printer()


def chunk_list(orig_list, n_chunks):
    for i in range(0, len(orig_list), n_chunks):
        yield orig_list[i : i + n_chunks]


def load_bert(bert_model_name="sentence-transformers/all-MiniLM-L6-v2"):

    with msg_print.loading("Loading BERT model"):
        device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        bert_model = SentenceTransformer(bert_model_name, device=device)
        bert_model.max_seq_length = 512
    msg.good("BERT model loaded")
    return bert_model


def get_embeddings(
    text_list: list,
    bert_model,
    embed_chunk_size: int = 500,
    batch_size: int = 32,
) -> np.array:
    """
    Get embeddings for a list of texts

    Args:
        text_list (list): A lists of texts.
        bert_model: An initialised SentenceTransformer BERT model.
        embed_chunk_size (int): The number of texts per chunk to process.
        batch_size (int): BERT batch_size.
    Returns:
        np.array: The embeddings for the input list of texts
    """

    msg.info(
        f"Finding embeddings for {len(text_list)} texts chunked into {round(len(text_list)/embed_chunk_size)} chunks"
    )
    all_embeddings = []
    for batch_texts in tqdm(chunk_list(text_list, embed_chunk_size)):
        all_embeddings.append(
            bert_model.encode(np.array(batch_texts), batch_size=batch_size)
        )
    all_embeddings = np.concatenate(all_embeddings)
    msg.good("Texts embedded.")

    return all_embeddings
