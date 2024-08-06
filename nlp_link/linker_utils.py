def chunk_list(orig_list, n_chunks):
    for i in range(0, len(orig_list), n_chunks):
        yield orig_list[i : i + n_chunks]
