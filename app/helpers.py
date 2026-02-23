def deduplicate(chunks):
    seen = set()
    unique = []

    for chunk in chunks:
        if chunk.chunk_id not in seen:
            unique.append(chunk)
            seen.add(chunk.chunk_id)

    return unique