import nltk
import tiktoken
from typing import List, Dict

# download punkt tokenizer (only once)
nltk.download("punkt", quiet=True)

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[Dict[str, str]]:
    """
    Split text into overlapping chunks based on token count.

    Args:
        text: The full text to chunk.
        chunk_size: Maximum number of tokens per chunk.
        overlap: Number of tokens to overlap between chunks.
        by: Split strategy — "sentence" or "paragraph".

    Returns:
        List of dicts with chunk metadata: {"id", "text", "tokens"}
    """
    encoding = tiktoken.get_encoding("cl100k_base")  # good default up to gpt-4
    # encoding = tiktoken.get_encoding("o200k_base")  # good default > gpt-4
    # encoding = tiktoken.encoding_for_model("gpt-4o") # specific model encoding, if needed


    units = nltk.sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    def count_tokens(s: str) -> int:
        return len(encoding.encode(s))

    for unit in units:
        unit_tokens = count_tokens(unit)
        # if adding this unit exceeds limit, flush chunk
        if current_tokens + unit_tokens > chunk_size:
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    chunk_text
                )
                # create overlap
                overlap_text = encoding.decode(
                    encoding.encode(chunk_text)[-overlap:]
                )
                current_chunk = [overlap_text]
                current_tokens = count_tokens(overlap_text)
        current_chunk.append(unit)
        current_tokens += unit_tokens

    # flush last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(chunk_text)

    return chunks
