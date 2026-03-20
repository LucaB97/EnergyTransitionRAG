import argparse
import pandas as pd
import pdfplumber
import json
from pathlib import Path

from utils.pdf_extraction import extract_text_two_columns
from utils.text_cleaning import clean_text, remove_headers_footers, remove_references
from utils.chunking import create_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"


parser = argparse.ArgumentParser(description="Build text chunks for RAG ingestion")
parser.add_argument("--chunk-size", type=int)
parser.add_argument("--overlap", type=int)
args = parser.parse_args()

chunk_size = args.chunk_size
overlap = args.overlap

if chunk_size is None or overlap is None:
    chunk_size = 500
    overlap = 100

if overlap >= chunk_size:
    raise ValueError("Overlap must be smaller than chunk size.")

metadata = pd.read_csv(METADATA_PATH)
chunks = []

print("STARTING Text extraction & Chunking")
for i in range(len(metadata)):
    print(f"Paper: {i+1}/{len(metadata)}")
    pdf_path = PAPERS_DIR / f"{metadata.paper_id.iloc[i]}.pdf"
    pages_lines = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = extract_text_two_columns(page)
            pages_lines.append(page_text.split("\n"))

    pages_lines = remove_headers_footers(pages_lines)

    text = "\n\n".join("\n".join(lines) for lines in pages_lines)
    text = remove_references(text)
    text = clean_text(text)

    chunks.extend(create_chunks(text, metadata, i, chunk_size, overlap))


output_filename = f"chunks_{chunk_size}t_{overlap}o.json"
output_path = PROJECT_ROOT / "data" / output_filename

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)
