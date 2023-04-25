import os
import glob
import re
import math
import pandas as pd
import openai
from typing import List
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, wait_random_exponential, stop_after_attempt


API_KEY = XXX

# Set the API key

openai.api_key = API_KEY

# Merge all text files

search_directory = "/Users/e/Python/ChatGPT Programs/EL AI/Texts"
output_file = "/Users/e/Python/ChatGPT Programs/EL AI/Texts/Combinedpreprocessed.txt"

# Search for .txt files in the specified directory
txt_files = glob.glob(os.path.join(search_directory, "*.txt"))

# Merge the .txt files into the output file
with open(output_file, "w") as outfile:
    for txt_file in txt_files:
        with open(txt_file, "r") as infile:
            content = infile.read()
            outfile.write(content + "\n")

print(f"Merged {len(txt_files)} .txt files into '{output_file}'.")

# Generate embeddings

input_file = "/Users/e/Python/ChatGPT Programs/EL AI/Texts/Combinedpreprocessed.txt"
num_parts = 20
base_dir = "/Users/e/Python/ChatGPT Programs/EL AI/Embeddings"


@retry(wait=wait_random_exponential(multiplier=1, min=1, max=60), stop=stop_after_attempt(6))
def get_embedding_with_backoff(text: str) -> List[float]:
    result = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return result["data"][0]["embedding"]


def get_embeddings(sentences, part_number):
    print("Generating embeddings using ThreadPoolExecutor...")
    with ThreadPoolExecutor(max_workers=540) as executor:
        embeddings = []
        for i, sentence in enumerate(sentences):
            embedding = executor.submit(
                get_embedding_with_backoff, sentence).result()
            embeddings.append(embedding)
            print(
                f"Completed embedding {i + 1}/{len(sentences)} for part {part_number}")
    return embeddings


def load_preprocessed_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    sentences = re.findall(r"From .*? Page [\d]+:.*?\.", content, re.DOTALL)
    return [sentence.replace('\n', ' ').strip() for sentence in sentences]


def split_into_parts(sentences, num_parts):
    part_size = math.ceil(len(sentences) / num_parts)
    return [sentences[i:i + part_size] for i in range(0, len(sentences), part_size)]


def process_part(part_number, part):
    embeddings_csv_path = f"{base_dir}/Combinedpreprocessed_part{part_number}.csv"

    if os.path.exists(embeddings_csv_path):
        print(f"Loading embeddings for part {part_number} from CSV file...")
        df = pd.read_csv(embeddings_csv_path)
        embeddings = df["embedding"].apply(
            lambda x: [float(val) for val in x.strip('[]').split(',')])
        df["embedding"] = embeddings
        df = df[["combined", "embedding"]]
    else:
        print(f"Generating embeddings for part {part_number}...")
        embeddings = get_embeddings(part, part_number)
        print(f"Embeddings generated for part {part_number}")
        data = {"combined": part, "embedding": embeddings}
        df = pd.DataFrame(data)
        print(
            f"Saving embeddings for part {part_number} as {embeddings_csv_path}")
        df.to_csv(embeddings_csv_path, index=False)
        print(
            f"Saved embeddings for part {part_number} as {embeddings_csv_path}")

    return df


def main():
    lines = load_preprocessed_file(input_file)
    parts = split_into_parts(lines, num_parts)

    all_parts_dfs = []

    for i, part in enumerate(parts):
        df = process_part(i + 1, part)
        all_parts_dfs.append(df)

    combined_df = pd.concat(all_parts_dfs)
    combined_csv_path = f"{base_dir}/Combinedpreprocessed.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Combined embeddings saved as {combined_csv_path}")


if __name__ == "__main__":
    main()
