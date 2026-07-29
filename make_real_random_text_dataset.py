import os
from datasets import load_dataset, Dataset

N_DOCS = 2500
TARGET_DOC_LEN = 150000 

print("Loading a real text corpus from Hugging Face....")
# Use wikitext-103-raw-v1 because it contains long, high-quality, coherent articles.
raw_data = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")

print("Generation of linked documents...")
docs = []
current_text = []
current_len = 0

for row in raw_data:
    text_line = row["text"].strip()
    if not text_line:
        continue
        
    current_text.append(text_line)
    current_len += len(text_line)
    
       if current_len >= TARGET_DOC_LEN:
        combined_text = "\n".join(current_text)
        docs.append(combined_text)
        current_text = []
        current_len = 0
        
    if len(docs) >= N_DOCS:
        break

if len(docs) < N_DOCS:
    print(f"Warning: Only {len(docs)} documents could be collected from the corpus")
    N_DOCS = len(docs)

titles = [f"doc_{i}" for i in range(N_DOCS)]

ds = Dataset.from_dict({
    "text": docs,
    "title": titles,
})

os.makedirs("data", exist_ok=True)
ds.save_to_disk("data/raw_make_real_random_text_dataset(150000lengs)")

print("\n=== The dataset was successfully created using the methodology from the paper. ===")
print("Saved at: data/raw_make_real_random_text_dataset")
print("Number of documents (N_DOCS):", N_DOCS)
print(f"Example of the beginning doc_0:\n{docs[0][:400]}...")
