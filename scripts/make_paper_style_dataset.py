from datasets import Dataset

N_DOCS = 400

base_text = (
    "This is a synthetic document for testing copyright traps in language models. "
    "It contains neutral academic text about datasets, memorization, privacy, "
    "membership inference, language modeling, and evaluation. "
)

docs = []
titles = []

for i in range(N_DOCS):
    # long enough for trap injection and continued pretraining
    text = base_text * 400
    docs.append(text)
    titles.append(f"doc_{i}")

ds = Dataset.from_dict({
    "text": docs,
    "title": titles,
})

ds.save_to_disk("data/raw_paper_style_dataset")

print("Saved dataset to data/raw_paper_style_dataset")
print("Documents:", N_DOCS)
