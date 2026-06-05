from datasets import Dataset

N_DOCS = 2500

base_texts = [
    "This document discusses language models, datasets, memorization, privacy, and evaluation. ",
    "The study considers machine learning systems, training data, generalization, and privacy risks. ",
    "This text describes artificial intelligence, model behavior, benchmark evaluation, and data quality. ",
    "The document contains neutral academic prose about language modeling and statistical learning. ",
]

docs = []
titles = []

for i in range(N_DOCS):
    base = base_texts[i % len(base_texts)]
    text = base * 500
    docs.append(text)
    titles.append(f"doc_{i}")

ds = Dataset.from_dict({
    "text": docs,
    "title": titles,
})

ds.save_to_disk("data/raw_large_paper_style_dataset")

print("Saved dataset to data/raw_large_paper_style_dataset")
print("Documents:", N_DOCS)
