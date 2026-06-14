# AI FAQ Explorer

A Django website that answers questions about advanced artificial intelligence concepts using FAQ retrieval plus a neural reranking layer.

## Features

- Seeded FAQ knowledge base focused on advanced AI concepts
- Django admin for editing FAQs and categories
- NLP preprocessing with tokenization and stemming
- Hybrid matcher using cosine similarity and an ANN reranker
- Chat UI and searchable FAQ library

## Run locally

1. Install dependencies from `requirements.txt`.
2. Run `python manage.py migrate`.
3. Start the development server with `python manage.py runserver`.

## Accuracy check

Run a small top-1 exact-match evaluation against published FAQs:

```bash
python manage.py evaluate_accuracy --sample-size 10
```

This reports how often the assistant matches the same FAQ question from the database and is a practical proxy for accuracy.
