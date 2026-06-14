from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import hashlib

import numpy as np
from django.db.models import Count, Max
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from nltk.stem import PorterStemmer
except Exception:  # pragma: no cover - fallback if nltk is unavailable at runtime
    PorterStemmer = None

from .models import FAQ

REFERENCE_DOCS_DIR = Path(__file__).resolve().parent / "reference_docs"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
    "you",
    "your",
}


RULE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(safe|safety|harmful things|harmful outputs?|jailbreak|prompt injection|follow (my )?instructions|ignore (its|the) rules|aligned)\b"), "ai_safety"),
    (re.compile(r"\b(alignment|intended policy|policy adherence|stay aligned|behave correctly|follow (my )?rules|rules? compliance|ignoring (its|the) rules|ignore (its|the) rules)\b"), "ai_alignment"),
    (re.compile(r"\b(rlhf|human feedback|preference feedback)\b"), "rlhf"),
    (re.compile(r"\b(red team|red teaming|probe .* risks|find failures)\b"), "red_teaming"),
    (re.compile(r"\b(refusal|decline unsafe|unsafe requests|refuse requests)\b"), "refusal_behavior"),
    (re.compile(r"\b(monit(or|oring)|drift|abuse|deployed)\b"), "model_monitoring"),
    (re.compile(r"\b(supervised|labeled examples?|examples? with labels|examples? that already have labels|train on labels|teach from examples|labels? as examples)\b"), "supervised_ml"),
    (re.compile(r"\b(logistic regression|sigmoid|probability of a class)\b"), "logistic_regression"),
    (re.compile(r"\b(decision tree|branch(es)?|split on feature)\b"), "decision_tree_learning"),
    (re.compile(r"\b(random forest|ensemble of trees|many trees)\b"), "random_forest"),
    (re.compile(r"\b(support vector machine|svm|largest margin|separate classes)\b"), "svm"),
    (re.compile(r"\b(k nearest neighbors|k-nearest neighbors|nearest examples|closest examples)\b"), "knn"),
    (re.compile(r"\b(naive bayes|bayes theorem|probabilistic classifier)\b"), "naive_bayes"),
    (re.compile(r"\b(cluster|clustering|group similar data)\b"), "clustering"),
    (re.compile(r"\b(hyperparameter|hyperparameter optimization|best validation performance)\b"), "hyperparameter_optimization"),
    (re.compile(r"\b(converg(e|ence)|loss or validation metrics stop improving)\b"), "model_convergence"),
    (re.compile(r"\b(computer vision|visual data|video|videos|vision)\b"), "computer_vision_definition"),
    (re.compile(r"\b(object detection|find objects|bounding boxes? around objects)\b"), "object_detection"),
        (re.compile(r"\b(cross-validation|cross validation|train-test splits?|split data to test|test generalization|generalization across folds?)\b"), "cross_validation"),
    (re.compile(r"\b(image segmentation|segment(ing)?|pixel(s)? and boundaries)\b"), "image_segmentation"),
    (re.compile(r"\b(ocr|optical character recognition|scanned documents|extract text from image)\b"), "ocr"),
    (re.compile(r"\b(face recognition|verify a person|identify a person)\b"), "face_recognition"),
    (re.compile(r"\b(image generation|text to image|text-to-image|written description|from a written description|create an image from a text prompt|generate image)\b"), "image_generation"),
    (re.compile(r"\b(diffusion model|remove noise step by step)\b"), "diffusion_model"),
    (re.compile(r"\b(generative ai|create new content|generate content)\b"), "generative_ai_definition"),
    (re.compile(r"\b(multimodal|multiple modalities|text images audio video)\b"), "multimodal_ai_definition"),
    (re.compile(r"\b(image captioning|turn a photo into text|describe an image)\b"), "image_captioning"),
    (re.compile(r"\b(visual question answering|answer questions about an image|vqa)\b"), "vqa"),
    (re.compile(r"\b(search documents|answer from documents|search with a language model|combine search with a language model|grounded in documents|retrieval augmented generation|rag)\b"), "rag_definition"),
        (re.compile(r"\b(reduce overfitting|prevent overfitting|combat overfitting|regularization|regularize)\b"), "regularization"),
        (re.compile(r"\b(remembers? sequence information step by step|sequence information step by step|step by step|time step by step)\b"), "rnn_definition"),
    (re.compile(r"\b(evaluate|evaluation|metric|correctness|accuracy|precision|recall|f1)\b"), "model_evaluation"),
]


_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


_embedding_cache: dict = {"fingerprint": None, "embeddings": None, "faqs": None}


def _faqs_fingerprint(faqs: list[FAQ]) -> str:
    h = hashlib.md5()
    for faq in faqs:
        h.update(f"{faq.id}:{faq.updated_at}".encode())
    return h.hexdigest()


def get_faq_embeddings(faqs: list[FAQ]):
    fingerprint = _faqs_fingerprint(faqs)
    if _embedding_cache["fingerprint"] != fingerprint:
        model = get_embedding_model()
        texts = [f"{faq.question} {faq.answer}" for faq in faqs]
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        _embedding_cache.update(fingerprint=fingerprint, embeddings=embeddings, faqs=faqs)
    return _embedding_cache["embeddings"], _embedding_cache["faqs"]


def semantic_search(question: str, faqs: list[FAQ], top_k: int = 5):
    if not faqs:
        return [], 0.0

    embeddings, cached_faqs = get_faq_embeddings(faqs)
    model = get_embedding_model()
    query_vec = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]

    scores = embeddings @ query_vec
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = [(cached_faqs[i], float(scores[i])) for i in top_indices]
    best_score = results[0][1] if results else 0.0
    return results, best_score


@dataclass(frozen=True)
class MatchResult:
    faq: FAQ | None
    confidence: float
    method: str
    matched_question: str
    answer: str
    category: str
    semantic_matches: list[tuple[FAQ, float]] | None = None
    fallback_message: str | None = None


@dataclass(frozen=True)
class ReferenceDocument:
    title: str
    path: str
    content: str


def _top_related_faqs(question: str, faqs: list[FAQ], word_vectorizer: TfidfVectorizer, char_vectorizer: TfidfVectorizer, word_matrix, char_matrix, limit: int = 3) -> list[FAQ]:
    if not faqs or word_matrix is None or char_matrix is None:
        return []

    word_query = word_vectorizer.transform([question])
    char_query = char_vectorizer.transform([question])
    word_scores = cosine_similarity(word_query, word_matrix).flatten()
    char_scores = cosine_similarity(char_query, char_matrix).flatten()
    combined_scores = (0.7 * word_scores) + (0.3 * char_scores)
    ranked_indexes = combined_scores.argsort()[::-1][:limit]
    return [faqs[int(index)] for index in ranked_indexes]


def _load_reference_documents() -> list[ReferenceDocument]:
    if not REFERENCE_DOCS_DIR.exists():
        return []

    documents: list[ReferenceDocument] = []
    for path in sorted(REFERENCE_DOCS_DIR.glob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        first_line = content.splitlines()[0].strip()
        title = first_line.lstrip("# ").strip() if first_line else path.stem.replace("_", " ").title()
        if not title:
            title = path.stem.replace("_", " ").title()
        documents.append(ReferenceDocument(title=title, path=str(path), content=content))
    return documents


def _top_reference_documents(question: str, documents: list[ReferenceDocument], word_vectorizer: TfidfVectorizer, char_vectorizer: TfidfVectorizer, word_matrix, char_matrix, limit: int = 2) -> list[ReferenceDocument]:
    if not documents or word_matrix is None or char_matrix is None:
        return []

    word_query = word_vectorizer.transform([question])
    char_query = char_vectorizer.transform([question])
    word_scores = cosine_similarity(word_query, word_matrix).flatten()
    char_scores = cosine_similarity(char_query, char_matrix).flatten()
    combined_scores = (0.7 * word_scores) + (0.3 * char_scores)
    ranked_indexes = combined_scores.argsort()[::-1][:limit]
    return [documents[int(index)] for index in ranked_indexes]


def _doc_snippet(content: str, sentence_count: int = 2) -> str:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", content.strip()) if sentence.strip()]
    if not sentences:
        return content.strip()
    return " ".join(sentences[:sentence_count]).strip()


def _expand_answer(faq: FAQ, related_faqs: list[FAQ] | None = None) -> str:
    related_faqs = related_faqs or []
    related_topics = [item.question for item in related_faqs if item.id != faq.id][:2]

    parts = [faq.answer.strip()]
    if faq.category.description:
        parts.append(f"This belongs to the {faq.category.name} area, which covers {faq.category.description.lower()}")

    if related_topics:
        if len(related_topics) == 1:
            parts.append(f"A closely related topic is {related_topics[0].lower()}, which can add useful context.")
        else:
            parts.append(
                "Related topics that help explain this better are "
                + ", ".join(topic.lower() for topic in related_topics)
                + "."
            )

    parts.append(
        "In practice, the key idea is to focus on the core definition first, then apply it to the specific AI system or task you are working with."
    )
    return " ".join(parts)


def _expand_answer_with_documents(faq: FAQ, related_faqs: list[FAQ] | None = None, related_docs: list[ReferenceDocument] | None = None) -> str:
    related_docs = related_docs or []
    answer = _expand_answer(faq, related_faqs)
    if not related_docs:
        return answer

    doc = related_docs[0]
    snippet = _doc_snippet(doc.content, sentence_count=2)
    return (
        f"{answer} Reference note from {doc.title}: {snippet}"
    )


def _build_rag_answer(question: str, related_faqs: list[FAQ]) -> str:
    related_topics = [f"{faq.question} ({faq.category.name})" for faq in related_faqs[:3]]
    if related_topics:
        return (
            "I do not have a direct FAQ answer for that yet, but the closest related topics in my knowledge base are: "
            + "; ".join(related_topics)
            + ". Try asking with more AI-specific terms, or narrow the topic so I can map it to the best concept."
        )

    return (
        "I do not have a direct FAQ answer for that yet. Please narrow the question or ask about a topic covered in the knowledge base."
    )


def _build_document_rag_answer(question: str, related_docs: list[ReferenceDocument], related_faqs: list[FAQ] | None = None) -> str:
    related_faqs = related_faqs or []
    if related_docs:
        doc = related_docs[0]
        snippet = _doc_snippet(doc.content, sentence_count=3)
        faq_context = ""
        if related_faqs:
            faq_context = " Related FAQ topics: " + "; ".join(faq.question for faq in related_faqs[:3]) + "."
        return (
            f"I do not have a direct FAQ answer for that yet, but the strongest reference I found is '{doc.title}'. "
            f"{snippet}{faq_context}"
        )

    return _build_rag_answer(question, related_faqs)


def _normalize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    if PorterStemmer is not None:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def _normalize_topic(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip().rstrip("?.!"))
    return cleaned.lower()


def _strip_prefix(question: str, prefixes: tuple[str, ...]) -> str | None:
    lowered = question.lower().strip()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return question[len(prefix):].strip(" ?.")
    return None


def _generate_query_variants(question: str, intent_label: str, answer: str, category_name: str) -> list[str]:
    topic = _normalize_topic(question)
    variants: set[str] = {topic}

    prefix_variants = [
        ("what is ", ["define {topic}", "explain {topic}", "tell me about {topic}", "what does {topic} mean", "how do you define {topic}", "{topic} definition"]),
        ("what are ", ["define {topic}", "explain {topic}", "tell me about {topic}", "what do {topic} mean", "{topic} definition"]),
        ("what’s ", ["what is {topic}", "define {topic}", "explain {topic}", "tell me about {topic}"]),
        ("define ", ["what is {topic}", "explain {topic}", "tell me about {topic}"]),
        ("explain ", ["what is {topic}", "define {topic}", "tell me about {topic}"]),
        ("tell me about ", ["what is {topic}", "define {topic}", "explain {topic}"]),
        ("how do ", ["how does {topic}", "explain how {topic}", "what is the best way to {topic}"]),
        ("how does ", ["how do {topic}", "explain how {topic}", "what is the best way to {topic}"]),
        ("how can i ", ["what is the best way to {topic}", "how do i {topic}", "how can i {topic}"]),
        ("why do ", ["why is {topic}", "what causes {topic}", "why is {topic} important"]),
        ("why does ", ["why is {topic}", "what causes {topic}", "why is {topic} important"]),
        ("what is the difference between ", ["compare {topic}", "how is {topic} different", "difference between {topic}"]),
    ]

    for prefix, templates in prefix_variants:
        stripped = _strip_prefix(question, (prefix,))
        if stripped:
            variants.add(_normalize_topic(stripped))
            for template in templates:
                variants.add(template.format(topic=stripped))

    if "_" in intent_label or "-" in intent_label:
        label_text = intent_label.replace("_", " ").replace("-", " ").strip()
        variants.update(
            {
                label_text,
                f"what is {label_text}",
                f"define {label_text}",
                f"explain {label_text}",
                f"tell me about {label_text}",
            }
        )

    answer_tokens = " ".join(_normalize(answer))
    if answer_tokens:
        variants.update(
            {
                f"what is {answer_tokens}",
                f"explain {answer_tokens}",
                f"tell me about {answer_tokens}",
            }
        )

    category_tokens = _normalize_topic(category_name)
    if category_tokens:
        variants.update(
            {
                f"{category_tokens} question",
                f"{category_tokens} concept",
                f"{category_tokens} definition",
            }
        )

    return sorted(variant for variant in variants if variant)


class FAQMatcher:
    def __init__(self, faqs: Iterable[FAQ]):
        self.faqs = list(faqs)
        self.intent_lookup = {faq.intent_label: faq for faq in self.faqs if faq.intent_label}
        self.question_lookup = {faq.question.strip().lower(): faq for faq in self.faqs}
        self.training_texts: list[str] = []
        for faq in self.faqs:
            variants = _generate_query_variants(faq.question, faq.intent_label, faq.answer, faq.category.name)
            training_text = " ".join(
                [faq.question, faq.answer, faq.intent_label, faq.category.name, *variants]
            )
            self.training_texts.append(training_text)

        self.word_vectorizer = TfidfVectorizer(tokenizer=_normalize, ngram_range=(1, 2))
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))

        self.word_matrix = self.word_vectorizer.fit_transform(self.training_texts) if self.training_texts else None
        self.char_matrix = self.char_vectorizer.fit_transform(self.training_texts) if self.training_texts else None

        self.reference_documents = _load_reference_documents()
        self.reference_texts = [f"{doc.title} {doc.content}" for doc in self.reference_documents]
        self.reference_word_vectorizer = TfidfVectorizer(tokenizer=_normalize, ngram_range=(1, 2))
        self.reference_char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.reference_word_matrix = self.reference_word_vectorizer.fit_transform(self.reference_texts) if self.reference_texts else None
        self.reference_char_matrix = self.reference_char_vectorizer.fit_transform(self.reference_texts) if self.reference_texts else None

    def _rule_match(self, question: str) -> FAQ | None:
        normalized = _normalize_topic(question)
        direct = self.question_lookup.get(normalized)
        if direct is not None:
            return direct

        for pattern, intent_label in RULE_PATTERNS:
            if pattern.search(normalized):
                faq = self.intent_lookup.get(intent_label)
                if faq is not None:
                    return faq
        return None

    def match(self, question: str) -> MatchResult:
        if not self.faqs or self.word_matrix is None or self.char_matrix is None:
            return MatchResult(
                faq=None,
                confidence=0.0,
                method="fallback",
                matched_question="",
                answer="I do not have any FAQs loaded yet. Please add some advanced AI questions in the admin panel.",
                category="",
                fallback_message="No FAQs available.",
            )

        ruled = self._rule_match(question)
        if ruled is not None:
            related_faqs = _top_related_faqs(question, self.faqs, self.word_vectorizer, self.char_vectorizer, self.word_matrix, self.char_matrix, limit=3)
            related_docs = _top_reference_documents(question, self.reference_documents, self.reference_word_vectorizer, self.reference_char_vectorizer, self.reference_word_matrix, self.reference_char_matrix, limit=2)
            return MatchResult(
                faq=ruled,
                confidence=0.98,
                method="rules",
                matched_question=ruled.question,
                answer=_expand_answer_with_documents(ruled, related_faqs, related_docs),
                category=ruled.category.name,
            )

        word_query = self.word_vectorizer.transform([question])
        char_query = self.char_vectorizer.transform([question])
        word_scores = cosine_similarity(word_query, self.word_matrix).flatten() if self.word_matrix is not None else 0.0
        char_scores = cosine_similarity(char_query, self.char_matrix).flatten() if self.char_matrix is not None else 0.0
        cosine_scores = (0.7 * word_scores) + (0.3 * char_scores)
        best_index = int(cosine_scores.argmax())
        best_faq = self.faqs[best_index]
        cosine_score = float(cosine_scores[best_index])

        combined_score = cosine_score
        chosen = best_faq
        method = "hybrid"

        confidence = round(min(1.0, max(0.0, combined_score)), 3)
        if confidence < 0.28:
            semantic_results, best_semantic_score = semantic_search(question, self.faqs, top_k=5)

            if best_semantic_score >= 0.70:
                top_faq, top_score = semantic_results[0]
                related_faqs = [faq for faq, _ in semantic_results[1:4]]
                related_docs = _top_reference_documents(question, self.reference_documents, self.reference_word_vectorizer, self.reference_char_vectorizer, self.reference_word_matrix, self.reference_char_matrix, limit=2)
                return MatchResult(
                    faq=top_faq,
                    confidence=round(best_semantic_score, 3),
                    method="semantic",
                    matched_question=top_faq.question,
                    answer=_expand_answer_with_documents(top_faq, related_faqs, related_docs),
                    category=top_faq.category.name,
                    semantic_matches=semantic_results,
                )

            related_faqs = _top_related_faqs(question, self.faqs, self.word_vectorizer, self.char_vectorizer, self.word_matrix, self.char_matrix, limit=3)
            related_docs = _top_reference_documents(question, self.reference_documents, self.reference_word_vectorizer, self.reference_char_vectorizer, self.reference_word_matrix, self.reference_char_matrix, limit=2)
            return MatchResult(
                faq=chosen,
                confidence=confidence,
                method="rag" if (related_faqs or related_docs) else method,
                matched_question=chosen.question,
                answer=_build_document_rag_answer(question, related_docs, related_faqs),
                category=chosen.category.name,
                semantic_matches=semantic_results if semantic_results else None,
                fallback_message="Low-confidence match.",
            )

        return MatchResult(
            faq=chosen,
            confidence=confidence,
            method=method,
            matched_question=chosen.question,
            answer=_expand_answer_with_documents(
                chosen,
                _top_related_faqs(question, self.faqs, self.word_vectorizer, self.char_vectorizer, self.word_matrix, self.char_matrix, limit=3),
                _top_reference_documents(question, self.reference_documents, self.reference_word_vectorizer, self.reference_char_vectorizer, self.reference_word_matrix, self.reference_char_matrix, limit=2),
            ),
            category=chosen.category.name,
        )


def _cache_signature() -> tuple[int, str | None]:
    stats = FAQ.objects.filter(is_published=True).aggregate(total=Count("id"), latest=Max("updated_at"))
    latest = stats["latest"]
    return stats["total"] or 0, latest.isoformat() if latest else None


@lru_cache(maxsize=8)
def _build_matcher(signature: tuple[int, str | None]) -> FAQMatcher:
    return FAQMatcher(FAQ.objects.filter(is_published=True).select_related("category"))


def get_matcher() -> FAQMatcher:
    return _build_matcher(_cache_signature())


def match_question(question: str) -> MatchResult:
    question = question.strip()
    if not question:
        return MatchResult(
            faq=None,
            confidence=0.0,
            method="empty",
            matched_question="",
            answer="Please type a question about advanced AI concepts.",
            category="",
            fallback_message="Empty question.",
        )
    return get_matcher().match(question)
 