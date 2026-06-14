from __future__ import annotations

import random

from django.core.management.base import BaseCommand, CommandError

from faqs.models import FAQ
from faqs.services import match_question


class Command(BaseCommand):
    help = "Evaluate the assistant's top-1 exact-match accuracy against published FAQs."

    def add_arguments(self, parser):
        parser.add_argument(
            "--sample-size",
            type=int,
            default=10,
            help="Number of published FAQs to sample for the accuracy check.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed used when sampling FAQs.",
        )

    def handle(self, *args, **options):
        sample_size = options["sample_size"]
        seed = options["seed"]

        faqs = list(FAQ.objects.filter(is_published=True).select_related("category"))
        if not faqs:
            raise CommandError("No published FAQs are available for evaluation.")

        if sample_size <= 0:
            raise CommandError("--sample-size must be greater than 0.")

        if sample_size > len(faqs):
            sample_size = len(faqs)

        rng = random.Random(seed)
        sample = rng.sample(faqs, sample_size)

        correct = 0
        rows: list[tuple[str, str, str, float, bool]] = []

        for faq in sample:
            result = match_question(faq.question)
            is_correct = result.faq is not None and result.faq.id == faq.id
            correct += int(is_correct)
            rows.append((faq.question, result.matched_question, result.category, result.confidence, is_correct))

        accuracy = correct / sample_size

        self.stdout.write(self.style.SUCCESS("Accuracy evaluation complete"))
        self.stdout.write(f"Sample size: {sample_size}")
        self.stdout.write(f"Correct matches: {correct}")
        self.stdout.write(f"Accuracy: {accuracy:.2%}")
        self.stdout.write("")
        self.stdout.write("Examples:")
        for source_question, matched_question, category, confidence, is_correct in rows:
            status = "OK" if is_correct else "MISS"
            self.stdout.write(
                f"- [{status}] source={source_question!r} | matched={matched_question!r} | category={category!r} | confidence={confidence:.3f}"
            )