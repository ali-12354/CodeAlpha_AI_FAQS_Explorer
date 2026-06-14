from django.db.models.signals import post_migrate
from django.dispatch import receiver

from .models import FAQ, FAQCategory
from .seed_data import SEED_FAQS


@receiver(post_migrate)
def seed_faqs(sender, **kwargs) -> None:
    if sender.name != "faqs":
        return

    for item in SEED_FAQS:
        category, _ = FAQCategory.objects.update_or_create(
            slug=item.category_slug,
            defaults={"name": item.category_name, "description": item.category_description},
        )

        FAQ.objects.update_or_create(
            category=category,
            question=item.question,
            defaults={
                "answer": item.answer,
                "intent_label": item.intent_label,
                "sort_order": item.sort_order,
                "is_published": True,
            },
        )
