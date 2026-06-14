from __future__ import annotations

import json

from django.db.models import Count
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_GET, require_POST

from .forms import AskQuestionForm
from .models import FAQ, FAQCategory
from .services import match_question


@require_GET
def home_view(request):
    categories = FAQCategory.objects.annotate(faq_count=Count("faqs")).order_by("name")
    featured_faqs = FAQ.objects.filter(is_published=True).select_related("category")[:6]
    total_faqs = FAQ.objects.filter(is_published=True).count()
    return render(
        request,
        "faqs/home.html",
        {
            "categories": categories,
            "featured_faqs": featured_faqs,
            "total_faqs": total_faqs,
            "chat_form": AskQuestionForm(),
        },
    )


@require_GET
def chat_page(request):
    return render(request, "faqs/chat.html", {"chat_form": AskQuestionForm()})


@require_GET
def browse_view(request):
    categories = FAQCategory.objects.prefetch_related("faqs").order_by("name")
    return render(request, "faqs/browse.html", {"categories": categories})


@require_GET
def faq_detail(request, faq_id: int):
    faq = get_object_or_404(FAQ.objects.select_related("category"), pk=faq_id, is_published=True)
    return render(request, "faqs/detail.html", {"faq": faq})


@require_POST
def ask_api(request):
    if request.content_type == "application/json":
        payload = json.loads(request.body.decode("utf-8") or "{}")
        question = (payload.get("question") or "").strip()
    else:
        question = (request.POST.get("question") or "").strip()

    result = match_question(question)
    return JsonResponse(
        {
            "question": question,
            "answer": result.answer,
            "matched_question": result.matched_question,
            "category": result.category,
            "confidence": result.confidence,
            "method": result.method,
            "fallback_message": result.fallback_message,
            "semantic_matches": (
                [{"question": f.question, "score": round(s, 3)} for f, s in result.semantic_matches]
                if result.semantic_matches else []
            ),
        }
    )
 