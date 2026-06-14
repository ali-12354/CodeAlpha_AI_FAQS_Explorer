from __future__ import annotations

import csv
import io

from django.contrib import admin, messages
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect
from django.template.response import TemplateResponse
from django.urls import path, reverse
from django.utils.text import slugify

from .models import FAQ, FAQCategory


@admin.register(FAQCategory)
class FAQCategoryAdmin(admin.ModelAdmin):
    list_display = ("name", "slug")
    search_fields = ("name", "slug", "description")
    prepopulated_fields = {"slug": ("name",)}


@admin.register(FAQ)
class FAQAdmin(admin.ModelAdmin):
    change_list_template = "admin/faqs/faq/change_list.html"
    list_display = ("question", "category", "intent_label", "is_published", "sort_order")
    list_filter = (("category", admin.RelatedOnlyFieldListFilter), "is_published")
    search_fields = ("question", "answer", "intent_label")
    list_editable = ("is_published", "sort_order")
    autocomplete_fields = ("category",)
    actions = ("export_selected_as_csv",)
    list_per_page = 25

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path("import-csv/", self.admin_site.admin_view(self.import_csv_view), name="faqs_faq_import_csv"),
            path("export-csv/", self.admin_site.admin_view(self.export_csv_view), name="faqs_faq_export_csv"),
        ]
        return custom_urls + urls

    def export_selected_as_csv(self, request: HttpRequest, queryset):
        return self._build_csv_response(queryset, "selected_faqs.csv")

    export_selected_as_csv.short_description = "Export selected FAQs as CSV"

    def import_csv_view(self, request: HttpRequest):
        if request.method == "POST":
            upload = request.FILES.get("csv_file")
            if not upload:
                messages.error(request, "Please choose a CSV file to import.")
                return redirect("..")

            decoded = upload.read().decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(decoded))
            created_count = 0
            updated_count = 0

            for row in reader:
                question = (row.get("question") or "").strip()
                answer = (row.get("answer") or "").strip()
                category_name = (row.get("category") or row.get("category_name") or "").strip()
                category_slug = (row.get("category_slug") or slugify(category_name) or "uncategorized").strip()
                category_description = (row.get("category_description") or "").strip()
                intent_label = (row.get("intent_label") or "").strip()
                is_published = str(row.get("is_published", "true")).strip().lower() in {"1", "true", "yes", "y", "on"}
                sort_order = int(row.get("sort_order") or 0)

                if not question or not answer:
                    continue

                category, _ = FAQCategory.objects.get_or_create(
                    slug=category_slug,
                    defaults={"name": category_name or category_slug.replace("-", " ").title(), "description": category_description},
                )
                updated_fields = []
                if category_description and category.description != category_description:
                    category.description = category_description
                    updated_fields.append("description")
                if category_name and category.name != category_name:
                    category.name = category_name
                    updated_fields.append("name")
                if updated_fields:
                    category.save(update_fields=updated_fields)

                faq, created = FAQ.objects.update_or_create(
                    category=category,
                    question=question,
                    defaults={
                        "answer": answer,
                        "intent_label": intent_label,
                        "is_published": is_published,
                        "sort_order": sort_order,
                    },
                )
                if created:
                    created_count += 1
                else:
                    updated_count += 1

            messages.success(
                request,
                f"Import complete: {created_count} created, {updated_count} updated.",
            )
            return redirect(reverse("admin:faqs_faq_changelist"))

        context = dict(
            self.admin_site.each_context(request),
            title="Import FAQs from CSV",
            opts=self.model._meta,
        )
        return TemplateResponse(request, "admin/faqs/faq/import_csv.html", context)

    def export_csv_view(self, request: HttpRequest):
        queryset = self.get_queryset(request).select_related("category")
        return self._build_csv_response(queryset, "faqs_export.csv")

    def _build_csv_response(self, queryset, filename: str) -> HttpResponse:
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = f'attachment; filename="{filename}"'

        writer = csv.writer(response)
        writer.writerow([
            "question",
            "answer",
            "category",
            "category_slug",
            "category_description",
            "intent_label",
            "is_published",
            "sort_order",
        ])
        for faq in queryset:
            writer.writerow([
                faq.question,
                faq.answer,
                faq.category.name,
                faq.category.slug,
                faq.category.description,
                faq.intent_label,
                "yes" if faq.is_published else "no",
                faq.sort_order,
            ])
        return response

