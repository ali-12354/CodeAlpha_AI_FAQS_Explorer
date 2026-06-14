from django.apps import AppConfig


class FaqsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "faqs"

    def ready(self) -> None:
        from . import signals  # noqa: F401
