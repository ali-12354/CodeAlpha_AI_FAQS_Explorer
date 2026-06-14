from django.urls import path

from . import views


urlpatterns = [
    path("chat/", views.chat_page, name="chat-page"),
    path("browse/", views.browse_view, name="browse-faqs"),
    path("faq/<int:faq_id>/", views.faq_detail, name="faq-detail"),
    path("api/ask/", views.ask_api, name="ask-api"),
]
