from django.contrib import admin
from django.urls import include, path

from faqs.views import home_view


urlpatterns = [
    path("admin/", admin.site.urls),
    path("", home_view, name="home"),
    path("", include("faqs.urls")),
]
