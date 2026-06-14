from django import forms


class AskQuestionForm(forms.Form):
    question = forms.CharField(
        max_length=255,
        widget=forms.TextInput(
            attrs={
                "placeholder": "Ask about transformers, attention, embeddings, RAG, CNNs, and more...",
                "autocomplete": "off",
            }
        ),
    )
