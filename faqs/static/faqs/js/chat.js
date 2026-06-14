document.addEventListener("DOMContentLoaded", () => {
  const chatForm = document.getElementById("chat-form");
  const chatFeed = document.getElementById("chat-feed");
  const chatInput = document.getElementById("chat-input");
  const sampleButton = document.querySelector("[data-fill-sample]");

  if (!chatForm || !chatFeed || !chatInput) {
    return;
  }

  const askUrl = chatForm.getAttribute("action");
  const csrfToken = chatForm.querySelector("input[name='csrfmiddlewaretoken']")?.value || "";

  const appendBubble = (text, role) => {
    const bubble = document.createElement("div");
    bubble.className = `message ${role}`;
    bubble.innerHTML = `<strong>${role === "user" ? "You" : "Assistant"}</strong><p>${text}</p>`;
    chatFeed.appendChild(bubble);
    chatFeed.scrollTop = chatFeed.scrollHeight;
  };

  const sendQuestion = async (question) => {
    appendBubble(question, "user");
    const typingBubble = document.createElement("div");
    typingBubble.className = "message bot typing";
    typingBubble.innerHTML = "<strong>Assistant</strong><p>Thinking through the FAQ set...</p>";
    chatFeed.appendChild(typingBubble);

    try {
      const response = await fetch(askUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": csrfToken,
        },
        body: JSON.stringify({ question }),
      });
      const data = await response.json();
      typingBubble.innerHTML = `<strong>Assistant${data.confidence ? ` • ${(data.confidence * 100).toFixed(1)}% match` : ""}</strong><p>${data.answer}</p>`;
      typingBubble.classList.remove("typing");
    } catch (error) {
      typingBubble.innerHTML = "<strong>Assistant</strong><p>The chatbot is temporarily unavailable. Please try again.</p>";
      typingBubble.classList.remove("typing");
    }
    chatFeed.scrollTop = chatFeed.scrollHeight;
  };

  chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = chatInput.value.trim();
    if (!question) {
      return;
    }
    chatInput.value = "";
    sendQuestion(question);
  });

  sampleButton?.addEventListener("click", () => {
    chatInput.value = "What is retrieval augmented generation?";
    chatInput.focus();
  });

  document.querySelectorAll("[data-suggestion]").forEach((button) => {
    button.addEventListener("click", () => {
      chatInput.value = button.getAttribute("data-suggestion") || "";
      chatInput.focus();
    });
  });
});
