// Sends questions to the assistant API and shows the results.
(function () {
  "use strict";

  const form = document.getElementById("ask-form");
  const input = document.getElementById("question");
  const answers = document.getElementById("answers");

  function element(tag, className, text) {
    const node = document.createElement(tag);
    node.className = className;
    if (text) {
      node.textContent = text;
    }
    return node;
  }

  function showMessage(text, className) {
    answers.textContent = "";
    answers.append(element("div", `rounded-lg p-4 ${className}`, text));
  }

  function showPassage(passage) {
    const card = element("div", "bg-white rounded-lg shadow-md p-6 mb-4");
    card.append(element("h2", "text-xl font-bold mb-2", passage.title));
    card.append(element("p", "text-gray-700 mb-3", passage.text));
    const link = element("a", "text-blue-600 hover:underline text-sm", `Source: ${passage.source_label}`);
    link.href = passage.source_url;
    card.append(link);
    answers.append(card);
  }

  async function ask(question) {
    showMessage("Searching...", "bg-gray-100 text-gray-700");
    try {
      const response = await fetch("/api/assistant/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const body = await response.json();
      if (!body.answered) {
        showMessage(
          "I couldn't find that in what Prashand has shared here. Try one of the questions above.",
          "bg-yellow-100 text-yellow-800",
        );
        return;
      }
      answers.textContent = "";
      body.passages.forEach(showPassage);
    } catch (error) {
      console.error(error);
      showMessage("Something went wrong. Please try again.", "bg-red-100 text-red-800");
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (question) {
      ask(question);
    }
  });

  for (const button of document.querySelectorAll("[data-question]")) {
    button.addEventListener("click", () => {
      input.value = button.dataset.question;
      ask(button.dataset.question);
    });
  }
})();
