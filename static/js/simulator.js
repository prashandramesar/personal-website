// Page behavior for the call center simulator. Everything runs in the browser.
(function () {
  "use strict";

  // Fictional scripted calls. Pauses for a choice at any step with `choice`.
  const SCENARIOS = [
    {
      id: "router",
      label: "Router keeps dropping",
      why: "When the listening agent is confident, the human agent gets the answer in one click and can keep their attention on the customer.",
      steps: [
        { who: "customer", text: "Hi, my internet keeps dropping every few minutes since this morning." },
        { who: "listen", text: "Intent: connection drops. Confidence: high (91%)." },
        {
          who: "suggest",
          text: "How do I run a remote line check and restart a router?",
          choice: true,
          typed: "how to check line and restart router remotely",
        },
        { who: "retrieve", text: "Support guide, section 4: Remote line checks (match 0.89)" },
        { who: "retrieve", text: "Support guide, section 7: Restarting a router remotely (match 0.84)" },
        { who: "answer", text: "Run the line check from the account screen. If it passes, restart the router remotely and ask the customer to wait two minutes." },
        { who: "agent", text: "Thanks for waiting. I've checked your line and it looks fine, so I'll restart your router from here. It takes about two minutes." },
        { who: "customer", text: "Great, it's already back. Thank you!" },
      ],
    },
    {
      id: "bill",
      label: "A question only the human asks",
      why: "Not every question can be predicted. The human agent can always ask directly, and retrieval still does the searching.",
      steps: [
        { who: "customer", text: "My bill went up this month and I don't know why." },
        { who: "listen", text: "Intent: billing question. Confidence: low (42%). Too low to suggest anything, so the agent stays quiet." },
        { who: "agent", text: "Let me look into that. I can see a new charge on your account." },
        { who: "prompt", text: "Why would a monthly bill rise after a free trial?" },
        { who: "retrieve", text: "Billing policy, section 2: When free trials end (match 0.87)" },
        { who: "answer", text: "A free add-on becomes a paid one when its trial ends, unless the customer cancels first. The customer can cancel it at any time." },
        { who: "agent", text: "Your free trial of the speed upgrade has ended, which is the new charge. I can remove it for you if you'd like." },
      ],
    },
    {
      id: "upset",
      label: "When staying quiet is right",
      why: "A wrong suggestion during a tense call does more harm than no suggestion. Knowing when to stay quiet is part of the design.",
      steps: [
        { who: "customer", text: "This is the third time I've called. Nobody fixes anything and I've had enough." },
        { who: "listen", text: "Intent: unclear, possibly a complaint or cancellation. Confidence: low (35%). Customer sounds frustrated." },
        { who: "hold", text: "No suggestion shown. The human agent's full attention belongs on the customer." },
        { who: "agent", text: "I'm really sorry you've had to call three times. I've got your history here, and I'll stay with this until it's sorted." },
        { who: "customer", text: "Okay. Thank you. That's all I wanted to hear." },
      ],
    },
  ];

  const ONE_CLICK_SECONDS = 2;
  const TYPING_SECONDS = 18;

  const PANEL_STYLES = {
    listen: ["Listening agent", "bg-gray-100 text-gray-700"],
    suggest: ["Suggested prompt (NBA)", "bg-blue-50 border border-blue-200 text-blue-800"],
    prompt: ["Human agent's prompt", "bg-gray-100 text-gray-800"],
    retrieve: ["Retrieved from knowledge base", "bg-gray-100 text-gray-700"],
    answer: ["Answer for the human agent", "bg-green-100 text-green-800"],
    hold: ["No suggestion", "bg-yellow-100 text-yellow-800"],
  };

  const byId = (id) => document.getElementById(id);
  const tabs = byId("scenario-tabs");
  const transcript = byId("call-transcript");
  const panel = byId("assist-panel");
  const nextButton = byId("next-step");
  const restartButton = byId("restart-scenario");
  const outcome = byId("scenario-outcome");

  let scenario = SCENARIOS[0];
  let position = 0;
  let waitingForChoice = false;
  let typingSeconds = null;

  function bubble(who, text) {
    const isCustomer = who === "customer";
    const row = document.createElement("div");
    row.className = `flex mb-3 ${isCustomer ? "justify-start" : "justify-end"}`;
    const box = document.createElement("div");
    box.className = `rounded-lg p-3 max-w-md ${isCustomer ? "bg-gray-100 text-gray-800" : "bg-blue-600 text-white"}`;
    const name = document.createElement("p");
    name.className = "text-xs font-bold mb-1";
    name.textContent = isCustomer ? "Customer" : "Human agent";
    const body = document.createElement("p");
    body.textContent = text;
    box.append(name, body);
    row.append(box);
    transcript.append(row);
    transcript.scrollTop = transcript.scrollHeight;
  }

  function card(kind, text) {
    const [title, style] = PANEL_STYLES[kind];
    const box = document.createElement("div");
    box.className = `rounded-lg p-3 mb-3 ${style}`;
    const name = document.createElement("p");
    name.className = "text-xs font-bold mb-1";
    name.textContent = title;
    const body = document.createElement("p");
    body.textContent = text;
    box.append(name, body);
    panel.append(box);
    panel.scrollTop = panel.scrollHeight;
    return box;
  }

  function choiceButton(label, style, onClick) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.className = `font-bold py-2 px-4 rounded-lg transition duration-300 ${style}`;
    button.addEventListener("click", onClick);
    return button;
  }

  function offerChoice(box, step) {
    waitingForChoice = true;
    const row = document.createElement("div");
    row.className = "flex flex-wrap gap-2 mt-3";

    function choose(seconds, message) {
      typingSeconds = seconds;
      waitingForChoice = false;
      row.remove();
      card("prompt", message);
      render();
    }

    row.append(
      choiceButton("Use suggestion", "bg-blue-600 text-white hover:bg-blue-700", () =>
        choose(ONE_CLICK_SECONDS, `${step.text} (sent in one click)`),
      ),
      choiceButton("Type it myself", "border-2 border-blue-600 text-blue-600 hover:bg-blue-600 hover:text-white", () =>
        choose(TYPING_SECONDS, `${step.typed} (typed while the customer waits)`),
      ),
    );
    box.append(row);
  }

  function showOutcome() {
    outcome.textContent = "";
    const finished = position >= scenario.steps.length;
    outcome.hidden = !finished;
    if (!finished) {
      return;
    }
    const why = document.createElement("p");
    why.className = "font-bold";
    why.textContent = scenario.why;
    outcome.append(why);
    if (typingSeconds !== null) {
      const timing = document.createElement("p");
      timing.className = "text-sm mt-2";
      timing.textContent =
        typingSeconds === ONE_CLICK_SECONDS
          ? `Illustrative time to send the prompt: about ${ONE_CLICK_SECONDS} seconds in one click, instead of about ${TYPING_SECONDS} seconds of typing.`
          : `Illustrative time to send the prompt: about ${TYPING_SECONDS} seconds of typing, instead of about ${ONE_CLICK_SECONDS} seconds in one click.`;
      outcome.append(timing);
    }
  }

  function render() {
    const finished = position >= scenario.steps.length;
    nextButton.disabled = waitingForChoice || finished;
    nextButton.classList.toggle("opacity-50", nextButton.disabled);
    nextButton.classList.toggle("cursor-not-allowed", nextButton.disabled);
    nextButton.textContent = position === 0 ? "Start the call" : "Next step";
    showOutcome();
  }

  function advance() {
    if (waitingForChoice || position >= scenario.steps.length) {
      return;
    }
    const step = scenario.steps[position];
    position += 1;
    if (step.who === "customer" || step.who === "agent") {
      bubble(step.who, step.text);
    } else {
      const box = card(step.who, step.text);
      if (step.choice) {
        offerChoice(box, step);
      }
    }
    render();
  }

  function start(next) {
    scenario = next;
    position = 0;
    waitingForChoice = false;
    typingSeconds = null;
    transcript.textContent = "";
    panel.textContent = "";
    for (const button of tabs.children) {
      const active = button.dataset.id === scenario.id;
      button.setAttribute("aria-pressed", String(active));
      button.classList.toggle("bg-blue-600", active);
      button.classList.toggle("text-white", active);
      button.classList.toggle("bg-white", !active);
      button.classList.toggle("text-blue-600", !active);
    }
    render();
  }

  for (const item of SCENARIOS) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.id = item.id;
    button.textContent = item.label;
    button.className = "border-2 border-blue-600 font-medium py-2 px-4 rounded-lg";
    button.addEventListener("click", () => start(item));
    tabs.append(button);
  }
  nextButton.addEventListener("click", advance);
  restartButton.addEventListener("click", () => start(scenario));
  start(SCENARIOS[0]);

  // Business case
  const Model = window.SimulatorModel;
  const PRESETS = {
    pilot: { calls: 50000, adoptionPct: 50, minutesSaved: 0.75, wrongPct: 12, extraMinutes: 1.5, hourlyCost: 30, runningCost: 6000 },
    weak: { calls: 50000, adoptionPct: 50, minutesSaved: 0.75, wrongPct: 30, extraMinutes: 1.5, hourlyCost: 30, runningCost: 6000 },
    unused: { calls: 50000, adoptionPct: 15, minutesSaved: 0.75, wrongPct: 8, extraMinutes: 1.5, hourlyCost: 30, runningCost: 6000 },
    scale: { calls: 50000, adoptionPct: 75, minutesSaved: 1, wrongPct: 6, extraMinutes: 1.5, hourlyCost: 30, runningCost: 6000 },
  };
  const sliders = document.querySelectorAll("[data-input]");
  const euros = new Intl.NumberFormat("en-IE", { style: "currency", currency: "EUR", maximumFractionDigits: 0 });
  const number = new Intl.NumberFormat("en-IE", { maximumFractionDigits: 0 });
  const UNITS = {
    calls: (v) => number.format(v),
    adoptionPct: (v) => `${v}%`,
    minutesSaved: (v) => `${v} min`,
    wrongPct: (v) => `${v}%`,
    extraMinutes: (v) => `${v} min`,
    hourlyCost: (v) => euros.format(v),
    runningCost: (v) => euros.format(v),
  };
  const STAGE_STYLES = {
    "drawing-board": "bg-red-100 text-red-800",
    "fix-experience": "bg-yellow-100 text-yellow-800",
    hold: "bg-yellow-100 text-yellow-800",
    pilot: "bg-blue-50 text-blue-800",
    scale: "bg-green-100 text-green-800",
  };

  function update() {
    const inputs = {};
    for (const slider of sliders) {
      const key = slider.dataset.input;
      inputs[key] = Number(slider.value);
      byId(`value-${key}`).textContent = UNITS[key](inputs[key]);
    }
    const results = Model.calculate(inputs);
    const verdict = Model.assess(inputs, results);
    const breakEven = Model.breakEvenAdoptionPct(inputs);

    byId("result-net").textContent = euros.format(results.netMonthly);
    byId("result-net").className = `text-3xl font-bold ${results.netMonthly > 0 ? "text-green-800" : "text-red-800"}`;
    byId("result-hours").textContent = number.format(results.netHours);
    byId("result-breakeven").textContent =
      breakEven === null ? "Never at these settings" : `${number.format(Math.ceil(breakEven))}% adoption`;

    const stage = byId("rollout-stage");
    stage.className = `rounded-lg p-4 mb-4 ${STAGE_STYLES[verdict.stage.id]}`;
    byId("rollout-title").textContent = verdict.stage.title;
    byId("rollout-detail").textContent = verdict.stage.detail;

    const list = byId("rollout-checks");
    list.textContent = "";
    for (const check of verdict.checks) {
      const item = document.createElement("li");
      item.className = check.pass ? "text-green-800" : "text-red-800";
      item.textContent = `${check.pass ? "Pass" : "Not yet"}: ${check.label}`;
      list.append(item);
    }
  }

  for (const slider of sliders) {
    slider.addEventListener("input", update);
  }
  for (const button of document.querySelectorAll("[data-preset]")) {
    button.addEventListener("click", () => {
      const preset = PRESETS[button.dataset.preset];
      for (const slider of sliders) {
        slider.value = preset[slider.dataset.input];
      }
      update();
    });
  }
  update();
})();
