// Business case calculations for the call center simulator. All figures on the
// page are synthetic assumptions. Kept free of page code so it can be tested alone.
(function (root) {
  "use strict";

  const THRESHOLDS = {
    maxWrongPct: 20,
    targetWrongPct: 10,
    minAdoptionPct: 30,
    targetAdoptionPct: 60,
  };

  function calculate(inputs) {
    const assistedCalls = inputs.calls * (inputs.adoptionPct / 100);
    const minutesSaved = assistedCalls * inputs.minutesSaved;
    const minutesLost =
      assistedCalls * (inputs.wrongPct / 100) * inputs.extraMinutes;
    const netHours = (minutesSaved - minutesLost) / 60;
    const staffSaving = netHours * inputs.hourlyCost;
    return {
      assistedCalls,
      minutesSaved,
      minutesLost,
      netHours,
      staffSaving,
      netMonthly: staffSaving - inputs.runningCost,
    };
  }

  // The adoption level where savings cover running costs, or null if none can.
  function breakEvenAdoptionPct(inputs) {
    const netMinutesPerCall =
      inputs.minutesSaved - (inputs.wrongPct / 100) * inputs.extraMinutes;
    const valuePerCall = (netMinutesPerCall / 60) * inputs.hourlyCost;
    if (inputs.calls <= 0 || valuePerCall <= 0) {
      return null;
    }
    const pct = (inputs.runningCost / (inputs.calls * valuePerCall)) * 100;
    return pct <= 100 ? pct : null;
  }

  function assess(inputs, results) {
    const t = THRESHOLDS;
    const checks = [
      {
        label: `Wrong suggestions at or below ${t.maxWrongPct}%`,
        pass: inputs.wrongPct <= t.maxWrongPct,
      },
      {
        label: `Adoption at or above ${t.minAdoptionPct}%`,
        pass: inputs.adoptionPct >= t.minAdoptionPct,
      },
      { label: "Savings cover running costs", pass: results.netMonthly > 0 },
      {
        label: `Ready to scale: wrong suggestions at or below ${t.targetWrongPct}% and adoption at or above ${t.targetAdoptionPct}%`,
        pass:
          inputs.wrongPct <= t.targetWrongPct &&
          inputs.adoptionPct >= t.targetAdoptionPct,
      },
    ];

    let stage;
    if (inputs.wrongPct > t.maxWrongPct) {
      stage = {
        id: "drawing-board",
        title: "Delay and go back to the drawing board",
        detail:
          "Too many suggestions are wrong. Launching now would cost trust as well as time, however good the savings look.",
      };
    } else if (inputs.adoptionPct < t.minAdoptionPct) {
      stage = {
        id: "fix-experience",
        title: "Fix the experience before scaling",
        detail:
          "The suggestions are good enough, but too few people use them. Improve the user experience to drive adoption.",
      };
    } else if (results.netMonthly <= 0) {
      stage = {
        id: "hold",
        title: "Hold: the case doesn't pay yet",
        detail:
          "Quality and adoption pass, but savings don't cover running costs at this volume.",
      };
    } else if (!checks[3].pass) {
      stage = {
        id: "pilot",
        title: "Keep going in a pilot",
        detail:
          "The case pays, and it's worth expanding carefully while quality and adoption improve.",
      };
    } else {
      stage = {
        id: "scale",
        title: "Scale it up",
        detail: "Quality, adoption and the business case all clear the bar.",
      };
    }
    return { stage, checks };
  }

  root.SimulatorModel = { THRESHOLDS, calculate, breakEvenAdoptionPct, assess };
})(window);
