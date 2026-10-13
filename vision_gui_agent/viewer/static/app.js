// Alpine component for the run-detail page. Server-rendered markup does the work;
// this only tracks screenshot/overlay toggles, the hovered element and keyboard nav.
function readPreference(key, fallback) {
  try {
    const value = localStorage.getItem("viewer." + key);
    return value === null ? fallback : value === "true";
  } catch (_) {
    return fallback;
  }
}

function writePreference(key, value) {
  try { localStorage.setItem("viewer." + key, String(value)); } catch (_) { /* storage unavailable */ }
}

function stepView() {
  return {
    marked: readPreference("marked", true),
    boxes: readPreference("boxes", false),
    hover: null,
    elements: [],
    init() {
      const source = this.$root.querySelector(".step-elements");
      try { this.elements = source ? JSON.parse(source.textContent) : []; } catch (_) { this.elements = []; }
      this.$watch("marked", (value) => writePreference("marked", value));
      this.$watch("boxes", (value) => writePreference("boxes", value));
    },
    go(direction) {
      const target = document.activeElement;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      const link = this.$root.querySelector(`#steps a[data-nav="${direction}"]`);
      if (link) link.click();
    },
  };
}
