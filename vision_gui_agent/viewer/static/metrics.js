// Chart.js renders the metrics already embedded in the page; tables beside each chart carry the exact numbers.
(function () {
  const metrics = JSON.parse(document.getElementById("metrics-data").textContent);
  const style = getComputedStyle(document.documentElement);
  const color = (name, fallback) => style.getPropertyValue(name).trim() || fallback;
  Chart.defaults.color = color("--pico-muted-color", "#666");
  Chart.defaults.borderColor = color("--pico-muted-border-color", "#ddd");
  const series = [color("--viewer-series-1", "#2a6fdb"), color("--viewer-series-2", "#e07b00"), color("--viewer-series-3", "#2e9e6a")];

  const latency = document.getElementById("latency-chart");
  if (latency && metrics.latency.length) {
    new Chart(latency, {
      type: "bar",
      data: {
        labels: metrics.latency.map((row) => row.model),
        datasets: ["observe_ms", "model_ms", "execute_ms"].map((name, index) => ({
          label: name.replace("_ms", "") + " (mean)",
          data: metrics.latency.map((row) => row[name + "_mean"] / 1000),
          backgroundColor: series[index],
        })),
      },
      options: { maintainAspectRatio: false, scales: { y: { title: { display: true, text: "seconds" }, beginAtZero: true } } },
    });
  }

  for (const key of ["statuses", "verification", "decision_origins", "decision_sources"]) {
    const canvas = document.getElementById("chart-" + key);
    const values = metrics[key] || {};
    if (!canvas || !Object.keys(values).length) continue;
    new Chart(canvas, {
      type: "bar",
      data: { labels: Object.keys(values), datasets: [{ data: Object.values(values), backgroundColor: series[0] }] },
      options: { indexAxis: "y", maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { beginAtZero: true, ticks: { precision: 0 } } } },
    });
  }
})();
