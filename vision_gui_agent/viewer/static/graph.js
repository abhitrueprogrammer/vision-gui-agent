// Renders /api/graph with Cytoscape + dagre. Layout, pan/zoom and hit-testing are the library's.
(function () {
  const page = document.querySelector(".graph-page");
  const container = document.getElementById("cy");
  const panel = document.getElementById("node-panel");
  const status = document.getElementById("graph-status");
  const showAll = document.getElementById("show-all");
  const expand = document.getElementById("expand");
  const root = page.dataset.root;
  const run = page.dataset.run;
  let cy = null;

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || undefined;
  }

  function escapeHtml(value) {
    const span = document.createElement("span");
    span.textContent = value == null ? "" : String(value);
    return span.innerHTML;
  }

  function style() {
    const text = css("--pico-color") || "#333";
    const muted = css("--pico-muted-color") || "#888";
    const surface = css("--pico-card-background-color") || "#fff";
    const accent = css("--pico-primary") || "#1095c1";
    return [
      { selector: "node", style: {
        "label": "data(label)", "color": text, "font-size": 12, "text-wrap": "wrap", "text-max-width": 140,
        "text-valign": "center", "background-color": surface, "border-width": 2, "border-color": muted,
        "shape": "round-rectangle", "width": 150, "height": 46 } },
      { selector: "node[?on_path]", style: { "border-color": accent, "border-width": 4 } },
      { selector: "node:selected", style: { "border-color": accent, "border-width": 5, "overlay-opacity": 0.08 } },
      { selector: "edge", style: {
        "curve-style": "bezier", "target-arrow-shape": "triangle", "arrow-scale": 0.9,
        "width": "mapData(count, 1, 20, 2.5, 10)", "opacity": 0.9,
        "line-color": "data(color)", "target-arrow-color": "data(color)",
        "label": "data(edgeLabel)", "font-size": 10, "color": text, "text-rotation": "autorotate",
        "text-margin-y": -9, "text-outline-color": surface, "text-outline-width": 2 } },
      { selector: "edge[?on_path]", style: { "opacity": 1, "line-style": "solid", "z-index": 10, "underlay-color": accent, "underlay-opacity": 0.25, "underlay-padding": 3 } },
      { selector: "edge[?dim]", style: { "opacity": 0.35 } },
    ];
  }

  function edgeColor(edge) {
    if (edge.successes === edge.count) return css("--viewer-ok") || "#2e7d32";
    if (edge.successes === 0) return css("--viewer-fail") || "#c62828";
    return css("--viewer-mixed") || "#b26a00";
  }

  function showNode(node) {
    const data = node.data();
    const outgoing = node.outgoers("edge").map((edge) => {
      const e = edge.data();
      return `<li><span class="outcome ${e.successes ? "ok" : "fail"}">${e.successes}/${e.count}</span> ${escapeHtml(e.summary)} → ${escapeHtml(edge.target().data("label"))}</li>`;
    }).join("");
    panel.innerHTML = `
      <h4>${escapeHtml(data.label)}</h4>
      <p class="muted small">${escapeHtml(data.url)}<br>${data.elements} elements · <code>${escapeHtml(data.id)}</code></p>
      ${data.screenshot ? `<a href="${encodeURI(data.screenshot)}" target="_blank" rel="noopener"><img src="${encodeURI(data.screenshot)}" alt="Screenshot of ${escapeHtml(data.label)}"></a>` : ""}
      <h5>Outgoing actions</h5>
      ${outgoing ? `<ul class="plain">${outgoing}</ul>` : `<p class="muted">none recorded</p>`}`;
  }

  async function load() {
    const params = new URLSearchParams();
    if (run) params.set("run", run);
    if (showAll && showAll.checked) params.set("all", "true");
    if (expand.checked) params.set("expand", "true");
    status.textContent = "Loading…";
    const response = await fetch(`/api/graph/${encodeURIComponent(root)}?${params}`);
    if (!response.ok) {
      status.textContent = response.status === 404 ? "No state graph in this artifact root." : `Failed to load graph (${response.status}).`;
      if (cy) cy.destroy();
      cy = null;
      return;
    }
    const graph = await response.json();
    for (const edge of graph.edges) {
      const e = edge.data;
      e.color = edgeColor(e);
      e.edgeLabel = e.count > 1 ? `${e.summary} ×${e.count}` : e.summary;
      e.dim = Boolean(run) && !e.on_path;
    }
    status.textContent = graph.run_missing && !graph.nodes.length
      ? "This run is not in the state graph (runs with --memory-mode none do not save one). Show whole graph to see the folder's other runs."
      : `${graph.nodes.length}/${graph.total_nodes} states · ${graph.edges.length} edges from ${graph.total_edges} recorded transitions` +
      (graph.filtered ? " · filtered to this run and its neighbours" : "");
    if (cy) cy.destroy();
    cy = cytoscape({
      container, elements: { nodes: graph.nodes, edges: graph.edges }, style: style(),
      layout: { name: "dagre", rankDir: "LR", nodeSep: 40, rankSep: 150, edgeSep: 20 },
      wheelSensitivity: 0.3, minZoom: 0.1, maxZoom: 3,
    });
    cy.on("tap", "node", (event) => showNode(event.target));
  }

  document.getElementById("fit").addEventListener("click", () => cy && cy.fit(undefined, 30));
  expand.addEventListener("change", load);
  if (showAll) showAll.addEventListener("change", load);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => cy && cy.style(style()));
  load();
})();
