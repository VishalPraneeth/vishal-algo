const heatmap = document.getElementById("heatmap");
const exchangeSelect = document.getElementById("exchange");

exchangeSelect.addEventListener("change", loadData);

async function loadData() {
  heatmap.innerHTML = "<span class='loading loading-spinner'></span>";

  const exchange = 'NSE';
  const res = await fetch(`/api/v1/sector-heatmap?exchange=${exchange}`);
  const json = await res.json();

  heatmap.innerHTML = "";

  Object.entries(json.data).forEach(([sector, info]) => {
    heatmap.innerHTML += `
      <div class="card ${color(info.strength)} text-white shadow-xl">
        <div class="card-body">
          <h2 class="card-title">${sector}</h2>
          <p>Strength: ${info.strength}</p>
          <p>Avg % Change: ${info.avg_change}%</p>
        </div>
      </div>
    `;
  });
}

function color(strength) {
  if (strength.includes("STRONG_BUY")) return "bg-green-600";
  if (strength === "BUY") return "bg-green-400";
  if (strength === "SELL") return "bg-red-400";
  if (strength.includes("STRONG_SELL")) return "bg-red-600";
  return "bg-gray-400";
}

loadData();
setInterval(loadData, 300000); // refresh every 5 min
