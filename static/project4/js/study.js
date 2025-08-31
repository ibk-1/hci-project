const URL_NEXT   = window.URL_NEXT;
const URL_SUBMIT = window.URL_SUBMIT;
const CSRF_TOKEN = window.CSRF_TOKEN;      // comes from the template

// how many ratings we want before finishing
const TARGET = 10;                         // must match RATING_TARGET in views.py
let answered = 0;

async function loadNext() {
  const res  = await fetch(URL_NEXT);
  const data = await res.json();

  if (data.done) {
    window.location = "/project4/results/";
    return;
  }

  // ---- put movie title ----
  document.getElementById("title-text").textContent = data.title;
  document.getElementById("card").dataset.id = data.id;

  // ---- influence bars ----
// ---- influence bars ----
const container = document.getElementById("influence");
container.innerHTML = "";                      // wipe old rows

data.influence.forEach(item => {
  // row wrapper
  const row = document.createElement("div");
  row.className = "inf-row";

  // title
  const title = document.createElement("span");
  title.className = "inf-title";
  title.textContent = item.title;
  row.appendChild(title);

  // signed delta
  const deltaTxt = document.createElement("span");
  deltaTxt.className = "inf-delta";
  deltaTxt.textContent = (item.delta > 0 ? '+' : '') + item.delta.toFixed(2);
  row.appendChild(deltaTxt);

  // bar
  const progOuter = document.createElement("div");
  progOuter.className = "progress inf-prog";
  const progBar  = document.createElement("div");
  progBar.className = "progress-bar" +
                      (item.delta >= 0 ? " bg-success" : " bg-danger");
  progBar.style.width = `${Math.min(Math.abs(item.delta) * 40, 100)}%`;
  progOuter.appendChild(progBar);
  row.appendChild(progOuter);

  container.appendChild(row);
});

}


document.querySelectorAll("#buttons button").forEach(btn => {
  btn.addEventListener("click", async e => {
    const mid    = document.getElementById("card").dataset.id;
    const rating = e.target.dataset.rating;

    // POST the answer
    await fetch(URL_SUBMIT, {
      method:  "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken":  CSRF_TOKEN,
      },
      body: JSON.stringify({ movie_id: mid, rating }),
    });

    // update progress badge
    answered += 1;
    document.getElementById("progress").textContent = `${answered} / ${TARGET}`;

    loadNext();
  });
});

loadNext();   // kick things off
