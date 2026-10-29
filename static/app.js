const documentForm = document.querySelector("#document-form");
const questionForm = document.querySelector("#question-form");
const saveStatus = document.querySelector("#save-status");
const answerBox = document.querySelector("#answer");

function setBusy(form, busy) {
  const button = form.querySelector("button[type=submit]");
  button.disabled = busy;
  button.dataset.label ||= button.textContent;
  button.textContent = busy ? "Working…" : button.dataset.label;
}

async function readResponse(response) {
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || "The request could not be completed.");
  return body;
}

documentForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy(documentForm, true);
  saveStatus.textContent = "Saving notes…";
  saveStatus.classList.remove("error");
  try {
    const payload = Object.fromEntries(new FormData(documentForm));
    const result = await readResponse(await fetch("/documents", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
    }));
    document.querySelector("#filter-user").value = result.user_name;
    saveStatus.textContent = `${result.chunks_added} passage${result.chunks_added === 1 ? "" : "s"} saved to “${result.user_name}”.`;
  } catch (error) {
    saveStatus.textContent = error.message;
    saveStatus.classList.add("error");
  } finally { setBusy(documentForm, false); }
});

questionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy(questionForm, true);
  answerBox.classList.remove("hidden");
  answerBox.innerHTML = "<p>Searching your library…</p>";
  try {
    const question = document.querySelector("#question").value.trim();
    const user = document.querySelector("#filter-user").value.trim();
    const params = new URLSearchParams({ question });
    if (user) params.set("user", user);
    const result = await readResponse(await fetch(`/ask?${params}`));
    answerBox.replaceChildren();
    const heading = document.createElement("h3"); heading.textContent = "Answer";
    const paragraph = document.createElement("p"); paragraph.textContent = result.answer;
    answerBox.append(heading, paragraph);
    if (result.context_used?.length) {
      const sourceHeading = document.createElement("h3"); sourceHeading.textContent = "Retrieved passages";
      answerBox.append(sourceHeading);
      result.context_used.forEach((text) => {
        const source = document.createElement("div"); source.className = "source"; source.textContent = text;
        answerBox.append(source);
      });
    }
  } catch (error) {
    answerBox.innerHTML = "";
    const message = document.createElement("p"); message.className = "error"; message.textContent = error.message;
    answerBox.append(message);
  } finally { setBusy(questionForm, false); }
});
