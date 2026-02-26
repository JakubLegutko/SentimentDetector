chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.type === "UPDATE_BADGE") {
        const { score, label, active } = request.data;

        if (!active) {
            chrome.action.setBadgeText({ text: "", tabId: sender.tab.id });
            return;
        }

        let color = "#888888"; // Gray default
        let text = "MIX";

        if (label === "Objective" && score > 33) {
            color = "#10b981"; // Green
            text = "OBJ";
        } else if (label === "Subjective" && score < -33) {
            color = "#ef4444"; // Red
            text = "SUB";
        } else {
            color = "#f59e0b"; // Yellow (Mixed)
            text = "MIX";
        }

        chrome.action.setBadgeBackgroundColor({ color: color, tabId: sender.tab.id });
        chrome.action.setBadgeText({ text: text, tabId: sender.tab.id });
    } else if (request.type === "FETCH_API") {
        fetch(`http://127.0.0.1:8000${request.endpoint}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(request.payload)
        })
            .then(async (res) => {
                let data = null;
                try { data = await res.json(); } catch (e) { }
                sendResponse({ ok: res.ok, status: res.status, data: data });
            })
            .catch(err => {
                sendResponse({ error: err.message || err.toString() });
            });
        return true; // Keep message channel open for async response
    }
});

chrome.action.onClicked.addListener((tab) => {
    if (tab.id) {
        chrome.tabs.sendMessage(tab.id, { type: "TOGGLE_PANEL" }).catch((e) => {
            console.warn("Could not send TOGGLE_PANEL to tab:", e);
        });
    }
});
