(() => {
  const form = document.querySelector("#chat-form");
  const input = document.querySelector("#message-input");
  const sendButton = document.querySelector("#send-button");
  const messages = document.querySelector("#messages");
  const status = document.querySelector("#status");
  const modelContainer = document.querySelector("#model-container");
  const modelFallback = document.querySelector("#model-fallback");
  const expressionBadge = document.querySelector("#expression-badge");
  const companionState = document.querySelector("#companion-state");
  const csrfToken = document.querySelector('meta[name="csrf-token"]').content;
  const live2dCanvas = document.querySelector("#live2d-canvas");
  const expressionLabels = {
    neutral: "平靜",
    happy: "開心",
    angry: "生氣",
    surprised: "驚訝",
    sad: "低落",
    disgust: "嫌棄",
  };
  let live2dApp;

  function setExpression(expression) {
    const mood = expressionLabels[expression] ? expression : "neutral";
    expressionBadge.textContent = expressionLabels[mood];
    companionState.textContent = {
      neutral: "在這裡陪你聊聊",
      happy: "看來她心情不錯",
      angry: "她好像有點不服氣",
      surprised: "她似乎有些意外",
      sad: "她看起來有點低落",
      disgust: "她露出嫌棄的表情",
    }[mood];
  }

  function speakJapanese(text) {
    if (!text) {
      status.textContent = "這則回覆沒有可朗讀的日語內容。";
      return;
    }
    if (!("speechSynthesis" in window) || typeof SpeechSynthesisUtterance === "undefined") {
      status.textContent = "這個瀏覽器不支援語音朗讀，請改用 Safari 或 Chrome。";
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "ja-JP";
    utterance.rate = 0.96;
    utterance.pitch = 1.04;
    const japaneseVoice = window.speechSynthesis
      .getVoices()
      .find((voice) => voice.lang.toLowerCase().startsWith("ja"));
    if (japaneseVoice) {
      utterance.voice = japaneseVoice;
    }
    utterance.onstart = () => {
      status.textContent = "正在用手機的日語語音朗讀…";
    };
    utterance.onend = () => {
      status.textContent = "";
    };
    utterance.onerror = (event) => {
      status.textContent = event.error === "canceled"
        ? ""
        : "手機語音播放失敗；請確認媒體音量，或在手機設定中安裝日語語音。";
    };
    window.speechSynthesis.speak(utterance);
  }

  function addMessage(text, role, sources = [], japaneseReply = "") {
    const article = document.createElement("article");
    article.className = `message ${role}-message`;

    if (role === "assistant") {
      const heading = document.createElement("div");
      heading.className = "message-heading";

      const avatar = document.createElement("div");
      avatar.className = "message-avatar";
      avatar.setAttribute("aria-hidden", "true");
      avatar.textContent = "A";

      const label = document.createElement("div");
      label.className = "message-label";
      label.textContent = "Amadeus";
      heading.append(avatar, label);
      article.append(heading);
    } else {
      const label = document.createElement("div");
      label.className = "message-label";
      label.textContent = "你";
      article.append(label);
    }

    const content = document.createElement("p");
    content.textContent = text;
    article.append(content);

    if (role === "assistant" && /[\u3040-\u30ff]/.test(japaneseReply)) {
      const listenButton = document.createElement("button");
      listenButton.className = "listen-button";
      listenButton.type = "button";
      listenButton.textContent = "♫ 聽日語語音";
      listenButton.setAttribute("aria-label", "播放日語語音回覆");
      listenButton.addEventListener("click", () => speakJapanese(japaneseReply));
      article.append(listenButton);
    }

    if (Array.isArray(sources) && sources.length) {
      const sourceList = document.createElement("div");
      sourceList.className = "sources";
      for (const source of sources) {
        let url;
        try {
          url = new URL(source.url);
        } catch {
          continue;
        }
        if (!["http:", "https:"].includes(url.protocol)) {
          continue;
        }
        const link = document.createElement("a");
        link.href = url.href;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = source.title || source.url;
        sourceList.append(link);
      }
      if (sourceList.childElementCount) {
        article.append(sourceList);
      }
    }

    messages.append(article);
    article.scrollIntoView({ behavior: "smooth", block: "end" });
  }

  async function loadLive2dModel() {
    try {
      if (!window.L2D?.init) {
        throw new Error("Live2D runtime did not load");
      }

      live2dApp = window.L2D.init(live2dCanvas);
      live2dApp.on("loaded", () => {
        modelFallback.hidden = true;
      });
      await live2dApp.load({
        path: "/models/kurisu/kurisu.model.json",
        scale: 0.9,
        logLevel: "warn",
      });
      modelFallback.hidden = true;
      window.addEventListener("resize", () => live2dApp.resize());
    } catch (error) {
      console.error("Live2D model failed to load:", error);
      modelFallback.textContent = "Live2D 載入失敗，請確認網路、瀏覽器與模型資源。";
      modelFallback.classList.add("model-fallback-error");
    }
  }

  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (!message || sendButton.disabled) {
      return;
    }

    addMessage(message, "user");
    companionState.textContent = "正在認真想怎麼回答你…";
    input.value = "";
    input.style.height = "auto";
    sendButton.disabled = true;
    status.textContent = "Amadeus 正在思考…";

    try {
      const response = await fetch("/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRF-Token": csrfToken,
        },
        body: JSON.stringify({ user_id: "user_001", message }),
      });
      const data = await response.json();

      if (response.status === 401) {
        window.location.assign("/login");
        return;
      }
      if (!response.ok && response.status !== 429) {
        throw new Error(data.reply || data.error || `HTTP ${response.status}`);
      }

      const expression = data.expression || "neutral";
      addMessage(
        data.reply || "沒有收到回覆，請稍後再試。",
        "assistant",
        data.sources,
        data.japanese_reply || "",
      );
      setExpression(expression);
      if (data.error === "quota_exceeded") {
        status.textContent = "Gemini 額度不足，請稍後再試。";
      } else if (data.error) {
        status.textContent = "這次沒有成功取得 Gemini 回覆，請稍後再試。";
      } else {
        status.textContent = "";
      }
    } catch (error) {
      status.textContent = `連線失敗：${error.message}`;
      companionState.textContent = "暫時連不上電腦上的 Amadeus";
    } finally {
      sendButton.disabled = false;
      input.focus();
    }
  });

  loadLive2dModel();
})();
