import json
import os
import threading
import time
import openai

class MemoryManager:
    def __init__(self, path, autosave_interval=10, cap_messages=200, summary_threshold=150):
        self.path = path
        self.lock = threading.RLock()
        self.data = {}  # agent_id -> list of {"role":..., "content":..., "ts":...}
        self.cap_messages = cap_messages
        self.summary_threshold = summary_threshold
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._load()
        self._stop = False
        self._autosave_interval = autosave_interval
        self._autosave_thread = threading.Thread(target=self._autosave_loop, daemon=True)
        self._autosave_thread.start()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def save(self):
        with self.lock:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.data, f, indent=2)
            os.replace(tmp, self.path)

    def _autosave_loop(self):
        while not self._stop:
            time.sleep(self._autosave_interval)
            try:
                self.save()
            except Exception:
                pass

    def stop(self):
        self._stop = True
        self._autosave_thread.join(timeout=2)
        try:
            self.save()
        except Exception:
            pass

    def get_history(self, agent_id):
        return self.data.get(str(agent_id), [])

    def append(self, agent_id, role, content, ts=None):
        with self.lock:
            aid = str(agent_id)
            if aid not in self.data:
                self.data[aid] = []
            self.data[aid].append({"role": role, "content": content, "ts": ts})
            # enforce cap: keep newest messages but preserve one summary slot
            if len(self.data[aid]) > self.cap_messages:
                # preserve last cap_messages, move older to a compacted summary placeholder
                overflow = self.data[aid][:-self.cap_messages]
                self.data[aid] = self.data[aid][-self.cap_messages:]
                # attach compacted summary marker
                self.data[aid].insert(0, {"role": "system", "content": "[previous conversation compacted]"})

    def summarize_if_needed(self, agent_id, summarizer_prompt=None, max_tokens=512):
        # call LLM to summarise when large; safe no-op if openai not configured
        hist = self.get_history(agent_id)
        if len(hist) < self.summary_threshold:
            return None
        try:
            # build a short summary prompt
            messages = [{"role":"system","content":"Summarize the conversation into a 3-4 sentence context that can be used as system context."}]
            # include last N messages to summarize
            tail = hist[-self.summary_threshold:]
            for m in tail:
                messages.append({"role": m["role"], "content": m["content"]})
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini", # replace with your preferred model
                messages=messages,
                max_tokens= max_tokens,
                temperature=0.2,
            )
            summary = resp["choices"][0]["message"]["content"].strip()
            # compact: replace oldest messages with summary marker + summary
            with self.lock:
                aid = str(agent_id)
                # remove the tail we summarized and insert summary at front
                self.data[aid] = [{"role":"system", "content":"[COMPACTED SUMMARY] " + summary}] + self.data[aid][-self.cap_messages:]
            self.save()
            return summary
        except Exception:
            return None