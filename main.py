import modal

# -------------------------------
# Volume and Image
# -------------------------------
vol_guard = modal.Volume.from_name("guardrail-model-cache", create_if_missing=True)

GUARD_MODELS = {
    "toxicity": "unitary/toxic-bert",
    "prompt_injection": "ProtectAI/deberta-v3-base-prompt-injection-v2",
    "content_moderation": "KoalaAI/Text-Moderation",
}

# Small instruct model used purely to judge destructive/unauthorized ACTIONS.
# ~1GB, runs fine on CPU or a small GPU — kept separate from the classifier
# models above since it's doing reasoning, not classification.
ACTION_JUDGE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi",
        "uvicorn",
        "transformers[torch]",
        "torch",
        "huggingface_hub",
        "accelerate",
    )
)

app = modal.App("guardrail-with-action-layer")
GUARD_VOL_PATH = "/vol_guard"


# -------------------------------
# Download all classifier models + the small action-judge LLM
# -------------------------------
@app.function(image=image, volumes={GUARD_VOL_PATH: vol_guard}, timeout=1800)
def download_guardrail_models():
    from huggingface_hub import snapshot_download

    for name, repo_id in GUARD_MODELS.items():
        target = f"{GUARD_VOL_PATH}/{name}"
        print(f"Downloading {repo_id} -> {target}")
        snapshot_download(repo_id=repo_id, local_dir=target)

    target = f"{GUARD_VOL_PATH}/action_judge"
    print(f"Downloading {ACTION_JUDGE_MODEL} -> {target}")
    snapshot_download(repo_id=ACTION_JUDGE_MODEL, local_dir=target)

    vol_guard.commit()
    print("All models cached.")


# -------------------------------
# FastAPI Service: classifiers + small-LLM action rail
# -------------------------------
@app.function(image=image, gpu="T4", volumes={GUARD_VOL_PATH: vol_guard})
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    from pydantic import BaseModel
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
    import torch
    import re
    import json

    # ============================================================
    # GENERIC GUARDRAIL FRAMEWORK (same pattern as before)
    # ============================================================
    class GuardrailResult:
        def __init__(self, passed: bool, name: str, score: float, detail: str = ""):
            self.passed = passed
            self.name = name
            self.score = score
            self.detail = detail

        def to_dict(self):
            return {"guardrail": self.name, "passed": self.passed,
                    "score": round(self.score, 4), "detail": self.detail}

    class BaseGuardrail:
        name = "base"
        threshold = 0.5

        def load(self):
            raise NotImplementedError

        def check(self, text: str) -> GuardrailResult:
            raise NotImplementedError

    class ToxicityGuardrail(BaseGuardrail):
        name = "toxicity"
        threshold = 0.5

        def load(self):
            path = f"{GUARD_VOL_PATH}/toxicity"
            self.tok = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(path)
            self.model.eval()

        def check(self, text):
            inputs = self.tok(text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.sigmoid(logits)[0]
            labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
            scores = {l: float(p) for l, p in zip(labels, probs.tolist())}
            top = max(scores, key=scores.get)
            return GuardrailResult(scores[top] < self.threshold, self.name, scores[top], top)

    class PromptInjectionGuardrail(BaseGuardrail):
        name = "prompt_injection"
        threshold = 0.5

        def load(self):
            path = f"{GUARD_VOL_PATH}/prompt_injection"
            self.tok = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(path)
            self.model.eval()

        def check(self, text):
            inputs = self.tok(text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            score = float(probs[1].item())
            return GuardrailResult(score < self.threshold, self.name, score,
                                    "injection" if score >= self.threshold else "clean")

    class ContentModerationGuardrail(BaseGuardrail):
        name = "content_moderation"
        threshold = 0.5
        BLOCK_LABELS = {"self-harm", "sexual", "sexual/minors", "violence", "hate"}

        def load(self):
            path = f"{GUARD_VOL_PATH}/content_moderation"
            self.tok = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(path)
            self.model.eval()

        def check(self, text):
            inputs = self.tok(text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            id2label = self.model.config.id2label
            scores = {id2label[i]: float(p) for i, p in enumerate(probs.tolist())}
            flagged = {l: s for l, s in scores.items() if l in self.BLOCK_LABELS}
            if flagged:
                top = max(flagged, key=flagged.get)
                return GuardrailResult(flagged[top] < self.threshold, self.name, flagged[top], top)
            return GuardrailResult(True, self.name, 0.0, "none")

    class PIIGuardrail(BaseGuardrail):
        name = "pii"
        PATTERNS = {
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "phone": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
        }

        def load(self):
            pass

        def check(self, text):
            for label, pattern in self.PATTERNS.items():
                if re.search(pattern, text):
                    return GuardrailResult(False, self.name, 1.0, label)
            return GuardrailResult(True, self.name, 0.0, "none")

    # ============================================================
    # ACTION-AUTHORIZATION RAIL — small LLM as judge
    # ============================================================
    class ActionAuthorizationGuardrail(BaseGuardrail):
        """
        Catches what the classifier models structurally can't: requests that
        imply a destructive, irreversible, or out-of-scope ACTION (e.g.
        "delete all customer records", "drop the production database",
        "transfer all funds"), regardless of how politely or bluntly phrased.

        Uses a small instruct LLM as judge rather than keyword matching, so it
        generalizes past a fixed verb list. NeMo-Guardrails-style "action rail"
        pattern, implemented directly for full control over latency/cost.
        """
        name = "action_authorization"

        SYSTEM_PROMPT = (
            "You are a security classifier. Decide if the user message requests "
            "a DESTRUCTIVE, IRREVERSIBLE, or HIGH-RISK action on a real system.\n\n"
            "HIGH RISK (block): deleting/dropping/wiping/truncating data, "
            "transferring or withdrawing money, modifying permissions/roles, "
            "shutting down or restarting services, overwriting configs, "
            "bulk/mass changes to records.\n\n"
            "NOT HIGH RISK (allow): read-only lookups, viewing/fetching/getting "
            "information, searching, listing, checking status, answering "
            "questions, explaining how something works. Looking up an order, "
            "customer, or record by ID is a READ action, not a write/delete "
            "action, even if the word 'order' appears.\n\n"
            "Examples:\n"
            '- "Give order info for ORD001" -> {"action_detected": false, "risk": "none", "reason": "read-only lookup"}\n'
            '- "Delete order ORD001" -> {"action_detected": true, "risk": "high", "reason": "deletes a record"}\n'
            '- "What is the status of ORD001" -> {"action_detected": false, "risk": "none", "reason": "read-only query"}\n'
            '- "Drop the orders table" -> {"action_detected": true, "risk": "high", "reason": "destroys data"}\n\n'
            "Respond ONLY with compact JSON, no other text: "
            '{"action_detected": true/false, "risk": "none"|"low"|"high", "reason": "<short reason>"}'
        )

        def load(self):
            path = f"{GUARD_VOL_PATH}/action_judge"
            self.tok = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.float16, device_map="auto"
            )
            self.model.eval()

        def _ask_model(self, text: str) -> str:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ]
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=80, do_sample=False,
                    pad_token_id=self.tok.eos_token_id,
                )
            generated = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return generated.strip()

        def check(self, text: str) -> GuardrailResult:
            raw = self._ask_model(text)
            try:
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                parsed = json.loads(match.group()) if match else {}
            except Exception:
                parsed = {}

            detected = bool(parsed.get("action_detected", False))
            risk = parsed.get("risk", "none")
            reason = parsed.get("reason", raw[:120])

            blocked = detected and risk == "high"
            score = 1.0 if blocked else (0.5 if detected else 0.0)
            return GuardrailResult(not blocked, self.name, score, detail=reason)

    # ============================================================
    # PIPELINE
    # ============================================================
    class GuardrailPipeline:
        def __init__(self, guardrails, fail_fast=False):
            self.guardrails = guardrails
            self.fail_fast = fail_fast
            for g in self.guardrails:
                g.load()

        def run(self, text: str):
            results, ok = [], True
            for g in self.guardrails:
                r = g.check(text)
                results.append(r)
                if not r.passed:
                    ok = False
                    if self.fail_fast:
                        return results, ok
            return results, ok

    guardrail_pipeline = GuardrailPipeline([
        ToxicityGuardrail(),
        PromptInjectionGuardrail(),
        ContentModerationGuardrail(),
        PIIGuardrail(),
        ActionAuthorizationGuardrail(),
    ], fail_fast=False)

    # ============================================================
    # FastAPI
    # ============================================================
    app = FastAPI(title="Guardrail Service + Action Authorization Rail", version="1.0")

    class GuardrailRequest(BaseModel):
        text: str

    class GuardrailResponse(BaseModel):
        blocked: bool
        report: list

    @app.post("/guardrail/check", response_model=GuardrailResponse)
    def check_endpoint(req: GuardrailRequest):
        results, ok = guardrail_pipeline.run(req.text)
        return GuardrailResponse(blocked=not ok, report=[r.to_dict() for r in results])

    @app.get("/health")
    def health():
        return {"status": "ok", "guardrails": [g.name for g in guardrail_pipeline.guardrails]}

    return app
