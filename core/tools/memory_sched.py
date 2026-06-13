"""Memory, scheduling, and model-introspection tools."""

from __future__ import annotations


def register(r) -> None:

    # ── Long-term memory ─────────────────────────────────────────

    @r.register("remember", "Save a durable fact about the user or their setup",
                {"fact": "string: the fact, self-contained",
                 "?topic": "string: short category"})
    def remember(ctx, fact: str, topic: str = "") -> str:
        if not fact.strip():
            return "Nothing to remember."
        fid = ctx.memory.remember(fact, topic)
        return f"Remembered (#{fid}): {fact}"

    @r.register("recall_memory", "Search saved facts",
                {"query": "string: what to look for"})
    def recall_memory(ctx, query: str) -> str:
        hits = ctx.memory.recall(query, limit=10)
        return ("Relevant facts:\n" + "\n".join(f"- {h}" for h in hits)
                if hits else "No saved facts match.")

    @r.register("forget", "Delete a saved fact by its id",
                {"fact_id": "integer: id from the facts list"})
    def forget(ctx, fact_id: int) -> str:
        ok = ctx.memory.forget(int(fact_id))
        return f"Forgot fact #{fact_id}." if ok else f"No fact #{fact_id}."

    # ── Scheduled tasks ──────────────────────────────────────────

    @r.register("schedule_task",
                "Schedule a prompt to run later: 'in 10 minutes', 'at 18:30', 'every 2 hours'",
                {"when": "string: time expression",
                 "prompt": "string: what JARVIS should do when it fires"})
    def schedule_task(ctx, when: str, prompt: str) -> str:
        return ctx.scheduler.schedule(when, prompt)

    @r.register("list_tasks", "List scheduled tasks", {})
    def list_tasks(ctx) -> str:
        return ctx.scheduler.describe()

    @r.register("cancel_task", "Cancel a scheduled task by id",
                {"task_id": "integer: id from list_tasks"})
    def cancel_task(ctx, task_id: int) -> str:
        return ctx.scheduler.cancel(int(task_id))

    # ── Models ───────────────────────────────────────────────────

    @r.register("list_models", "List locally available AI models and routing", {})
    def list_models(ctx) -> str:
        return ctx.models.summary() if ctx.models else "Model manager unavailable."
