"""The evaluation harness: ordering, resume, snapshots, statistics and Gemini rate limits."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from vision_gui_agent.evaluation import (PHASE, MemorySnapshot, SERVICE_FAILURE, finished_keys, read_rows, run_statistics,
                                         sequence_jobs, wilson)
from vision_gui_agent.gemini import GeminiClientPool, RateLimiter
from vision_gui_agent.logging_store import RunLogger
from vision_gui_agent.models import ActionDecision, EvidenceRecord, Observation


class OrderingTests(unittest.TestCase):
    def test_sequence_order_and_experiment_phase(self):
        jobs = sequence_jobs("active-action-model", 1)
        apps = [job.app for job in jobs]
        self.assertEqual(apps.index("helpdesk") > apps.index("workspace"), True)
        self.assertEqual(jobs[-1].app, "orders")
        workspace = [job for job in jobs if job.app == "workspace"]
        self.assertEqual([job.group for job in workspace[:5]], ["exploration"] * 4 + [PHASE])
        self.assertFalse(any(job.task == PHASE for job in sequence_jobs("passive-action-model", 1)))
        # The unseen app runs its fixed order, with its unavailable tasks after it.
        orders = [job.task for job in jobs if job.app == "orders"]
        self.assertEqual(orders[:2], ["C1_draft", "C2_discount"])
        self.assertTrue(orders[-1].startswith("C_u_"))

    def test_two_exploration_orders_across_seeds(self):
        first = [job.task for job in sequence_jobs("none", 1) if job.group == "exploration" and job.app == "workspace"]
        second = [job.task for job in sequence_jobs("none", 2) if job.group == "exploration" and job.app == "workspace"]
        self.assertEqual(first, list(reversed(second)))


class ResumeTests(unittest.TestCase):
    def test_service_failures_are_requeued_twice_then_left(self):
        row = {"mode": "none", "seed": 1, "app": "workspace", "task": "A_ho_report", "layout": "classic"}
        key = tuple(row.values())
        self.assertNotIn(key, finished_keys([{**row, "service_failure": True}] * 2))
        self.assertIn(key, finished_keys([{**row, "service_failure": True}] * 3))
        self.assertIn(key, finished_keys([{**row, "passed": False}]))

    def test_rows_from_all_shards_are_read(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "results-shard0of2.jsonl").write_text(json.dumps({"a": 1}) + "\n")
            (out / "results-shard1of2.jsonl").write_text(json.dumps({"a": 2}) + "\n\n")
            self.assertEqual(read_rows(out), [{"a": 1}, {"a": 2}])

    def test_snapshot_restores_memory_after_a_cut_short_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "graph.json").write_text("before")
            snapshot = MemorySnapshot(root)
            snapshot.take()
            (root / "graph.json").write_text("half learned")
            (root / "action-model-v2.json").write_text("created by the failed run")
            snapshot.restore()
            self.assertEqual((root / "graph.json").read_text(), "before")
            self.assertFalse((root / "action-model-v2.json").exists())

    def test_a_run_killed_before_its_row_is_undone_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "graph.json").write_text("before")
            MemorySnapshot(root).take()
            (root / "graph.json").write_text("half learned by a killed run")
            MemorySnapshot(root).take()  # the resumed process starts the same job again
            self.assertEqual((root / "graph.json").read_text(), "before")
            settled = MemorySnapshot(root); settled.take()
            (root / "graph.json").write_text("learned by a finished run"); settled.settle()
            MemorySnapshot(root).take()
            self.assertEqual((root / "graph.json").read_text(), "learned by a finished run")

    def test_service_failure_classification(self):
        for message in ("429 RESOURCE_EXHAUSTED", "TimeoutError: ", "Gemini quota exhausted on every key", "503 UNAVAILABLE"):
            self.assertTrue(SERVICE_FAILURE.search(message), message)
        self.assertFalse(SERVICE_FAILURE.search("Step limit reached"))


class StatisticsTests(unittest.TestCase):
    def test_wilson_interval(self):
        low, high = wilson(8, 10)
        self.assertAlmostEqual(low, .49, places=2); self.assertAlmostEqual(high, .943, places=2)
        self.assertIsNone(wilson(0, 0))

    def test_calls_origins_and_attempts_on_the_target_come_from_the_log(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "runs.sqlite3"
            logger = RunLogger(database)
            logger.start_run("run", "goal", "model")
            observation = Observation("", "", [], "", "")
            target = ActionDecision("click", 1, grounding=(EvidenceRecord("element_text", "Generate report", 1),))
            logger.log("run", 0, None, None, target, False, observation, {}, "refused", decision_origin="gemini", gemini_calls={"policy": 1, "refine": 1})
            logger.log("run", 1, None, None, target, True, observation, {}, decision_origin="gemini", gemini_calls={"policy": 1})
            logger.log("run", 2, None, None, ActionDecision("done"), True, observation, {}, decision_origin="learned_unavailable")
            logger.close()
            stats = run_statistics(database, "run", "generate report")
        self.assertEqual(stats["gemini_calls"], {"policy": 2, "refine": 1})
        self.assertEqual(stats["target_attempts"], 2)  # the refused click counts too
        self.assertEqual(stats["origins"], {"gemini": 2, "learned_unavailable": 1})


class RateLimitTests(unittest.TestCase):
    def test_requests_and_tokens_per_minute_are_shared(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rate.sqlite3"
            clock = [1000.0]
            sleeps = []
            def sleep(seconds): sleeps.append(seconds); clock[0] += seconds
            first, second = RateLimiter(path, rpm=2, tpm=10_000), RateLimiter(path, rpm=2, tpm=10_000)
            first.acquire(now=lambda: clock[0], sleep=sleep)
            second.acquire(now=lambda: clock[0], sleep=sleep)
            self.assertEqual(sleeps, [])
            first.acquire(now=lambda: clock[0], sleep=sleep)  # third request in a minute must wait
            self.assertGreaterEqual(sum(sleeps), 59)
            tokens = RateLimiter(Path(directory) / "tokens.sqlite3", rpm=100, tpm=3000)
            sleeps.clear()
            row = tokens.acquire(now=lambda: clock[0], sleep=sleep)
            tokens.record(row, "other", 2900)  # actual usage corrects the estimate
            tokens.acquire(now=lambda: clock[0], sleep=sleep)
            self.assertGreaterEqual(sum(sleeps), 59)

    def test_concurrent_processes_never_exceed_the_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rate.sqlite3"
            granted = []
            def worker():
                limiter = RateLimiter(path, rpm=5, tpm=10**9)
                for _ in range(3):
                    try: limiter.acquire(sleep=lambda _: (_ for _ in ()).throw(TimeoutError()))
                    except TimeoutError: return
                    granted.append(1)
            threads = [threading.Thread(target=worker) for _ in range(4)]
            for thread in threads: thread.start()
            for thread in threads: thread.join()
            self.assertEqual(len(granted), 5)

    def test_quota_storm_backs_off_then_fails_as_a_service_failure(self):
        from vision_gui_agent import gemini
        self.addCleanup(gemini._KEY_COOLDOWN.clear)
        pool = object.__new__(GeminiClientPool)
        pool._keys, pool._index, pool.client = ["one", "two"], 0, object()
        pool._new_client = lambda: object()
        pauses = []
        def exhausted(_client): raise RuntimeError("429 RESOURCE_EXHAUSTED")
        with self.assertRaises(RuntimeError) as raised:
            pool.generate(exhausted, sleep=pauses.append)
        self.assertTrue(SERVICE_FAILURE.search(str(raised.exception)))
        self.assertGreater(len(pauses), 3)
        self.assertLessEqual(sum(pauses), GeminiClientPool.BACKOFF_LIMIT_S * 1.5 + 60)

    def test_calls_rotate_over_keys_and_skip_one_out_of_quota(self):
        from vision_gui_agent import gemini
        self.addCleanup(gemini._KEY_COOLDOWN.clear)
        pool = object.__new__(GeminiClientPool)
        pool._keys, pool._index, pool.client = ["k1", "k2", "k3"], 0, "k1"
        pool._new_client = lambda: pool._keys[pool._index]
        used = [pool.generate(lambda client: client) for _ in range(6)]
        self.assertEqual(sorted(used), ["k1", "k1", "k2", "k2", "k3", "k3"])
        def k2_daily_quota(client):
            if client == "k2": raise RuntimeError("429 RESOURCE_EXHAUSTED. Quota exceeded for metric: free_tier_requests. Please retry in 12h45m46.3s.")
            return client
        used = [pool.generate(k2_daily_quota) for _ in range(6)]
        self.assertNotIn("k2", used)
        self.assertGreater(gemini._KEY_COOLDOWN["k2"] - time.monotonic(), 12 * 3600)

    def test_every_key_out_of_daily_quota_fails_without_waiting(self):
        from vision_gui_agent import gemini
        self.addCleanup(gemini._KEY_COOLDOWN.clear)
        pool = object.__new__(GeminiClientPool)
        pool._keys, pool._index, pool.client = ["d1", "d2"], 0, object()
        pool._new_client = lambda: object()
        pauses = []
        def exhausted(_client): raise RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 3h2m.")
        with self.assertRaises(RuntimeError):
            pool.generate(exhausted, sleep=pauses.append)
        self.assertEqual(pauses, [])

    def test_retry_delay_is_read_from_either_error_form(self):
        from vision_gui_agent.gemini import _retry_after
        self.assertAlmostEqual(_retry_after(RuntimeError("Please retry in 12h45m46.5s.")), 45946.5)
        self.assertEqual(_retry_after(RuntimeError("{'retryDelay': '37s'}")), 37)
        self.assertEqual(_retry_after(RuntimeError("429 RESOURCE_EXHAUSTED")), 30)

    def test_a_temporary_overload_is_waited_out(self):
        pool = object.__new__(GeminiClientPool)
        pool._keys, pool._index, pool.client = ["one"], 0, object()
        calls, pauses = [], []
        def flaky(_client):
            calls.append(1)
            if len(calls) < 3: raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")
            return SimpleNamespace(text="ok")
        self.assertEqual(pool.generate(flaky, sleep=pauses.append).text, "ok")
        self.assertEqual((len(calls), len(pauses)), (3, 2))

    def test_usage_and_model_version_are_recorded(self):
        from vision_gui_agent.gemini import LAST_MODEL_VERSION
        pool = object.__new__(GeminiClientPool)
        pool._keys, pool._index, pool.client = ["one"], 0, object()
        response = SimpleNamespace(text="ok", model_version="gemini-3.5-flash-lite-001", usage_metadata=SimpleNamespace(total_token_count=1234))
        self.assertIs(pool.generate(lambda _client: response), response)
        self.assertEqual(LAST_MODEL_VERSION["value"], "gemini-3.5-flash-lite-001")


if __name__ == "__main__":
    unittest.main()
