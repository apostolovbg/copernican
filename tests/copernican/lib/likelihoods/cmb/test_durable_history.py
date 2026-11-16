"""Exercise completion history persistence across process boundaries."""

import hashlib
import json
import subprocess  # nosec B404 - fixed local Python test producers.
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy

from copernican.lib.likelihoods.cmb import durable_history, evidence_assets
from copernican.lib.likelihoods.cmb.runtime import cache
from copernican.lib.likelihoods.cmb.runtime.history_store import (
    retain_source_histories,
)


class DurableHistoryTestCase(unittest.TestCase):
    """Reject stale and damaged raw work without borrowing a certificate."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(cache.clear_cmb_caches)
        cache.clear_cmb_caches()
        self.root = Path(self.directory.name)
        self.identity = {"physical": "a" * 64, "numerical": "b" * 64}
        self.key = ("mode", 0.3, (b"weights", 63))
        self.values = {"source": numpy.array([1.0, 2.0, 3.0])}

    def test_bdfcheckpoint_due_respects_elapsed_interval(self):
        """BDFCheckpoint.due uses the configured monotonic time budget."""

        store = durable_history.DurableHistories(
            self.root, self.identity, "cases/first"
        )
        with mock.patch.object(
            durable_history.time, "monotonic", return_value=1
        ):
            checkpoint = durable_history.BDFCheckpoint(
                store, self.key, interval_seconds=3
            )
        self.assertIsInstance(checkpoint, durable_history.BDFCheckpoint)
        with mock.patch.object(
            durable_history.time, "monotonic", return_value=3
        ):
            self.assertFalse(checkpoint.due())
        with mock.patch.object(
            durable_history.time, "monotonic", return_value=4
        ):
            self.assertTrue(checkpoint.due())

    def test_rk_checkpoint_validates_identity_chunks_and_atomic_index(self):
        """Partial histories reject changed identities and damaged chunks."""

        shape = (2, 6, 3)
        histories = numpy.arange(numpy.prod(shape), dtype=float).reshape(shape)
        stats = {
            "rk_stage_count": 4,
            "total_substep_count": 1,
            "maximum_substeps": 1,
            "split_interval_count": 0,
            "split_group_count": 0,
            "completed_work": 2,
        }
        store = durable_history.DurableHistories(
            self.root, self.identity, "cases/first"
        )
        key = ("mode", 0.3, "schedule-a")
        checkpoint = durable_history.RKCheckpoint(
            store, key, interval_seconds=0
        )
        self.assertIsNone(checkpoint.restore(numpy.empty(shape)))
        checkpoint.record(histories, 1, stats)
        old_index = evidence_assets.contained_path(
            self.root, checkpoint.stem + ".rk-index.json"
        ).read_bytes()
        original_write = evidence_assets.write_json_artifact

        def interrupt_index(root, relative, payload):
            if str(relative).endswith(".rk-index.json"):
                raise RuntimeError("interrupted checkpoint index")
            return original_write(root, relative, payload)

        with (
            mock.patch.object(
                evidence_assets,
                "write_json_artifact",
                side_effect=interrupt_index,
            ),
            self.assertRaisesRegex(RuntimeError, "checkpoint index"),
        ):
            checkpoint.record(histories, 3, stats)
        self.assertEqual(
            evidence_assets.contained_path(
                self.root, checkpoint.stem + ".rk-index.json"
            ).read_bytes(),
            old_index,
        )
        restored = durable_history.RKCheckpoint(
            durable_history.DurableHistories(
                self.root, self.identity, "cases/resume"
            ),
            key,
            interval_seconds=0,
        )
        output = numpy.empty(shape)
        self.assertEqual(restored.restore(output), (1, stats))
        numpy.testing.assert_array_equal(output[:, :2], histories[:, :2])
        restored.record(histories, 3, stats)
        output = numpy.empty(shape)
        self.assertEqual(restored.restore(output), (3, stats))
        numpy.testing.assert_array_equal(output[:, :4], histories[:, :4])
        self.assertIsNone(
            durable_history.RKCheckpoint(
                store, ("mode", 0.3, "schedule-b")
            ).restore(numpy.empty(shape))
        )
        changed = {**self.identity, "physical": "c" * 64}
        self.assertIsNone(
            durable_history.RKCheckpoint(
                durable_history.DurableHistories(
                    self.root, changed, "cases/changed"
                ),
                key,
            ).restore(numpy.empty(shape))
        )
        artifact = restored.entries[-1]["artifact"]
        path = evidence_assets.contained_path(self.root, artifact["path"])
        path.write_bytes(path.read_bytes() + b"damaged")
        with self.assertRaises(ValueError):
            restored.restore(numpy.empty(shape))

    def test_interrupted_producer_reloads_in_fresh_process(self):
        """Completed source and independent native histories survive exit."""

        with self.assertRaisesRegex(RuntimeError, "interruption"):
            self.assertIsNone(durable_history.active_durable_histories())
            with durable_history.retain_durable_histories(
                self.root, self.identity, "cases/attempt"
            ) as active:
                self.assertIs(
                    durable_history.active_durable_histories(), active
                )
                with retain_source_histories(max_bytes=0):
                    cache.set_cmb_source_history(self.key, self.values)
                    cache.set_cmb_evolution_history(
                        (self.key, "independent", 64),
                        numpy.array([1.0, 2.0, 3.0]),
                        {"delta": numpy.array([4.0, 5.0, 6.0])},
                    )
                    raise RuntimeError("interruption")
        code = """
import json, sys
from copernican.lib.likelihoods.cmb.durable_history import (
    retain_durable_histories,
)
from copernican.lib.likelihoods.cmb.runtime import cache
from copernican.lib.likelihoods.cmb.runtime.history_store import (
    retain_source_histories,
)
key = ("mode", 0.3, (b"weights", 63))
identity = {"physical": "a" * 64, "numerical": "b" * 64}
with retain_durable_histories(sys.argv[1], identity, "cases/retry") as durable:
    with retain_source_histories(max_bytes=0):
        source = cache.get_cmb_source_history(key)["source"].tolist()
        eta, states = cache.get_cmb_evolution_history((key, "independent", 64))
        missing = cache.get_cmb_evolution_history((key, "independent", 128))
        print(json.dumps([source, eta.tolist(), states["delta"].tolist(),
                          missing, durable.hits]))
"""
        # Fixed interpreter and test-owned script; no shell or user code.
        completed = subprocess.run(  # nosec B603
            [sys.executable, "-c", code, str(self.root)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        values = json.loads(completed.stdout)
        self.assertEqual(values[:3], [[1, 2, 3], [1, 2, 3], [4, 5, 6]])
        self.assertIsNone(values[3])
        self.assertEqual(values[4], 2)

    def test_changed_dependencies_and_exact_keys_do_not_reuse_work(self):
        """Physics, numerical settings and independent tiers stay separate."""

        store = durable_history.DurableHistories(
            self.root, self.identity, "cases/first"
        )
        store.set(self.key, self.values)
        token = store.verified_history_token(self.key)
        self.assertIsNotNone(token)
        self.assertFalse(store.verifies_history_wave(token, 0.3))
        wave_key = ("source", 0.3)
        store.set(wave_key, self.values)
        wave_token = store.verified_history_token(wave_key)
        self.assertTrue(store.verifies_history_wave(wave_token, 0.3))
        self.assertFalse(store.verifies_history_wave(wave_token, 0.4))
        with self.assertRaises(TypeError):
            durable_history._add_key_component(hashlib.sha256(), object())
        self.assertNotEqual(
            durable_history._key_digest(("ab", "c")),
            durable_history._key_digest(("a", "bc")),
        )
        with self.assertRaisesRegex(ValueError, "keys must be strings"):
            durable_history._visit_product(hashlib.sha256(), {1: "invalid"})
        self.assertNotEqual(
            durable_history.product_fingerprint({"grid": numpy.arange(3)}),
            durable_history.product_fingerprint({"grid": numpy.arange(4)}),
        )
        numpy.testing.assert_array_equal(
            store.get(self.key)["source"], [1, 2, 3]
        )
        self.assertIsNone(store.get(("mode", 0.3, (b"weights", 95))))
        for changed in ("physical", "numerical"):
            identity = dict(self.identity)
            identity[changed] = "c" * 64
            self.assertIsNone(
                durable_history.DurableHistories(
                    self.root, identity, "cases/other"
                ).get(self.key)
            )

    def test_interrupted_scalar_split_resumes_in_fresh_process(self):
        """A scalar massive-neutrino trajectory survives process exit."""

        code = """
import hashlib, json, os, sys
from copernican.lib.likelihoods.cmb.durable_history import (
    RKCheckpoint, retain_durable_histories,
)
from copernican.lib.likelihoods.cmb.runtime import execution, mode_evolution
from copernican.lib.likelihoods.cmb.runtime.history_store import (
    retain_source_histories,
)
from tests.copernican.lib.likelihoods.cmb import test_cmb
raw = test_cmb._speedup_contract(
    test_cmb._declared_scalar_hierarchy_contract(
        include_massive_neutrino=True))
raw["numerical"].update(k_sample_count=4, k_max=0.03)
if sys.argv[2] == "schedule":
    raw["numerical"]["evolution_eta_sample_count"] = 128
contract = test_cmb._prepare_declared_contract(raw)
identity = {"physical": "a" * 64, "numerical": "b" * 64}
if sys.argv[2] == "changed":
    identity["physical"] = "c" * 64
mode_evolution._can_batch_declared_evolution = lambda **kwargs: False
if sys.argv[2] == "interrupted":
    def interrupted_checkpoint(store, key, **kwargs):
        checkpoint = RKCheckpoint(store, key, interval_seconds=0, **kwargs)
        original_record = checkpoint.record
        def interrupt(histories, interval, stats):
            original_record(histories, interval, stats)
            if interval == 3:
                os._exit(42)
        checkpoint.record = interrupt
        return checkpoint
    execution.RKCheckpoint = interrupted_checkpoint
with retain_durable_histories(sys.argv[1], identity, "cases/" + sys.argv[2]):
    with retain_source_histories(max_bytes=0):
        result = execution._compute_custom_cmb_spectrum_data_impl(
            contract, (2,), source_probe=(0.001,),
            requested_spectra=("TT",),
            performance_timer=execution.PhaseTimer())
        digests = {name: hashlib.sha256(values.tobytes()).hexdigest()
                   for name, values in result.sources[0.001].items()}
        print(json.dumps({"digests": digests,
            "restored": result.runtime_envelope.get(
                "scalar_checkpoint_restored_intervals", 0),
            "new": result.runtime_envelope["probe_new_modes"]}))
"""
        outcomes = []
        for attempt in (
            "interrupted",
            "resumed",
            "repeat",
            "changed",
            "schedule",
        ):
            # The interpreter and producer are fixed test-owned inputs.
            result = subprocess.run(  # nosec B603
                [sys.executable, "-c", code, str(self.root), attempt],
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(
                result.returncode,
                42 if attempt == "interrupted" else 0,
                result.stderr,
            )
            if attempt != "interrupted":
                outcomes.append(json.loads(result.stdout))
        self.assertEqual(outcomes[0]["restored"], 3)
        self.assertEqual(outcomes[1]["new"], 0)
        self.assertEqual(outcomes[2]["restored"], 0)
        self.assertEqual(outcomes[3]["restored"], 0)
        self.assertGreater(outcomes[3]["new"], 0)
        self.assertEqual(outcomes[0]["digests"], outcomes[1]["digests"])

    def test_interrupted_native_rk_resumes_in_fresh_process(self):
        """A partial physical trajectory survives its producer's exit."""

        code = """
import hashlib, json, os, sys
from copernican.lib.likelihoods.cmb.durable_history import (
    RKCheckpoint, retain_durable_histories,
)
from copernican.lib.likelihoods.cmb.runtime import execution
from copernican.lib.likelihoods.cmb.runtime.history_store import (
    retain_source_histories,
)
from tests.copernican.lib.likelihoods.cmb import test_cmb
raw = test_cmb._speedup_contract(
    test_cmb._declared_scalar_hierarchy_contract(
        include_massive_neutrino=True))
raw["numerical"].update(k_sample_count=4, k_max=0.03)
if sys.argv[2] == "schedule":
    raw["numerical"]["evolution_eta_sample_count"] = 128
contract = test_cmb._prepare_declared_contract(raw)
identity = {"physical": "a" * 64, "numerical": "b" * 64}
if sys.argv[2] == "changed":
    identity["numerical"] = "c" * 64
restored = []
original_restore = RKCheckpoint.restore
def observe_restore(checkpoint, histories):
    result = original_restore(checkpoint, histories)
    if result is not None:
        restored.append(result[0])
    return result
RKCheckpoint.restore = observe_restore
if sys.argv[2] == "interrupted":
    def interrupted_checkpoint(store, key):
        checkpoint = RKCheckpoint(store, key, interval_seconds=0)
        original_record = checkpoint.record
        def interrupt(histories, interval, stats):
            original_record(histories, interval, stats)
            if interval == 3:
                os._exit(42)
        checkpoint.record = interrupt
        return checkpoint
    execution.RKCheckpoint = interrupted_checkpoint
with retain_durable_histories(sys.argv[1], identity, "cases/" + sys.argv[2]):
    with retain_source_histories(max_bytes=0):
        result = execution._compute_custom_cmb_spectrum_data_impl(
            contract, (2,), source_probe=(0.001, 0.03),
            requested_spectra=("TT",),
            performance_timer=execution.PhaseTimer())
        digests = {str(k): {name: hashlib.sha256(values.tobytes()).hexdigest()
                            for name, values in sources.items()}
                   for k, sources in result.sources.items()}
        print(json.dumps({"digests": digests, "restored": restored,
                          "new": result.runtime_envelope["probe_new_modes"]}))
"""
        outcomes = []
        for attempt in (
            "interrupted",
            "resumed",
            "repeat",
            "changed",
            "schedule",
        ):
            # The interpreter and producer are fixed test-owned inputs.
            result = subprocess.run(  # nosec B603
                [sys.executable, "-c", code, str(self.root), attempt],
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(
                result.returncode,
                42 if attempt == "interrupted" else 0,
                result.stderr,
            )
            if attempt != "interrupted":
                outcomes.append(json.loads(result.stdout))
        self.assertIn(3, outcomes[0]["restored"])
        self.assertEqual(outcomes[1]["new"], 0)
        self.assertEqual(outcomes[2]["restored"], [])
        self.assertEqual(outcomes[3]["restored"], [])
        self.assertGreater(outcomes[3]["new"], 0)
        self.assertEqual(outcomes[0]["digests"], outcomes[1]["digests"])

    def test_interrupted_primary_batch_resumes_in_fresh_process(self):
        """Ordinary completed modes survive a producer's abrupt exit."""

        code = """
import hashlib, inspect, json, os, sys
from copernican.lib.likelihoods.cmb.durable_history import (
    retain_durable_histories,
)
from copernican.lib.likelihoods.cmb.runtime import cache, execution
from copernican.lib.likelihoods.cmb.runtime.history_store import (
    retain_source_histories,
)
from tests.copernican.lib.likelihoods.cmb import test_cmb
raw = test_cmb._speedup_contract(
    test_cmb._declared_scalar_hierarchy_contract(
        include_massive_neutrino=True))
raw["numerical"].update(k_sample_count=4, k_max=0.03)
contract = test_cmb._prepare_declared_contract(raw)
identity = {"physical": "a" * 64, "numerical": "b" * 64}
if sys.argv[2] == "changed":
    identity["numerical"] = "c" * 64
if sys.argv[2] == "interrupted":
    original = cache.set_cmb_source_history
    def stop_after_save(key, sources):
        original(key, sources)
        if any(frame.function == "retain_completed_primary_mode"
               for frame in inspect.stack()):
            os._exit(42)
    cache.set_cmb_source_history = stop_after_save
with retain_durable_histories(sys.argv[1], identity, "cases/" + sys.argv[2]):
    with retain_source_histories(max_bytes=0):
        result = execution._compute_custom_cmb_spectrum_data_impl(
            contract, (2,), requested_spectra=("TT",),
            performance_timer=execution.PhaseTimer())
        arrays = {"TT": result.spectra["TT"], **result.transfer_components}
        digests = {name: hashlib.sha256(values.tobytes()).hexdigest()
                   for name, values in arrays.items()}
        print(json.dumps({"digests": digests, "hits": result.runtime_envelope[
            "source_history_cache_hit_count"]}))
"""
        outcomes = []
        for attempt in ("interrupted", "resumed", "repeat", "changed"):
            # The interpreter and producer are fixed test-owned inputs.
            result = subprocess.run(  # nosec B603
                [sys.executable, "-c", code, str(self.root), attempt],
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(
                result.returncode,
                42 if attempt == "interrupted" else 0,
                result.stderr,
            )
            if attempt != "interrupted":
                outcomes.append(json.loads(result.stdout))
        self.assertGreaterEqual(outcomes[0]["hits"], 1)
        self.assertGreater(outcomes[1]["hits"], outcomes[0]["hits"])
        self.assertEqual(outcomes[2]["hits"], 0)
        self.assertEqual(outcomes[0]["digests"], outcomes[1]["digests"])

    def test_interrupted_multi_mode_probe_resumes_in_fresh_process(self):
        """A completed hierarchy mode survives interruption of its peers."""

        code = """
import hashlib, json, os, sys
from copernican.lib.likelihoods.cmb.durable_history import (
    retain_durable_histories,
)
from copernican.lib.likelihoods.cmb.runtime import cache, execution
from copernican.lib.likelihoods.cmb.runtime.history_store import (
    retain_source_histories,
)
from tests.copernican.lib.likelihoods.cmb import test_cmb
raw = test_cmb._speedup_contract(
    test_cmb._declared_scalar_hierarchy_contract(
        include_massive_neutrino=True))
raw["numerical"].update(k_sample_count=4, k_max=0.03)
contract = test_cmb._prepare_declared_contract(raw)
identity = {"physical": "a" * 64, "numerical": "b" * 64}
if sys.argv[2] == "interrupted":
    original = cache.set_cmb_source_history
    def stop_after_save(key, sources):
        original(key, sources)
        os._exit(42)
    cache.set_cmb_source_history = stop_after_save
with retain_durable_histories(sys.argv[1], identity, "cases/" + sys.argv[2]):
    with retain_source_histories(max_bytes=0):
        result = execution._compute_custom_cmb_spectrum_data_impl(
            contract, (2,), source_probe=(0.001, 0.03),
            requested_spectra=("TT",),
            performance_timer=execution.PhaseTimer())
        digests = {str(k): {name: hashlib.sha256(values.tobytes()).hexdigest()
                            for name, values in sources.items()}
                   for k, sources in result.sources.items()}
        print(json.dumps({"digests": digests,
                          "new": result.runtime_envelope["probe_new_modes"],
                          "saved": result.runtime_envelope[
                              "durable_source_modes_checkpointed"]}))
"""
        outcomes = []
        for attempt in ("interrupted", "resumed", "repeat"):
            # The interpreter and producer are fixed test-owned inputs.
            result = subprocess.run(  # nosec B603
                [sys.executable, "-c", code, str(self.root), attempt],
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(
                result.returncode,
                42 if attempt == "interrupted" else 0,
                result.stderr,
            )
            if attempt != "interrupted":
                outcomes.append(json.loads(result.stdout))
        self.assertEqual(outcomes[0]["new"], 1)
        self.assertEqual(outcomes[0]["saved"], 1)
        self.assertEqual(outcomes[1]["new"], 0)
        self.assertEqual(outcomes[1]["saved"], 0)
        self.assertEqual(outcomes[0]["digests"], outcomes[1]["digests"])

    def test_real_declared_sources_resume_in_fresh_process(self):
        """A physical mode is identical after its producer has exited."""

        code = """
import hashlib, json, os, sys
from copernican.lib.likelihoods.cmb.durable_history import (
    retain_durable_histories,
)
from copernican.lib.likelihoods.cmb.durable_history import DurableHistories
from copernican.lib.likelihoods.cmb.runtime import execution
from copernican.lib.likelihoods.cmb.runtime.history_store import (
    retain_source_histories,
)
from tests.copernican.lib.likelihoods.cmb import test_cmb
raw = test_cmb._analytic_signal_contract(decay_rate=float(sys.argv[4]))
raw["numerical"].update(ode_rtol=1e-9, ode_atol=1e-12,
                        evolution_eta_sample_count=int(sys.argv[3]))
contract = test_cmb._prepare_declared_contract(raw)
identity = {"physical": "a" * 64, "numerical": "b" * 64}
waves = (0.001,) if sys.argv[2] == "interrupted" else (0.001, 0.002)
background_signatures = []
original_get = DurableHistories.get
def record_background(self, key):
    if isinstance(key, tuple) and len(key) == 11:
        background_signatures.append(key[8])
    return original_get(self, key)
DurableHistories.get = record_background
with retain_durable_histories(
        sys.argv[1], identity, "cases/" + sys.argv[2]) as durable:
    with retain_source_histories(max_bytes=0):
        result = execution._compute_custom_cmb_spectrum_data_impl(
            contract, (2,), performance_timer=execution.PhaseTimer(),
            requested_spectra=("TT",), source_probe=waves)
        digests = [hashlib.sha256(
                       result.sources[k]["signal_source"].tobytes()
                   ).hexdigest()
                   for k in waves]
        print(json.dumps({"digests": digests,
                          "new": result.runtime_envelope["probe_new_modes"],
                          "hits": durable.hits,
                          "background_hits": durable.product_hits,
                          "background_writes": durable.product_writes,
                          "background": background_signatures[0]}), flush=True)
        if sys.argv[2] == "interrupted":
            os._exit(42)
"""
        outcomes = []
        for attempt, nodes, decay in (
            ("interrupted", 256, "0.0001"),
            ("resumed", 256, "0.0001"),
            ("repeat", 256, "0.0001"),
            ("changed", 128, "0.0001"),
            ("physics", 256, "0.0002"),
        ):
            # Fixed interpreter and test-owned script; no shell or user code.
            result = subprocess.run(  # nosec B603
                [
                    sys.executable,
                    "-c",
                    code,
                    str(self.root),
                    attempt,
                    str(nodes),
                    decay,
                ],
                capture_output=True,
                text=True,
                timeout=45,
            )
            self.assertEqual(
                result.returncode,
                42 if attempt == "interrupted" else 0,
                result.stderr,
            )
            outcomes.append(json.loads(result.stdout))
        self.assertEqual(outcomes[0]["new"], 1)
        self.assertGreater(outcomes[0]["background_writes"], 0)
        self.assertGreater(outcomes[1]["background_hits"], 0)
        self.assertEqual(outcomes[0]["background"], outcomes[1]["background"])
        resumed_hits = int(
            outcomes[0]["background"] == outcomes[1]["background"]
        )
        self.assertEqual(outcomes[1]["hits"], resumed_hits)
        self.assertEqual(outcomes[1]["new"], 2 - resumed_hits)
        if resumed_hits:
            self.assertEqual(
                outcomes[0]["digests"][0], outcomes[1]["digests"][0]
            )
        repeat_hits = (
            2
            if outcomes[2]["background"] == outcomes[1]["background"]
            else int(outcomes[2]["background"] == outcomes[0]["background"])
        )
        self.assertEqual(outcomes[2]["hits"], repeat_hits)
        self.assertEqual(repeat_hits, 2)
        self.assertEqual(outcomes[2]["new"], 2 - repeat_hits)
        if repeat_hits == 2:
            self.assertEqual(outcomes[1]["digests"], outcomes[2]["digests"])
        self.assertEqual(outcomes[3]["new"], 2)
        self.assertEqual(outcomes[3]["hits"], 0)
        self.assertEqual(outcomes[4]["new"], 2)
        self.assertEqual(outcomes[4]["hits"], 0)

    def test_tampering_and_interrupted_replacement_are_rejected(self):
        """An interrupted publication preserves the old complete product."""

        store = durable_history.DurableHistories(
            self.root, self.identity, "cases/first"
        )
        store.set(self.key, self.values)
        original = evidence_assets.write_json_artifact

        def interrupt_index(root, relative, payload):
            if relative.endswith(".index.json"):
                raise RuntimeError("interrupted publication")
            return original(root, relative, payload)

        with (
            mock.patch.object(
                evidence_assets,
                "write_json_artifact",
                side_effect=interrupt_index,
            ),
            self.assertRaisesRegex(RuntimeError, "interrupted publication"),
        ):
            store.set(self.key, {"source": numpy.array([7.0, 8.0, 9.0])})
        numpy.testing.assert_array_equal(
            store.get(self.key)["source"], [1, 2, 3]
        )
        index = evidence_assets.contained_path(
            self.root, store._stem(self.key) + ".index.json"
        )
        pointer = json.loads(index.read_text())
        product = evidence_assets.contained_path(
            self.root, pointer["artifact"]["path"]
        )
        product.write_text("{}")
        with self.assertRaisesRegex(ValueError, "digest"):
            store.get(self.key)

    def test_missing_escaped_and_cross_key_artifacts_are_rejected(self):
        """A valid archive for a different mode is not valid reuse."""

        store = durable_history.DurableHistories(
            self.root, self.identity, "cases/first"
        )
        other_key = ("mode", 0.4, (b"weights", 63))
        store.set(self.key, self.values)
        store.set(other_key, {"source": numpy.array([4.0, 5.0, 6.0])})
        index = self.root / (store._stem(self.key) + ".index.json")
        other_index = self.root / (store._stem(other_key) + ".index.json")
        pointer = json.loads(index.read_text())
        original = dict(pointer)
        pointer["artifact"] = json.loads(other_index.read_text())["artifact"]
        index.write_text(json.dumps(pointer))
        with self.assertRaisesRegex(ValueError, "link differs"):
            store.get(self.key)
        pointer["artifact"] = {"path": "../escaped.json", "sha256": "a"}
        index.write_text(json.dumps(pointer))
        with self.assertRaisesRegex(ValueError, "relative and contained"):
            store.get(self.key)
        index.write_text(json.dumps(original))
        manifest_path = self.root / original["artifact"]["path"]
        binary = json.loads(manifest_path.read_text())["binary"]["path"]
        (self.root / binary).unlink()
        with self.assertRaises(FileNotFoundError):
            store.get(self.key)

    def test_general_products_validate_link_and_preserve_producer(self):
        """An interrupted product replacement cannot borrow another key."""

        store = durable_history.DurableHistories(
            self.root, self.identity, "cases/first"
        )
        key = ("projection", "input", 0)
        other = ("projection", "input", 1)
        self.assertIsNone(store.get_product(key))
        store.set_product(key, {"grid": numpy.arange(3.0)})
        store.set_product(other, {"grid": numpy.arange(4.0)})
        numpy.testing.assert_array_equal(
            store.get_product(key)["grid"], numpy.arange(3.0)
        )
        index = self.root / (store._stem(key) + ".product-index.json")
        other_index = self.root / (store._stem(other) + ".product-index.json")
        pointer = json.loads(index.read_text())
        self.assertEqual(pointer["producer"], "cases/first")
        pointer["artifact"] = json.loads(other_index.read_text())["artifact"]
        index.write_text(json.dumps(pointer))
        with self.assertRaisesRegex(ValueError, "product link differs"):
            store.get_product(key)


if __name__ == "__main__":
    unittest.main()
