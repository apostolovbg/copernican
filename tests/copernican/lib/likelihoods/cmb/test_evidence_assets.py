"""Integrity and selective-reuse tests, without scientific solver runs."""

import subprocess  # nosec B404 - reviewed fixed local Python test producer.
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy

from copernican.lib.likelihoods.cmb import evidence_assets as products


class CompletionProductsTestCase(unittest.TestCase):
    """Synthetic arrays test storage; they never certify physical parity."""

    def test_selective_readback_materializes_only_requested_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            entry = products.write_array_products(
                directory,
                "selected",
                {
                    "grid": numpy.arange(3.0),
                    "histories": {"x": numpy.ones((3, 10))},
                },
            )
            loaded = products.read_array_products(
                directory, entry, array_paths=(("grid",),)
            )
            numpy.testing.assert_array_equal(loaded["grid"], numpy.arange(3.0))
            self.assertIsInstance(
                loaded["histories"]["x"], products.RetainedArrayInfo
            )
            with self.assertRaisesRegex(ValueError, "missing"):
                products.read_array_products(
                    directory, entry, array_paths=(("absent",),)
                )

    def test_array_payloads_round_trip_without_dtype_or_shape_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = numpy.arange(12, dtype=numpy.longdouble).reshape(3, 4)
            entry = products.write_array_products(
                directory, "case/native", {"grids": [expected], "name": "x"}
            )
            loaded = products.read_array_products(
                directory, entry, materialize=True
            )
            numpy.testing.assert_array_equal(loaded["grids"][0], expected)
            self.assertEqual(
                loaded["name"], "x", "decode retains scalar values"
            )
            self.assertEqual(
                loaded["grids"][0].dtype,
                expected.dtype,
                "encode must preserve extended precision",
            )
            self.assertFalse(loaded["grids"][0].flags.writeable)
            metadata = products.read_array_products(directory, entry)
            self.assertIsInstance(
                metadata["grids"][0], products.RetainedArrayInfo
            )
            self.assertEqual(metadata["grids"][0].shape, (3, 4))
            self.assertEqual(metadata["grids"][0].dtype, expected.dtype.str)
            self.assertEqual(metadata["name"], "x")

    def test_selective_readback_rejects_malformed_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            entry = products.write_array_products(
                directory, "case", {"x": numpy.ones(3)}
            )
            manifest = products.read_json_artifact(directory, entry)
            for malformed in (None, ["array_0"], "array_0"):
                manifest["products"]["items"]["x"] = malformed
                forged = products.write_json_artifact(
                    directory, "forged.json", manifest
                )
                with self.assertRaisesRegex(ValueError, "not an array"):
                    products.read_array_products(
                        directory, forged, array_paths=(("x",),)
                    )

    def test_producer_exit_and_cleanup_leave_reloadable_actual_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with tempfile.TemporaryDirectory() as producer_directory:
                source = Path(producer_directory) / "producer.py"
                source.write_text(
                    "from copernican.lib.likelihoods.cmb "
                    "import evidence_assets as p\n"
                    "import numpy,sys\n"
                    "p.write_array_products(sys.argv[1], 'retained', "
                    "{'sources':numpy.arange(5.)})\n"
                )
                # Literal script owned by this test; no shell or user code.
                subprocess.run(  # nosec B603
                    [
                        sys.executable,
                        "-c",
                        "import runpy,sys; path=sys.argv.pop(1); "
                        "runpy.run_path(path,run_name='__main__')",
                        str(source),
                        str(root),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            manifest = root / "retained.json"
            entry = {
                "path": manifest.name,
                "sha256": products._file_sha256(manifest),
            }
            loaded = products.read_array_products(
                root, entry, materialize=True
            )
            numpy.testing.assert_array_equal(
                loaded["sources"], numpy.arange(5)
            )

    def test_tampered_or_missing_binary_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            entry = products.write_array_products(
                directory, "case", {"x": numpy.arange(4)}
            )
            manifest = products.read_json_artifact(directory, entry)
            self.assertEqual(
                products.read_json_artifact(directory, entry)[
                    "schema_version"
                ],
                1,
            )
            binary = Path(directory) / manifest["binary"]["path"]
            binary.write_bytes(binary.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "binary digest"):
                products.read_array_products(directory, entry)
            binary.unlink()
            with self.assertRaises(FileNotFoundError):
                products.read_array_products(directory, entry)

    def test_interrupted_manifest_publish_preserves_the_previous_array_pair(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            entry = products.write_array_products(
                directory, "case", {"array": numpy.arange(4)}
            )
            with mock.patch.object(
                products,
                "write_json_artifact",
                side_effect=OSError("interrupted"),
            ):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    products.write_array_products(
                        directory, "case", {"array": numpy.arange(6)}
                    )
            loaded = products.read_array_products(
                directory, entry, materialize=True
            )
            numpy.testing.assert_array_equal(loaded["array"], numpy.arange(4))

    def test_rehashed_metadata_cannot_forge_shapes_or_array_links(self):
        with tempfile.TemporaryDirectory() as directory:
            entry = products.write_array_products(
                directory, "case", {"x": numpy.arange(4)}
            )
            manifest = products.read_json_artifact(directory, entry)
            manifest["arrays"]["array_0"]["shape"] = [2, 2]
            forged = products.write_json_artifact(
                directory, "forged.json", manifest
            )
            with self.assertRaisesRegex(ValueError, "array differs"):
                products.read_array_products(directory, forged)
            manifest["arrays"]["array_0"]["shape"] = [4]
            manifest["products"]["items"]["x"] = {
                "kind": "array",
                "key": "absent",
            }
            forged = products.write_json_artifact(
                directory, "forged.json", manifest
            )
            with self.assertRaisesRegex(ValueError, "missing array"):
                products.read_array_products(directory, forged)

    def test_paths_and_nonfinite_or_object_arrays_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in (
                "../escape",
                "..\\escape",
                "C:\\escape",
                str(Path(directory).resolve()),
            ):
                with self.assertRaises(ValueError):
                    products.write_json_artifact(directory, name, {})
            for array in (numpy.array([numpy.nan]), numpy.array([object()])):
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    products.write_array_products(
                        directory, "bad", {"x": array}
                    )
            with self.assertRaisesRegex(ValueError, "keys must be strings"):
                products.write_array_products(
                    directory, "bad", {1: [2], "1": [3]}
                )
            with self.assertRaises(ValueError):
                products.write_json_artifact(
                    directory, "nan.json", {"x": float("nan")}
                )
            self.assertFalse((Path(directory) / "nan.json").exists())

    def test_contained_path_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inside").mkdir()
            (root / "inside/link").symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "escapes"):
                products.contained_path(root / "inside", "link/outside")

    def test_atomic_failure_preserves_preceding_complete_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retained"
            path.write_bytes(b"complete")

            def interrupted(stream):
                stream.write(b"partial")
                raise OSError("manufactured interrupted producer")

            with self.assertRaisesRegex(OSError, "interrupted producer"):
                products._atomic_write(path, interrupted)
            self.assertEqual(path.read_bytes(), b"complete")
            self.assertEqual(list(path.parent.glob(".pending-*")), [])


if __name__ == "__main__":
    unittest.main()
