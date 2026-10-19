# 2026-10-16: Regression checks for VASP/Mongo transfer without contacting a live database.
"""Run from the repository root: python -m unittest utilities.test_vasp_mongo"""
import copy
import csv
# 2026-10-17: Exercise attachment compression and byte-exact retrieval offline.
import gzip
import hashlib
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bson import BSON, ObjectId
from pymatgen.io.vasp.inputs import Poscar
# 2026-10-17: Model Mongo duplicate-key rejection for collision-protected imports.
from pymongo.errors import DuplicateKeyError

from utilities import vasp_mongo as vm

POSCAR = """LiCl
1
4 0 0
0 4 0
0 0 4
Li Cl
1 1
Selective dynamics
Direct
0 0 0 T F T
0.5 0.5 0.5 F F F
"""


class MemoryCollection:
    """Observe Mongo calls and require every written document to encode as BSON."""
    def __init__(self, documents=None):
        self.documents = copy.deepcopy(documents or [])
        self.last_filter = None

    # 2026-10-17: Enforce unique IDs in the offline collection, matching insert_one.
    def insert_one(self, document):
        """Insert a BSON-safe record or reject an existing ID without modifying it."""
        if any(existing["_id"] == document["_id"] for existing in self.documents):
            raise DuplicateKeyError("duplicate _id")
        BSON.encode(document)
        self.documents.append(copy.deepcopy(document))
        return SimpleNamespace(inserted_id=document["_id"])

    # 2026-10-17: Support the same attachment projections and dotted updates as Mongo.
    def find_one(self, query, projection=None):
        """Return a detached record by ID for attachment inspection."""
        return next((copy.deepcopy(doc) for doc in self.documents
                     if doc["_id"] == query["_id"]), None)

    def update_one(self, query, update, upsert=False):
        identifier = query["_id"]
        for document in self.documents:
            if document["_id"] == identifier:
                for key, value in update["$set"].items():
                    if key.startswith("files."):
                        document.setdefault("files", {})[key.split(".", 1)[1]] = copy.deepcopy(value)
                    else:
                        document[key] = copy.deepcopy(value)
                BSON.encode(document)
                return SimpleNamespace(upserted_id=None, matched_count=1)
        assert upsert
        document = {"_id": identifier, **copy.deepcopy(update["$set"]),
                    **copy.deepcopy(update["$setOnInsert"])}
        BSON.encode(document)
        self.documents.append(document)
        return SimpleNamespace(upserted_id=identifier)

    def find(self, filters, projection):
        self.last_filter = filters
        # Selection of element species is exercised by the actual export function.
        return iter(copy.deepcopy(self.documents))


# 2026-10-17: Use a GridFS-shaped store to verify file contents without touching a server.
class MemoryFileStore:
    """Track blobs, metadata, and deletions with the GridFS methods used by the utility."""
    def __init__(self):
        self.blobs = {}
        self.metadata = {}
        self.deleted = []

    def put(self, source, **metadata):
        """Store all streamed bytes and return an independent file identifier."""
        identifier = ObjectId()
        self.blobs[identifier] = source.read()
        self.metadata[identifier] = metadata
        return identifier

    def get(self, identifier):
        """Open an independent binary stream for an archived file."""
        return io.BytesIO(self.blobs[identifier])

    def exists(self, identifier):
        """Report whether an attachment is still available."""
        return identifier in self.blobs

    def delete(self, identifier):
        """Remove a retired blob and record the deletion."""
        self.blobs.pop(identifier, None)
        self.metadata.pop(identifier, None)
        self.deleted.append(identifier)


class VaspMongoTests(unittest.TestCase):
    """Check parsers, idempotent imports, element selection, and export round trips."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ipse-vasp-mongo-")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.output = io.StringIO()

    def folder(self, name="job", poscar=POSCAR):
        folder = self.root / name
        folder.mkdir(parents=True)
        (folder / "CONTCAR").write_text(poscar)
        return folder

    def test_last_energy_and_conventions(self):
        outcar = self.root / "OUTCAR"
        outcar.write_text(
            "free  energy   TOTEN  = -1.0 eV\n"
            "energy  without entropy= -1.1 energy(sigma->0) = -1.2\n"
            "free  energy   TOTEN  = -2.5D+01 eV\n"
            "energy  without entropy= -2.6D+01 energy(sigma->0) = -2.7D+01\n")
        self.assertEqual(vm.read_last_energy(outcar, "F"), -25.)
        self.assertEqual(vm.read_last_energy(outcar, "E0"), -27.)
        oszicar = self.root / "OSZICAR"
        oszicar.write_text("DAV: 1 -999\n1 F= -9. E0= -8.\n2 F= +0.000E+00 E0= -.5d+01\n")
        self.assertEqual(vm.read_last_energy(oszicar, "F"), 0.)
        self.assertEqual(vm.read_last_energy(oszicar, "E0"), -5.)

    def test_sources_fallback_and_final_structure(self):
        folder = self.folder()
        (folder / "OUTCAR").write_text("no energy\n")
        (folder / "OSZICAR").write_text("1 F= -12. E0= -11.\n")
        result = vm.read_vasp_result(folder)
        self.assertEqual(result["energy_file"], "OSZICAR")
        self.assertEqual(result["E_total"], -12.)
        self.assertEqual(result["E_per_atom"], -6.)
        self.assertTrue(result["structure_is_final"])
        BSON.encode(result)  # Selective-dynamics arrays must be converted to BSON-safe lists.
        (folder / "POSCAR").write_text(POSCAR)
        (folder / "CONTCAR").write_text("")
        result = vm.read_vasp_result(folder)
        self.assertEqual(result["structure_file"], "POSCAR")
        self.assertFalse(result["structure_is_final"])
        with self.assertRaises(ValueError):
            vm.read_vasp_result(folder, require_normal_end=True)
        (folder / "OUTCAR").write_text("free energy TOTEN = -13.\nVoluntary context switches: 1\n")
        self.assertTrue(vm.read_vasp_result(folder, require_normal_end=True)["outcar_normal_end"])

    def test_recursive_folder_depth(self):
        root_job = self.root
        (root_job / "OUTCAR").write_text("free energy TOTEN = -1.\n")
        child = self.folder("child")
        nested = self.folder("child/nested")
        self.assertEqual(vm.find_job_dirs(self.root, 0), [self.root])
        self.assertEqual(vm.find_job_dirs(self.root, 1), sorted([self.root, child]))
        self.assertEqual(vm.find_job_dirs(self.root, None), sorted([self.root, child, nested]))

    # 2026-10-17: Reimport must refuse collisions until updates are explicitly enabled.
    def test_import_updates_same_folder_but_keeps_polymorphs(self):
        first = self.folder("phase1")
        second = self.folder("phase2")
        for folder in [first, second]:
            (folder / "OSZICAR").write_text("1 F= -12. E0= -11.\n")
        broken = self.folder("missing_energy")
        collection = MemoryCollection()
        with redirect_stdout(self.output):
            counts = vm.import_to_mongo(collection, self.root)
        self.assertEqual(counts, {"inserted": 2, "updated": 0, "filtered": 0, "failed": 1})
        self.assertEqual(len(collection.documents), 2)
        created = {doc["_id"]: doc["created_at"] for doc in collection.documents}
        (first / "OSZICAR").write_text("1 F= -14. E0= -13.\n")
        before = copy.deepcopy(collection.documents)
        with redirect_stdout(self.output):
            refused = vm.import_to_mongo(collection, self.root)
        self.assertEqual(refused["updated"], 0)
        self.assertEqual(refused["failed"], 3)
        self.assertEqual(collection.documents, before)
        self.assertIn("update_existing=True", self.output.getvalue())
        with redirect_stdout(self.output):
            counts = vm.import_to_mongo(collection, self.root, update_existing=True)
        self.assertEqual(counts["inserted"], 0)
        self.assertEqual(counts["updated"], 2)
        self.assertEqual(len(collection.documents), 2)
        updated = next(doc for doc in collection.documents if doc["source_folder"] == str(first))
        self.assertEqual(updated["E_total"], -14.)
        for document in collection.documents:
            self.assertEqual(document["created_at"], created[document["_id"]])

    def test_element_selection_uses_species(self):
        licl = Poscar.from_str(POSCAR).structure
        nacl = Poscar.from_str(POSCAR.replace("Li", "Na")).structure
        iron = Poscar.from_str(POSCAR.replace("Li", "Fe")).structure
        self.assertTrue(vm.matches_elements(licl, ["Li", "Cl", "Na"], "subset"))
        self.assertFalse(vm.matches_elements(licl, ["Li", "Cl", "Na"], "exact"))
        self.assertTrue(vm.matches_elements(licl, ["Cl"], "contains_all"))
        self.assertFalse(vm.matches_elements(licl, ["Li"], "subset"))
        self.assertFalse(vm.matches_elements(licl, ["Li", "Cl"], excluded=["Li"]))
        self.assertFalse(vm.matches_elements(nacl, ["N", "Cl"], "subset"))
        self.assertFalse(vm.matches_elements(iron, ["F", "Cl"], "subset"))

    def test_export_round_trip_and_overwrite_protection(self):
        folder = self.folder()
        (folder / "OSZICAR").write_text("1 F= 0. E0= -1.\n")
        document = vm.read_vasp_result(folder)
        document["_id"] = ObjectId()
        other = copy.deepcopy(document)
        other["_id"] = ObjectId()
        other["struct"] = Poscar.from_str(POSCAR.replace("Li", "Na")).structure.as_dict()
        invalid = {"_id": ObjectId(), "struct": document["struct"], "E_total": float("nan")}
        collection = MemoryCollection([document, other, invalid])
        destination = self.root / "export"
        query = {"code": "VASP"}
        with redirect_stdout(self.output):
            counts = vm.export_from_mongo(collection, destination, elements=["Li", "Cl"], filters=query)
        self.assertEqual(counts, {"exported": 1, "filtered": 1, "failed": 1})
        self.assertEqual(collection.last_filter, query)
        with (destination / "energies.csv").open() as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0], ["label", "E_total"])
        label, energy = rows[1]
        self.assertEqual(float(energy), 0.)
        # 2026-10-18: Stored formulas prefix labels consistently in filenames and CSV rows.
        self.assertEqual(f"{document['formula']}_{document['_id']}", label)
        restored = Poscar.from_file(destination / ("POSCAR_" + label))
        self.assertEqual(restored.structure.composition, Poscar.from_str(POSCAR).structure.composition)
        # 2026-10-16: Pymatgen exposes selective dynamics as a list of array-like rows.
        flags = [[bool(value) for value in row] for row in restored.selective_dynamics]
        self.assertEqual(flags, [[True, False, True], [False, False, False]])
        previous = {path.name: path.read_bytes() for path in destination.iterdir()}
        with redirect_stdout(self.output), self.assertRaises(FileExistsError):
            vm.export_from_mongo(collection, destination, elements=["Li", "Cl"])
        self.assertEqual(previous, {path.name: path.read_bytes() for path in destination.iterdir()})
        with redirect_stdout(self.output):
            vm.export_from_mongo(collection, destination, elements=["Li", "Cl"], overwrite=True)

    # 2026-10-17: Export uses the literal Mongo ID with no added formula or digest.
    def test_nested_fields_legacy_structures_and_safe_labels(self):
        structure = Poscar.from_str(POSCAR).structure
        collection = MemoryCollection([{"_id": "mp-861724", "properties": {"energy": -4.},
                                      "data": {"poscar": POSCAR}}])
        destination = self.root / "legacy"
        with redirect_stdout(self.output):
            counts = vm.export_from_mongo(collection, destination,
                energy_field="properties.energy", structure_field="data.poscar")
        self.assertEqual(counts["exported"], 1)
        label = vm.make_label("mp-861724")
        self.assertEqual(label, "mp-861724")
        with (destination / "energies.csv").open() as handle:
            self.assertEqual(list(csv.reader(handle))[1], ["mp-861724", "-4"])
        self.assertTrue((destination / ("POSCAR_" + label)).is_file())
        self.assertEqual(vm.get_field({"energy.with.dot": 3}, "energy.with.dot"), 3)

    # 2026-10-17: Moving a project preserves IDs; folder depth and prefix are import-only.
    def test_folder_ids_are_portable_and_configurable(self):
        tail = "spin_discarded/Cu2S6Zn2_nomadIy7js_spin"
        expected = "spin_discarded__Cu2S6Zn2_nomadIy7js_spin"
        self.assertEqual(vm.make_import_id("/hpc/results/" + tail), expected)
        self.assertEqual(vm.make_import_id("/local/archive/" + tail), expected)
        self.assertEqual(vm.make_import_id("/local/archive/" + tail, 1), tail.split("/")[-1])
        self.assertEqual(vm.make_import_id("/hpc/results/" + tail, 2, "sulfides"),
                         "sulfides__" + expected)
        for depth in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                vm.make_import_id("/a/b", depth)
        with self.assertRaises(ValueError):
            vm.make_import_id("/a/b", id_prefix="bad/prefix")

    # 2026-10-17: Different leading paths with the same trailing folders must not overwrite.
    def test_colliding_projects_do_not_overwrite(self):
        first = self.folder("project1/spin/job")
        second = self.folder("project2/spin/job")
        for folder, energy in ((first, -10), (second, -20)):
            (folder / "OSZICAR").write_text(f"1 F= {energy}. E0= -1.\n")
        collection = MemoryCollection()
        with redirect_stdout(self.output):
            counts = vm.import_to_mongo(collection, self.root, max_depth=None)
        self.assertEqual(counts["inserted"], 1)
        self.assertEqual(counts["failed"], 1)
        self.assertEqual(collection.documents[0]["_id"], "spin__job")
        self.assertEqual(collection.documents[0]["E_total"], -10.)
        with redirect_stdout(self.output):
            vm.import_to_mongo(collection, second, max_depth=0, id_prefix="project2")
        self.assertEqual({doc["_id"] for doc in collection.documents},
                         {"spin__job", "project2__spin__job"})

    # 2026-10-17: Unsafe IDs are reported rather than silently renamed or used as paths.
    def test_export_rejects_unsafe_ids(self):
        for identifier in ("folder/job", "folder\\job", "bad\nname", "", ".."):
            with self.assertRaises(ValueError):
                vm.make_label(identifier)
        destination = self.root / "unsafe_export"
        collection = MemoryCollection([{"_id": "../outside", "struct": POSCAR, "E_total": -1}])
        with redirect_stdout(self.output):
            counts = vm.export_from_mongo(collection, destination)
        self.assertEqual(counts["failed"], 1)
        self.assertEqual(list(destination.glob("POSCAR_*")), [])
        self.assertEqual(vm.make_label(123), "123")

    # 2026-10-17: Verify all formats, per-file switches, reuse, replacement, and export bytes.
    def test_attachment_archival_and_export(self):
        folder = self.folder("spin/job")
        originals = {"INCAR": b"ENCUT = 520\n", "OSZICAR": b"1 F= -12. E0= -11.\n",
                     "OUTCAR": b"free energy TOTEN = -12.\n" * 50000}
        for name, content in originals.items():
            (folder / name).write_bytes(content)
        collection, store = MemoryCollection(), MemoryFileStore()
        options = dict(max_depth=0, file_store=store, upload_incar=True,
                       upload_oszicar=True, upload_outcar=True)
        with redirect_stdout(self.output):
            vm.import_to_mongo(collection, folder, **options)
        document = collection.documents[0]
        self.assertEqual(set(document["files"]), set(originals))
        for name, reference in document["files"].items():
            content = store.blobs[reference["gridfs_id"]]
            decoded = content if name == "INCAR" else gzip.decompress(content)
            self.assertEqual(decoded, originals[name])
            self.assertEqual(reference["sha256"], hashlib.sha256(decoded).hexdigest())
            self.assertEqual(reference["original_size"], len(decoded))
            self.assertEqual(reference["stored_size"], len(content))
        before = copy.deepcopy(document["files"])
        with redirect_stdout(self.output):
            vm.import_to_mongo(collection, folder, update_existing=True, **options)
        self.assertEqual(document["files"], before)
        self.assertEqual(len(store.blobs), 3)
        (folder / "INCAR").write_bytes(b"ENCUT = 600\n")
        originals["INCAR"] = b"ENCUT = 600\n"
        with redirect_stdout(self.output):
            vm.import_to_mongo(collection, folder, update_existing=True, **options)
        self.assertIn(before["INCAR"]["gridfs_id"], store.deleted)
        self.assertEqual(len(store.blobs), 3)
        destination = self.root / "attachments_export"
        with redirect_stdout(self.output):
            vm.export_from_mongo(collection, destination, file_store=store, export_original_files=True)
        for name, original in originals.items():
            filename = name if name == "INCAR" else name + ".gz"
            content = (destination / "spin__job" / filename).read_bytes()
            self.assertEqual(content if name == "INCAR" else gzip.decompress(content), original)
        # Attachment conflicts are checked before replacing the energy table or POSCAR.
        (destination / "energies.csv").unlink()
        (destination / f"POSCAR_{document['formula']}_spin__job").unlink()
        with self.assertRaises(FileExistsError), redirect_stdout(self.output):
            vm.export_from_mongo(collection, destination, file_store=store, export_original_files=True)
        self.assertFalse((destination / "energies.csv").exists())

    # 2026-10-17: Missing/disabled files and rejected duplicates never discard archived data.
    def test_attachment_selection_missing_and_duplicate(self):
        folder = self.folder("spin/job")
        (folder / "OSZICAR").write_text("1 F= -12. E0= -11.\n")
        (folder / "OUTCAR").write_text("free energy TOTEN = -12.\n")
        collection, store = MemoryCollection(), MemoryFileStore()
        with redirect_stdout(self.output):
            vm.import_to_mongo(collection, folder, max_depth=0, file_store=store,
                               upload_incar=True, upload_oszicar=True)
        document = collection.documents[0]
        self.assertEqual(set(document["files"]), {"OSZICAR"})
        before = copy.deepcopy(document["files"])
        (folder / "OSZICAR").unlink()
        with redirect_stdout(self.output):
            vm.import_to_mongo(collection, folder, max_depth=0, file_store=store,
                               upload_oszicar=True, update_existing=True)
            vm.import_to_mongo(collection, folder, max_depth=0, file_store=store,
                               upload_outcar=True)
        self.assertEqual(document["files"], before)
        self.assertEqual(len(store.blobs), 1)

    # 2026-10-17: A damaged archive never becomes a final exported attachment.
    def test_attachment_checksum_and_upload_errors(self):
        folder = self.folder("spin/job")
        (folder / "OSZICAR").write_text("1 F= -12. E0= -11.\n")
        collection, store = MemoryCollection(), MemoryFileStore()
        with patch.object(store, "put", side_effect=OSError("disk full")):
            with redirect_stdout(self.output):
                counts = vm.import_to_mongo(collection, folder, max_depth=0, file_store=store,
                                            upload_oszicar=True)
        self.assertEqual(counts["inserted"], 1)
        self.assertEqual(collection.documents[0]["E_total"], -12.)
        with redirect_stdout(self.output):
            vm.import_to_mongo(collection, folder, max_depth=0, file_store=store,
                               upload_oszicar=True, update_existing=True)
        reference = collection.documents[0]["files"]["OSZICAR"]
        store.blobs[reference["gridfs_id"]] = b"corrupted"
        destination = self.root / "checksum" / "OSZICAR.gz"
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            vm.export_attachment(store, reference, destination)
        self.assertFalse(destination.exists())
        self.assertEqual(list(destination.parent.iterdir()), [])

    # 2026-10-17: Attachment database failures do not count a stored calculation as failed.
    def test_attachment_lookup_failure_keeps_import_counts(self):
        folder = self.folder("spin/job")
        (folder / "OSZICAR").write_text("1 F= -12. E0= -11.\n")
        collection, store = MemoryCollection(), MemoryFileStore()
        with patch.object(collection, "find_one", side_effect=vm.PyMongoError("connection lost")):
            with redirect_stdout(self.output):
                counts = vm.import_to_mongo(collection, folder, max_depth=0, file_store=store,
                                            upload_oszicar=True)
        self.assertEqual(counts["inserted"], 1)
        self.assertEqual(counts["failed"], 0)
        self.assertIn("WARNING attachments", self.output.getvalue())

    # 2026-10-18: Check both independent switches with missing, null, and invalid counterparts.
    def test_partial_export_option_matrix(self):
        documents = [
            {"_id": "complete", "struct": POSCAR, "E_total": -2.},
            {"_id": "energy", "E_total": 0.},
            {"_id": "bad_structure", "struct": "invalid", "E_total": -3.},
            {"_id": "structure", "struct": POSCAR, "E_total": None},
            {"_id": "bad_energy", "struct": POSCAR, "E_total": float("nan")},
            {"_id": "neither", "struct": None},
        ]
        for energies_only in (False, True):
            for structures_only in (False, True):
                with self.subTest(energy=energies_only, structure=structures_only):
                    destination = self.root / f"partial_{energies_only}_{structures_only}"
                    with redirect_stdout(self.output):
                        counts = vm.export_from_mongo(MemoryCollection(documents), destination,
                            export_energy_without_structure=energies_only,
                            export_structure_without_energy=structures_only)
                    expected = 1 + 2 * energies_only + 2 * structures_only
                    self.assertEqual(counts["exported"], expected)
                    self.assertEqual(counts["failed"], len(documents) - expected)
                    with (destination / "energies.csv").open() as handle:
                        rows = list(csv.reader(handle))
                    energy_labels = {row[0] for row in rows[1:]}
                    expected_energies = {"complete"}
                    if energies_only:
                        expected_energies.update({"energy", "bad_structure"})
                    self.assertEqual(energy_labels, expected_energies)
                    expected_poscars = {"POSCAR_complete"}
                    if structures_only:
                        expected_poscars.update({"POSCAR_structure", "POSCAR_bad_energy"})
                    self.assertEqual({path.name for path in destination.glob("POSCAR_*")},
                                     expected_poscars)
                    if energies_only:
                        self.assertIn(["energy", "0"], rows)
        self.assertIn("PARTIAL Mongo record", self.output.getvalue())

    # 2026-10-18: Unknown species must never silently bypass energy-only element filters.
    def test_energy_only_export_element_filters(self):
        documents = [
            {"_id": "licl", "elements": ["Li", "Cl"], "E_total": -1.},
            {"_id": "nacl", "elements": ["Na", "Cl"], "E_total": -2.},
            {"_id": "unknown", "E_total": -3.},
            {"_id": "invalid", "elements": ["NotAnElement"], "E_total": -4.},
        ]
        for selection, elements, excluded in (("subset", ["Li", "Cl"], []),
                                               ("exact", ["Li", "Cl"], []),
                                               ("contains_all", ["Li"], []),
                                               ("subset", [], ["Na"])):
            destination = self.root / f"filter_{selection}_{len(elements)}"
            with redirect_stdout(self.output):
                counts = vm.export_from_mongo(MemoryCollection(documents), destination,
                    export_energy_without_structure=True, elements=elements,
                    selection=selection, excluded=excluded)
            self.assertEqual(counts, {"exported": 1, "filtered": 1, "failed": 2})
            with (destination / "energies.csv").open() as handle:
                self.assertEqual(list(csv.reader(handle)), [["label", "E_total"], ["licl", "-1"]])

    # 2026-10-18: Partial records retain attachments and overwrite protection for available outputs.
    def test_partial_export_attachments_and_conflicts(self):
        store = MemoryFileStore()
        content = b"ENCUT = 520\n"
        file_id = store.put(io.BytesIO(content))
        reference = {"gridfs_id": file_id, "compression": "none",
                     "stored_sha256": hashlib.sha256(content).hexdigest()}
        collection = MemoryCollection([{"_id": "energy_only", "E_total": -1.,
                                       "files": {"INCAR": reference}}])
        destination = self.root / "partial_files"
        with redirect_stdout(self.output):
            vm.export_from_mongo(collection, destination, file_store=store,
                export_original_files=True, export_energy_without_structure=True)
        self.assertEqual((destination / "energy_only" / "INCAR").read_bytes(), content)
        self.assertFalse((destination / "POSCAR_energy_only").exists())
        (destination / "energies.csv").unlink()
        with redirect_stdout(self.output), self.assertRaises(FileExistsError):
            vm.export_from_mongo(collection, destination, file_store=store,
                export_original_files=True, export_energy_without_structure=True)
        self.assertFalse((destination / "energies.csv").exists())
        # A structure-only record creates a POSCAR and a header-only energy table.
        destination = self.root / "structure_only"
        with redirect_stdout(self.output):
            vm.export_from_mongo(MemoryCollection([{"_id": "s", "struct": POSCAR}]), destination,
                                 export_structure_without_energy=True)
        self.assertTrue((destination / "POSCAR_s").is_file())
        with (destination / "energies.csv").open() as handle:
            self.assertEqual(list(csv.reader(handle)), [["label", "E_total"]])

    # 2026-10-18: Stored formulas label both complete and partial exports; no formula uses ID only.
    def test_formula_export_labels_and_fallback(self):
        documents = [
            {"_id": 29771, "formula": "Cu2S1", "struct": POSCAR, "E_total": -1.},
            {"_id": "no_formula", "struct": POSCAR, "E_total": -2.},
            {"_id": "null_formula", "formula": None, "struct": POSCAR, "E_total": -3.},
            {"_id": "empty_formula", "formula": "", "struct": POSCAR, "E_total": -4.},
            {"_id": "partial", "formula": "Cu1S1", "E_total": -5.},
            {"_id": "structure_only", "formula": "Cu1S2", "struct": POSCAR},
        ]
        destination = self.root / "formula_labels"
        with redirect_stdout(self.output):
            counts = vm.export_from_mongo(MemoryCollection(documents), destination,
                export_energy_without_structure=True, export_structure_without_energy=True)
        self.assertEqual(counts["exported"], 6)
        labels = {"Cu2S1_29771", "no_formula", "null_formula", "empty_formula", "Cu1S2_structure_only"}
        self.assertEqual({path.name for path in destination.glob("POSCAR_*")},
                         {"POSCAR_" + label for label in labels})
        with (destination / "energies.csv").open() as handle:
            rows = list(csv.reader(handle))
        self.assertEqual({row[0] for row in rows[1:]},
                         (labels - {"Cu1S2_structure_only"}) | {"Cu1S1_partial"})
        self.assertEqual((destination / "POSCAR_Cu2S1_29771").read_text().splitlines()[0],
                         "Cu2S1_29771")
        with self.assertRaises(ValueError):
            vm.make_label(29771, "bad/formula")
        self.assertEqual(vm.make_import_id("/archive/spin/job"), "spin__job")

    def test_missing_settings_do_not_connect(self):
        with patch.object(vm, "db_name", None), patch.object(vm, "MongoClient") as client:
            with self.assertRaisesRegex(ValueError, "Set db_name"):
                vm.main()
            client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
