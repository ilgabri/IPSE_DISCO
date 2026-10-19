#!/usr/bin/env python3
# 2026-10-16: Standalone transfer of VASP energies/structures to and from MongoDB.
"""Edit the options below, then run: python utilities/vasp_mongo.py

Import stores the last selected energy and CONTCAR (POSCAR fallback) per folder.
E_total is in eV per simulation cell; E_per_atom is in eV/atom. Formation
energies require separate elemental references. Reading the last energy does
not establish electronic or ionic convergence; check_vasp.py provides that check.

# 2026-10-18: Include the stored formula in export labels when available.
Export writes POSCAR_<formula>_<MongoDB-ID> (ID only if no formula) and an energy CSV.
The structure field defaults to 'struct', as in IPSE_DB. Other collections can
be read by changing export_energy_field and mongo_structure_field.

# 2026-10-17: Archive selected INCAR/OSZICAR/OUTCAR files in GridFS.
INCAR stays uncompressed; OSZICAR and OUTCAR use gzip. Optional export restores
the stored files under export_dir/<MongoDB-ID>/, retaining gzip compression.
"""

# 2026-10-16: All user options are defined here; no separate input module is needed.
action = "import"                      # "import": folders -> Mongo; "export": Mongo -> files
db_server_name = "mongodb://localhost:27017/"
db_name = None                        # Set the database name before running.
db_collection_name = None             # Set the collection name before running.
mongo_filters = {}                    # Additional Mongo query, used only when exporting.

# Element filters apply to both import and export. Empty elements means unrestricted.
elements = []                         # e.g. ["Pb", "S", "Cl"]
element_selection = "subset"           # subset: only these elements, including lower binaries
                                      # exact: exactly this set; contains_all: all, with extras allowed
excluded_elements = []                # e.g. ["O"]; always excluded

# Import: include root itself and subfolders to max_depth (None means any depth).
root_dir = "."
max_depth = 1                         # 0: root only; 1: also direct subfolders
# 2026-10-17: Import IDs depend only on trailing folders, with explicit update permission.
# GS: this is to crete id for importing. For exporting, it's POSCAR_<mongo id>
id_folder_levels = 2                  # Last N folders joined with "__"; import only.
id_prefix = ""                        # Optional project prefix, e.g. "sulfides".
update_existing = False               # True explicitly allows replacing an existing ID.
energy_type = "F"                     # F = OUTCAR TOTEN / OSZICAR F; E0 = sigma -> 0
energy_files = ("OUTCAR", "OSZICAR")   # Try in this order; use the next if unreadable/no energy.
structure_files = ("CONTCAR", "POSCAR")
require_normal_end = False            # True requires an OUTCAR normal termination marker.
# 2026-10-17: Optional original files are stored in GridFS, independently of energy parsing.
upload_incar = True                   # Store INCAR unchanged, if present.
upload_oszicar = True                 # Store OSZICAR.gz, if present.
upload_outcar = False                 # Store OUTCAR.gz, if present (potentially large).

# Export: energy values are taken from the chosen field without unit conversion.
export_dir = "mongo_export"
file_name_energies = "energies.csv"
export_energy_field = "E_total"        # Or "E_per_atom", "Eform", "E_total(eV/atom)", etc.
mongo_structure_field = "struct"       # Dotted nested field paths are also accepted.
overwrite = False                     # True permits replacing export files.
# 2026-10-17: Download stored attachments into export_dir/<MongoDB-ID>/.
export_original_files = True          # Preserve stored gzip compression on export.
# 2026-10-18: Optionally export the available result when its counterpart is missing/invalid.
export_energy_without_structure = False
export_structure_without_energy = False

import csv
# 2026-10-17: Stream gzip attachments through temporary files instead of loading OUTCAR in memory.
import gzip
import hashlib
import tempfile
from gridfs import GridFS
from gridfs.errors import GridFSError
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import socket
from datetime import datetime, timezone

from monty.json import MontyEncoder
from pymatgen.core import Element, Structure
from pymatgen.io.vasp.inputs import Poscar
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
ENERGY_PATTERNS = {
    ("OUTCAR", "F"): re.compile(r"free\s+energy\s+TOTEN\s*=\s*(" + NUMBER + r")"),
    ("OUTCAR", "E0"): re.compile(r"energy\(sigma->0\)\s*=\s*(" + NUMBER + r")"),
    ("OSZICAR", "F"): re.compile(r"\bF=\s*(" + NUMBER + r")"),
    ("OSZICAR", "E0"): re.compile(r"\bE0=\s*(" + NUMBER + r")"),
}


def find_job_dirs(root, max_depth=1):
    """Return sorted VASP folders, counting the root as depth zero."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
        raise ValueError("max_depth must be a non-negative integer or None")
    folders = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        depth = len(current.relative_to(root).parts)
        if max_depth is not None and depth >= max_depth:
            dirnames[:] = []
        if set(filenames).intersection({"OUTCAR", "OSZICAR", "CONTCAR", "POSCAR"}):
            folders.append(current)
    return sorted(folders)


def read_last_energy(path, energy_type="F"):
    """Read the last selected ionic-step energy; return None if no match exists."""
    path = Path(path)
    pattern = ENERGY_PATTERNS[(path.name, energy_type)]
    last_energy = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                value = float(match.group(1).replace("D", "E").replace("d", "e"))
                if math.isfinite(value):
                    last_energy = value
    return last_energy


def outcar_has_normal_end(path, nbytes=20000):
    """Check the OUTCAR tail for the normal-end markers used by check_vasp.py."""
    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(handle.tell() - nbytes, 0))
            tail = handle.read().decode(errors="replace")
        return ("Voluntary context switches" in tail or
                "General timing and accounting informations for this job:" in tail)
    except OSError:
        return False


def read_structure(folder, filenames=("CONTCAR", "POSCAR")):
    """Return (pymatgen Structure, source filename), trying valid files in order."""
    errors = []
    for name in filenames:
        path = Path(folder) / name
        if not path.is_file():
            continue
        try:
            if path.stat().st_size == 0:
                raise ValueError("empty file")
            poscar = Poscar.from_file(path, check_for_potcar=True, read_velocities=False)
            if not poscar.true_names:
                raise ValueError("element symbols unknown; supply VASP5 names or a POTCAR")
            return poscar.structure, name
        except (OSError, ValueError, TypeError, IndexError) as error:
            errors.append(f"{name}: {error}")
    raise ValueError("no readable structure; " + "; ".join(errors or ["files missing"]))


def matches_elements(structure, elements=None, selection="subset", excluded=None):
    """Select by actual structure species, avoiding formula-substring matches."""
    present = {element.symbol for element in structure.composition.elements}
    # 2026-10-18: Share species filtering with structure-free energy exports.
    return matches_element_symbols(present, elements, selection, excluded)


# 2026-10-18: Filter explicit element sets without requiring a structure object.
def matches_element_symbols(present, elements=None, selection="subset", excluded=None):
    """Apply element selection/exclusions to a known set of chemical symbols."""
    present = set(present)
    requested = set(elements or [])
    if present.intersection(excluded or []):
        return False
    if not requested:
        return True
    if selection == "subset":
        return present.issubset(requested)
    if selection == "exact":
        return present == requested
    if selection == "contains_all":
        return requested.issubset(present)
    raise ValueError("element_selection must be subset, exact, or contains_all")



# 2026-10-17: Keep calculation IDs readable and stable when results move between computers.
def make_import_id(folder, id_folder_levels=2, id_prefix=""):
    """Join the last N directory names, optionally preceded by a project prefix."""
    if type(id_folder_levels) is not int or id_folder_levels < 1:
        raise ValueError("id_folder_levels must be a positive integer")
    if not isinstance(id_prefix, str):
        raise ValueError("id_prefix must be a string")
    path = Path(folder).expanduser().absolute()
    parts = path.parts[1:] if path.anchor else path.parts
    if len(parts) < id_folder_levels:
        raise ValueError("folder has fewer components than id_folder_levels")
    label = "__".join(parts[-id_folder_levels:])
    return make_label(f"{id_prefix}__{label}" if id_prefix else label)


def read_vasp_result(folder, energy_type="F", energy_files=("OUTCAR", "OSZICAR"),
                     structure_files=("CONTCAR", "POSCAR"), require_normal_end=False,
                     id_folder_levels=2, id_prefix=""):
    # 2026-10-17: Folder-derived IDs are independent of host and leading path.
    """Build one Mongo-ready result with an ID from the last directory names.

    The document contains formula, elements, struct, E_total, E_per_atom,
    energy convention, source filenames, source host/folder, and end-marker status.
    Identical trailing folder names share an ID; imports reject duplicates by default.
    """
    folder = Path(folder).expanduser().resolve()
    normal_end = outcar_has_normal_end(folder / "OUTCAR")
    if require_normal_end and not normal_end:
        raise ValueError("OUTCAR has no normal termination marker")
    structure, structure_file = read_structure(folder, structure_files)
    errors = []
    for name in energy_files:
        try:
            energy = read_last_energy(folder / name, energy_type)
            if energy is None:
                raise ValueError(f"no final {energy_type} energy")
            break
        except (OSError, ValueError) as error:
            errors.append(f"{name}: {error}")
    else:
        raise ValueError("no readable energy; " + "; ".join(errors))

    host = socket.gethostname()
    identifier = make_import_id(folder, id_folder_levels, id_prefix)
    # 2026-10-16: Convert NumPy site properties (e.g. selective dynamics) to BSON-safe lists.
    struct_dict = json.loads(json.dumps(structure.as_dict(), cls=MontyEncoder))
    return {
        "_id": identifier, "schema_version": 1, "code": "VASP",
        "formula": structure.composition.reduced_formula,
        "formula_total": structure.composition.formula.replace(" ", ""),
        "elements": sorted(element.symbol for element in structure.composition.elements),
        "nsites": len(structure), "struct": struct_dict,
        "E_total": energy, "E_per_atom": energy / len(structure),
        "energy_type": energy_type, "energy_units": "eV",
        "energy_file": name, "structure_file": structure_file,
        "structure_is_final": structure_file == "CONTCAR",
        "outcar_normal_end": normal_end if (folder / "OUTCAR").is_file() else None,
        "source_folder": str(folder), "source_host": host,
    }



# 2026-10-17: Archive optional files separately so large OUTCARs never enter BSON documents.
def archive_vasp_files(collection, file_store, identifier, folder, filenames):
    """Upload selected files, link GridFS IDs, reuse identical content, and retire replaced blobs.

    Missing files leave existing attachments untouched. File errors are reported
    individually and do not prevent storage of the structure and energy.
    """
    # 2026-10-17: Attachment lookup failures must not misreport a successful energy import.
    try:
        previous = collection.find_one({"_id": identifier}, {"files": 1}) or {}
    except PyMongoError as error:
        print(f"WARNING attachments for {identifier}: {error}; structure/energy remain stored")
        return
    attachments = previous.get("files", {})
    for name in filenames:
        path = Path(folder) / name
        if not path.is_file():
            print(f"Attachment missing, skipped: {path}")
            continue
        try:
            compression = "none" if name == "INCAR" else "gzip"
            original_hash = hashlib.sha256()
            size = 0
            with tempfile.TemporaryFile() as payload:
                target = (gzip.GzipFile(filename="", mode="wb", fileobj=payload, mtime=0)
                          if compression == "gzip" else payload)
                try:
                    with path.open("rb") as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            original_hash.update(block)
                            size += len(block)
                            target.write(block)
                finally:
                    if target is not payload:
                        target.close()
                old = attachments.get(name, {})
                if (old.get("sha256") == original_hash.hexdigest()
                        and old.get("compression") == compression
                        and file_store.exists(old.get("gridfs_id"))):
                    print(f"Attachment unchanged: {name}")
                    continue
                stored_size = payload.tell()
                payload.seek(0)
                stored_hash = hashlib.sha256()
                for block in iter(lambda: payload.read(1024 * 1024), b""):
                    stored_hash.update(block)
                payload.seek(0)
                filename = name + (".gz" if compression == "gzip" else "")
                metadata = {"original_name": name, "filename": filename,
                            "compression": compression, "original_size": size,
                            "stored_size": stored_size, "sha256": original_hash.hexdigest(),
                            "stored_sha256": stored_hash.hexdigest()}
                file_id = file_store.put(payload, filename=filename,
                                         metadata={"record_id": identifier, **metadata})
            # Link only complete uploads. Failed links leave the blob available for recovery.
            result = collection.update_one({"_id": identifier},
                {"$set": {f"files.{name}": {"gridfs_id": file_id, **metadata}}})
            if result.matched_count != 1:
                file_store.delete(file_id)
                raise ValueError(f"record {identifier!r} disappeared during attachment upload")
            if old.get("gridfs_id") is not None:
                file_store.delete(old["gridfs_id"])
            print(f"Archived {name}: {size} -> {stored_size} bytes")
        except (OSError, ValueError, PyMongoError, GridFSError) as error:
            print(f"WARNING attachment {path}: {error}; structure/energy remain stored")


# 2026-10-17: Restore stored bytes with checksum verification before publishing the file.
def export_attachment(file_store, reference, destination, overwrite=False):
    """Stream one GridFS attachment to disk, retaining gzip and verifying its checksum."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            checksum = hashlib.sha256()
            with file_store.get(reference["gridfs_id"]) as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    checksum.update(block)
                    output.write(block)
        if checksum.hexdigest() != reference["stored_sha256"]:
            raise ValueError(f"attachment checksum mismatch: {destination}")
        if overwrite:
            temporary.replace(destination)
        else:
            os.link(temporary, destination)  # Atomic refusal if another export created the file.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def import_to_mongo(collection, root, max_depth=1, elements=None, selection="subset",
                    excluded=None, energy_type="F", energy_files=("OUTCAR", "OSZICAR"),
                    structure_files=("CONTCAR", "POSCAR"), require_normal_end=False,
                    id_folder_levels=2, id_prefix="", update_existing=False,
                    file_store=None, upload_incar=False, upload_oszicar=False, upload_outcar=False):
    # 2026-10-17: Duplicate IDs fail unless updates are explicitly enabled.
    """Import folders and return counts; reject existing IDs unless updates are enabled."""
    # 2026-10-17: Archive only files explicitly selected by the caller.
    selected_files = [name for name, enabled in (("INCAR", upload_incar),
                      ("OSZICAR", upload_oszicar), ("OUTCAR", upload_outcar)) if enabled]
    if selected_files and file_store is None:
        raise ValueError("A GridFS file_store is required to upload files")
    counts = {"inserted": 0, "updated": 0, "filtered": 0, "failed": 0}
    for folder in find_job_dirs(root, max_depth):
        try:
            document = read_vasp_result(folder, energy_type, energy_files,
                                        structure_files, require_normal_end, id_folder_levels, id_prefix)
            # 2026-10-16: Pymatgen may decode site properties in place; protect the BSON payload.
            structure = Structure.from_dict(deepcopy(document["struct"]))
            if not matches_elements(structure, elements, selection, excluded):
                counts["filtered"] += 1
                continue
            if not document["structure_is_final"]:
                print(f"WARNING {folder}: using POSCAR; final CONTCAR is unavailable")
            identifier = document.pop("_id")
            now = datetime.now(timezone.utc)
            document["updated_at"] = now
            # 2026-10-17: insert_one atomically rejects collisions, including concurrent imports.
            if update_existing:
                result = collection.update_one(
                    {"_id": identifier},
                    {"$set": document, "$setOnInsert": {"created_at": now}}, upsert=True)
                status = "inserted" if result.upserted_id is not None else "updated"
            else:
                try:
                    collection.insert_one({"_id": identifier, **document, "created_at": now})
                except DuplicateKeyError as error:
                    raise ValueError(
                        f"ID {identifier!r} already exists; choose another prefix/folder depth "
                        "or set update_existing=True to intentionally update it") from error
                status = "inserted"
            counts[status] += 1
            # 2026-10-17: A rejected duplicate never uploads attachments.
            if selected_files:
                archive_vasp_files(collection, file_store, identifier, folder, selected_files)
            print(f"{status}: {folder}  {document['formula']}  {document['E_total']:.16g} eV")
        except (OSError, ValueError, TypeError, KeyError, PyMongoError) as error:
            counts["failed"] += 1
            print(f"SKIPPED {folder}: {error}")
    print("Import summary:", counts)
    return counts


def get_field(document, field):
    """Read a literal top-level field or a dotted nested field path."""
    if field in document:
        return document[field]
    value = document
    for key in field.split("."):
        value = value[key]
    return value


def mongo_structure(value):
    """Decode a pymatgen structure dictionary or a POSCAR string from Mongo."""
    if isinstance(value, dict):
        structure = Structure.from_dict(value)
    elif isinstance(value, str):
        poscar = Poscar.from_str(value)
        if not poscar.true_names:
            raise ValueError("structure string has no identifiable element symbols")
        structure = poscar.structure
    else:
        raise ValueError("structure field must contain a structure dictionary or POSCAR text")
    if len(structure) == 0 or not structure.is_ordered:
        raise ValueError("POSCAR export requires a nonempty, ordered structure")
    return structure


# 2026-10-18: Optionally prefix the stored formula while keeping Mongo IDs unchanged.
def make_label(identifier, formula=None):
    """Return <formula>_<ID> when a nonempty formula exists, otherwise <ID>; reject unsafe paths."""
    label = str(identifier)
    if (not label or label in {".", ".."} or "/" in label or "\\" in label
            or any(ord(char) < 32 or ord(char) == 127 for char in label)):
        raise ValueError(f"ID {identifier!r} cannot be used as a safe filename")
    if isinstance(formula, str) and formula.strip():
        label = f"{make_label(formula.strip())}_{label}"
    return label


def export_from_mongo(collection, output_dir, energy_field="E_total", structure_field="struct",
                      elements=None, selection="subset", excluded=None, filters=None,
                      energy_filename="energies.csv", overwrite=False,
                      file_store=None, export_original_files=False,
                      export_energy_without_structure=False, export_structure_without_energy=False):
    """Write selected POSCARs and a label/energy CSV; return export counts.

    Energy values retain the chosen Mongo field's units. All candidate records
    and filename conflicts are checked before output files are written.

    With partial exports enabled, CSV rows require only a valid energy and POSCARs
    require only a valid structure. Missing/invalid counterparts are reported.
    Element filters without a structure require a nonempty, valid elements field.
    Records missing both results are skipped even if they have attachments.
    """
    output_dir = Path(output_dir).expanduser()
    if not energy_filename or Path(energy_filename).name != energy_filename:
        raise ValueError("file_name_energies must be a filename inside export_dir")
    records, labels = [], set()
    counts = {"exported": 0, "filtered": 0, "failed": 0}
    # 2026-10-17: Project attachment references only when file extraction is requested.
    projection = {"_id": 1, energy_field: 1, structure_field: 1}
    # 2026-10-18: Fetch stored formulas for readable labels without synthesizing missing ones.
    if not any(key == "formula" or key.startswith("formula.") for key in projection):
        projection["formula"] = 1
    # 2026-10-18: Explicit species metadata can filter records with no readable structure.
    if export_energy_without_structure and (elements or excluded):
        if not any(key == "elements" or key.startswith("elements.") for key in projection):
            projection["elements"] = 1
    if export_original_files:
        if file_store is None:
            raise ValueError("A GridFS file_store is required to export files")
        projection["files"] = 1
    for document in collection.find(filters or {}, projection):
        try:
            # 2026-10-18: Read structure and energy independently for optional partial exports.
            structure, energy = None, None
            missing = []
            try:
                structure = mongo_structure(get_field(document, structure_field))
            except (KeyError, ValueError, TypeError, IndexError) as error:
                missing.append(f"structure unavailable ({error})")
            try:
                raw_energy = get_field(document, energy_field)
                if isinstance(raw_energy, bool):
                    raise ValueError("energy must be numeric, not boolean")
                value = float(raw_energy)
                if not math.isfinite(value):
                    raise ValueError("energy is not finite")
                energy = value
            except (KeyError, ValueError, TypeError, OverflowError) as error:
                missing.append(f"energy unavailable ({error})")
            if structure is None and energy is None:
                raise ValueError("; ".join(missing))
            if structure is None and not export_energy_without_structure:
                raise ValueError("; ".join(missing))
            if energy is None and not export_structure_without_energy:
                raise ValueError("; ".join(missing))
            if structure is not None:
                selected = matches_elements(structure, elements, selection, excluded)
            elif elements or excluded:
                symbols = document.get("elements")
                if not isinstance(symbols, (list, tuple)) or not symbols:
                    raise ValueError("cannot apply element filters without structure or valid elements metadata")
                if any(not isinstance(symbol, str) for symbol in symbols):
                    raise ValueError("elements metadata must contain chemical symbols")
                for symbol in symbols:
                    Element(symbol)
                selected = matches_element_symbols(symbols, elements, selection, excluded)
            else:
                selected = True
            if not selected:
                counts["filtered"] += 1
                continue
            label = make_label(document["_id"], document.get("formula"))
            if label in labels:
                raise ValueError(f"duplicate export label {label}")
            attachments = []
            if export_original_files:
                for name, reference in document.get("files", {}).items():
                    if name not in {"INCAR", "OSZICAR", "OUTCAR"}:
                        raise ValueError(f"unknown attachment {name!r}")
                    filename = name + (".gz" if reference["compression"] == "gzip" else "")
                    # 2026-10-18: Attachment folders retain the Mongo ID independently of POSCAR labels.
                    attachments.append((output_dir / make_label(document["_id"]) / filename, reference))
            labels.add(label)
            # 2026-10-18: Never synthesize a missing structure or energy value.
            text = Poscar(structure, comment=label).get_str() if structure is not None else None
            records.append((label, energy, text, attachments))
            if missing:
                print(f"PARTIAL Mongo record {label}: " + "; ".join(missing))
        except (KeyError, ValueError, TypeError, IndexError) as error:
            counts["failed"] += 1
            print(f"SKIPPED Mongo record {document.get('_id')}: {error}")

    table_path = output_dir / energy_filename
    # 2026-10-18: Preflight only files that will actually be produced.
    poscar_paths = [output_dir / f"POSCAR_{label}" for label, _, text, _ in records
                    if text is not None]
    if len(set(poscar_paths + [table_path])) != len(poscar_paths) + 1:
        raise ValueError("energy table filename conflicts with a POSCAR filename")
    # 2026-10-17: Include attachment destinations and directories in export preflight.
    attachment_paths = [path for _, _, _, attachments in records for path, _ in attachments]
    output_paths = [table_path] + poscar_paths + attachment_paths
    for path in attachment_paths:
        if path.parent in output_paths or (path.parent.exists() and not path.parent.is_dir()):
            raise ValueError(f"attachment directory conflicts with an output file: {path.parent}")
    if not overwrite:
        existing = [path for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError(f"Export files already exist; set overwrite=True to replace: {existing[0]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with table_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", energy_field])
        # 2026-10-18: Write each available output independently, retaining the same Mongo ID.
        for label, energy, text, attachments in records:
            if text is not None:
                path = output_dir / f"POSCAR_{label}"
                with path.open(mode, encoding="utf-8") as poscar_file:
                    poscar_file.write(text)
            # 2026-10-17: Retrieval errors are explicit; downloaded bytes retain their stored format.
            for destination, reference in attachments:
                export_attachment(file_store, reference, destination, overwrite)
            if energy is not None:
                writer.writerow([label, format(energy, ".17g")])
            counts["exported"] += 1
            print(f"exported: {label}  structure={text is not None}  energy={energy}")
    print("Export summary:", counts)
    print("Energy table:", table_path)
    return counts


def main():
    """Validate the settings, connect to Mongo, and run the requested transfer."""
    if action not in {"import", "export"}:
        raise ValueError("action must be import or export")
    if not db_name or not db_collection_name:
        raise ValueError("Set db_name and db_collection_name at the beginning of this script")
    if element_selection not in {"subset", "exact", "contains_all"}:
        raise ValueError("element_selection must be subset, exact, or contains_all")
    for symbol in list(elements) + list(excluded_elements):
        Element(symbol)  # Validate symbols before accessing Mongo.
    if action == "import":
        if energy_type not in {"F", "E0"}:
            raise ValueError("energy_type must be F or E0")
        if not energy_files or any(name not in {"OUTCAR", "OSZICAR"} for name in energy_files):
            raise ValueError("energy_files must contain OUTCAR and/or OSZICAR")
        if not structure_files or any(name not in {"CONTCAR", "POSCAR"} for name in structure_files):
            raise ValueError("structure_files must contain CONTCAR and/or POSCAR")
        if not Path(root_dir).expanduser().is_dir():
            raise NotADirectoryError(root_dir)
    with MongoClient(db_server_name, serverSelectionTimeoutMS=10000) as client:
        client.admin.command("ping")
        collection = client[db_name][db_collection_name]
        # 2026-10-17: Each material collection uses its own GridFS bucket in the same database.
        file_store = GridFS(client[db_name], collection=db_collection_name + "_files")
        if action == "import":
            import_to_mongo(collection, root_dir, max_depth, elements, element_selection,
                            excluded_elements, energy_type, energy_files, structure_files,
                            require_normal_end, id_folder_levels, id_prefix, update_existing,
                            file_store, upload_incar, upload_oszicar, upload_outcar)
        else:
            export_from_mongo(collection, export_dir, export_energy_field,
                              mongo_structure_field, elements, element_selection,
                              excluded_elements, mongo_filters, file_name_energies, overwrite,
                              file_store, export_original_files,
                              export_energy_without_structure, export_structure_without_energy)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, PyMongoError, GridFSError) as error:
        raise SystemExit(f"ERROR: {error}")
