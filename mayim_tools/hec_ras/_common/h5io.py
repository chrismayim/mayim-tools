"""Opening HEC-RAS HDF5 files when h5py cannot load inside QGIS.

On Windows a pip-installed h5py brings its own HDF5 library. Inside QGIS,
GDAL has usually loaded QGIS's own HDF5 first and the two clash: importing
h5py fails with "DLL load failed while importing _errors: The specified
procedure could not be found". Outside QGIS (the same Python, run on its
own) h5py loads fine.

``open_h5(path)`` therefore tries, in order:

1. h5py in this process;
2. h5py in a separate Python process (QGIS's own python.exe), which copies
   every dataset under ``Geometry`` (or under the ``roots`` asked for; the
   whole file when none is there), with the attributes, into a temporary
   .npz file that is then read here;
3. GDAL's multidimensional HDF5 driver (ships with QGIS). Some GDAL
   versions (3.13) return only the first character of text fields inside
   tables, so names may be cut short; ``NOTES`` then says so.

Each returns an object used like a read-only ``h5py.File`` for what the
RAS readers need: a context manager, ``"a/b" in f``, ``f["a/b"]`` (a group
or a dataset), ``group.keys()``, ``dataset[()]`` (numpy array; tables as
numpy structured arrays, text as bytes like h5py) and
``dataset.dtype.names``. ``is_group(obj)`` replaces
``isinstance(obj, h5py.Group)``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

_H5PY = None  # the h5py module, False when it cannot be imported
H5PY_ERROR: str | None = None  # why h5py could not be imported
_EXTERNAL = None  # python.exe that can import h5py, False when none
NOTES: list[str] = []  # messages for the run log (reader used, problems)
EXTRACT_TIMEOUT_S = 900


def _note(text: str) -> None:
    if text not in NOTES:
        NOTES.append(text)


def _h5py():
    global _H5PY, H5PY_ERROR
    if _H5PY is None:
        forced = os.environ.get("MAYIM_HDF5_READER", "")
        if forced in ("gdal", "external"):
            _H5PY, H5PY_ERROR = False, f"reader chosen: MAYIM_HDF5_READER={forced}"
    if _H5PY is None:
        try:
            import h5py

            _H5PY = h5py
        except Exception as e:  # ImportError, including DLL load failures
            _H5PY = False
            H5PY_ERROR = f"{type(e).__name__}: {e}"
    return _H5PY or None


def _python_candidates() -> list[str]:
    names = ("python.exe", "python3.exe", "python3", "python")
    out = []
    exe = sys.executable or ""
    if os.path.basename(exe).lower().startswith("python"):
        out.append(exe)
    for base in dict.fromkeys((sys.prefix, sys.exec_prefix, sys.base_prefix)):
        for sub in ("", "bin"):
            out += [os.path.join(base, sub, n) for n in names]
    return [p for p in dict.fromkeys(out) if os.path.isfile(p)]


def _run(args, timeout):
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=dict(os.environ),
        **kwargs,
    )


def _external_python():
    """A Python executable (QGIS's own) that can import h5py on its own."""
    global _EXTERNAL
    if os.environ.get("MAYIM_HDF5_READER") == "gdal":
        return None
    if _EXTERNAL is None:
        _EXTERNAL = False
        for exe in _python_candidates():
            try:
                r = _run([exe, "-c", "import h5py, numpy"], 120)
            except Exception:
                continue
            if r.returncode == 0:
                _EXTERNAL = exe
                break
    return _EXTERNAL or None


def backend() -> str:
    """'h5py', 'h5py-external' or 'gdal': what open_h5 will use."""
    if _h5py() is not None:
        return "h5py"
    if _external_python() is not None:
        return "h5py-external"
    return "gdal"


def open_h5(path: str, force_gdal: bool = False, roots=None):
    """Read-only HDF5 file (h5py.File, the copied file, or the GDAL reader).
    ``roots``: the groups the separate-process copy takes (default
    ``Geometry``; e.g. add the results groups for a plan's results); the
    copy also keeps the attributes of every group and dataset it takes and
    of the file itself."""
    if not force_gdal:
        h5py = _h5py()
        if h5py is not None:
            return h5py.File(path, "r")
        exe = _external_python()
        if exe is not None:
            try:
                return _extract(exe, path, tuple(roots or ("Geometry",)))
            except Exception as e:
                _note(
                    f"Reading the HDF5 file with h5py in a separate process failed "
                    f"({e}); GDAL was used instead."
                )
    return GdalH5File(path)


def is_group(obj) -> bool:
    if isinstance(obj, (GdalH5Group, CopyH5Group)):
        return True
    h5py = _h5py()
    return h5py is not None and isinstance(obj, h5py.Group)


# ---------------------------------------------------------------------------
# h5py in a separate process -> .npz copy
# ---------------------------------------------------------------------------

_EXTRACT_SCRIPT = r"""
import json, sys
import numpy as np
import h5py

src, out, roots = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])


def as_bytes(values):
    return [v if isinstance(v, bytes) else str(v).encode("utf-8") for v in values]


def fixed(a):
    # object (variable-length text) -> fixed-length bytes, so np.savez
    # needs no pickling
    if a.dtype.names:
        fields, cols = [], {}
        for n in a.dtype.names:
            col = a[n]
            if col.dtype.kind == "O":
                vals = as_bytes(col.reshape(-1).tolist())
                width = max([len(v) for v in vals] + [1])
                cols[n] = np.array(vals, dtype=f"S{width}").reshape(col.shape)
                fields.append((n, cols[n].dtype))
            else:
                cols[n] = col
                ft = a.dtype.fields[n][0]  # no h5py metadata; keep sub-arrays
                fields.append((n, ft.base.str, ft.shape) if ft.shape else (n, ft.str))
        b = np.empty(a.shape, dtype=fields)
        for n in a.dtype.names:
            b[n] = cols[n]
        return b
    if a.dtype.kind == "O":
        vals = as_bytes(np.asarray(a).reshape(-1).tolist())
        width = max([len(v) for v in vals] + [1])
        return np.array(vals, dtype=f"S{width}").reshape(np.shape(a))
    a = np.asarray(a)
    return a.astype(np.dtype(a.dtype.str), copy=False)


arrays, names, groups, attrs = {}, {}, [""], []


def keep_attrs(path, obj):
    for k in obj.attrs:
        try:
            value = fixed(np.asarray(obj.attrs[k]))
        except Exception:
            continue
        key = f"a{len(attrs)}"
        arrays[key] = value
        attrs.append([path, k, key])


with h5py.File(src, "r") as f:
    keep_attrs("", f)
    present = [r for r in roots if r in f] or ([] if "Geometry" in f else [""])
    for root in present:
        parts = root.split("/")
        for i in range(1, len(parts) + 1):  # the root's parent groups
            g = "/".join(parts[:i])
            if g not in groups:
                groups.append(g)
                keep_attrs(g, f[g])
        start = f[root] if root else f

        def visit(name, obj, root=root):
            full = f"{root}/{name}" if root else name
            if isinstance(obj, h5py.Group):
                if full not in groups:
                    groups.append(full)
                keep_attrs(full, obj)
            elif isinstance(obj, h5py.Dataset):
                key = f"d{len(names)}"
                arrays[key] = fixed(obj[()])
                names[key] = full
                keep_attrs(full, obj)

        start.visititems(visit)
np.savez(out, **arrays)
with open(out + ".json", "w", encoding="utf-8") as fh:
    json.dump({"groups": groups, "datasets": names, "attrs": attrs}, fh)
"""


_COPIES: dict = {}  # (path, mtime, size) -> (data, groups): one copy per file


def clear_cache() -> None:
    """Drops the copied files (call at the end of a run)."""
    _COPIES.clear()


def _extract(exe: str, path: str, roots: tuple = ("Geometry",)):
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size, roots)
    if key in _COPIES:
        return CopyH5File(*_COPIES[key])
    tmp = tempfile.mkdtemp(prefix="mayim_h5_")
    out = os.path.join(tmp, "copy.npz")
    r = _run(
        [exe, "-c", _EXTRACT_SCRIPT, path, out, json.dumps(list(roots))],
        EXTRACT_TIMEOUT_S,
    )
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-1:]
        raise RuntimeError(tail[0] if tail else f"exit code {r.returncode}")
    with open(out + ".json", encoding="utf-8") as fh:
        index = json.load(fh)
    attrs: dict = {}
    with np.load(out, allow_pickle=False) as npz:
        data = {index["datasets"][k]: npz[k] for k in npz.files if k[0] == "d"}
        for obj, name, k in index.get("attrs", []):
            value = npz[k]
            attrs.setdefault(obj, {})[name] = value[()] if value.ndim == 0 else value
    for name in (out, out + ".json"):
        try:
            os.remove(name)
        except OSError:
            pass
    try:
        os.rmdir(tmp)
    except OSError:
        pass
    _note(
        "The HDF5 file was read with h5py in a separate Python process "
        f"({exe}), because h5py cannot load inside QGIS ({H5PY_ERROR})."
    )
    _COPIES[key] = (data, set(index["groups"]), attrs)
    return CopyH5File(*_COPIES[key])


class CopyH5Dataset:
    def __init__(self, array: np.ndarray, name: str, attrs: dict | None = None):
        self._a = array
        self.name = name
        self.attrs = attrs or {}
        self.shape = array.shape
        self.dtype = array.dtype

    def __len__(self) -> int:
        return self.shape[0] if self.shape else 1

    def __getitem__(self, key):
        if key == () or key is Ellipsis:
            return self._a
        return self._a[key]


class CopyH5Group:
    def __init__(self, data: dict, groups: set, name: str = "", attrs=None):
        self._data, self._groups, self.name = data, groups, name
        self._attrs = attrs if attrs is not None else {}
        self.attrs = self._attrs.get(name, {})

    def _full(self, path: str) -> str:
        parts = [p for p in f"{self.name}/{path}".split("/") if p]
        return "/".join(parts)

    def keys(self) -> list:
        prefix = f"{self.name}/" if self.name else ""
        kids = set()
        for full in list(self._data) + list(self._groups):
            if full and full.startswith(prefix) and full != self.name:
                kids.add(full[len(prefix) :].split("/", 1)[0])
        return sorted(kids)

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, path: str):
        full = self._full(path)
        if full in self._data:
            return CopyH5Dataset(self._data[full], full, self._attrs.get(full))
        if full in self._groups:
            return CopyH5Group(self._data, self._groups, full, self._attrs)
        raise KeyError(path)

    def __contains__(self, path: str) -> bool:
        full = self._full(path)
        return full in self._data or full in self._groups


class CopyH5File(CopyH5Group):
    def __init__(self, data: dict, groups: set, attrs: dict | None = None):
        super().__init__(data, groups, "", attrs)

    def close(self):
        self._data = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ---------------------------------------------------------------------------
# GDAL reader
# ---------------------------------------------------------------------------


def _numpy_type(gdal_type):
    from osgeo import gdal_array

    return np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(gdal_type))


def _text_components(dt) -> set:
    """Names of the text fields of a GDAL compound type: string components,
    and (GDAL 3.13) fixed-length strings shown as one byte with a gap
    after them."""
    from osgeo import gdal

    comps = dt.GetComponents()
    offsets = [c.GetOffset() for c in comps] + [dt.GetSize()]
    out = set()
    for i, c in enumerate(comps):
        t = c.GetType()
        if t.GetClass() != gdal.GEDTC_NUMERIC:
            out.add(c.GetName())
        elif t.GetSize() == 1 and offsets[i + 1] - offsets[i] > 1:
            out.add(c.GetName())
    return out


class GdalH5Dataset:
    def __init__(self, md_array, name: str):
        self._a = md_array
        self.name = name

    @property
    def shape(self) -> tuple:
        return tuple(int(d.GetSize()) for d in self._a.GetDimensions())

    def __len__(self) -> int:
        shape = self.shape
        return shape[0] if shape else 1

    def _components(self):
        from osgeo import gdal

        dt = self._a.GetDataType()
        if dt.GetClass() != gdal.GEDTC_COMPOUND:
            return None
        return dt.GetComponents()

    @property
    def dtype(self) -> np.dtype:
        from osgeo import gdal

        comps = self._components()
        dt = self._a.GetDataType()
        if comps is None:
            if dt.GetClass() == gdal.GEDTC_STRING:
                return np.dtype(object)
            return _numpy_type(dt.GetNumericDataType())
        fields = []
        text = _text_components(dt)
        for c in comps:
            t = c.GetType()
            if c.GetName() not in text:
                fields.append((c.GetName(), _numpy_type(t.GetNumericDataType())))
            else:
                fields.append((c.GetName(), object))  # strings, as bytes
        return np.dtype(fields)

    def _strings(self, values):
        """Text values as an object array of bytes in this dataset's shape.
        GDAL versions differ: a list of str, an array of str or bytes, or
        (newer GDAL, fixed-length strings) an integer array of character
        codes with one more dimension than the dataset."""
        n = int(np.prod(self.shape)) if self.shape else 1
        if isinstance(values, (str, bytes)):
            values = [values]
        arr = np.asarray(values)
        if arr.dtype.kind in "iu" and arr.size:
            codes = arr.reshape(n, -1).astype(np.uint8)
            out = [bytes(row).split(b"\0", 1)[0] for row in codes]
        else:
            out = [
                v.encode("utf-8") if isinstance(v, str) else bytes(v)
                for v in arr.reshape(-1).tolist()
            ]
        result = np.empty(n, dtype=object)
        result[:] = out
        return result.reshape(self.shape) if self.shape else result[0]

    def _read_plain(self, md_array, text: bool = False):
        from osgeo import gdal

        dt = md_array.GetDataType()
        if text and dt.GetClass() == gdal.GEDTC_NUMERIC:
            # GDAL 3.13: a fixed-length text field shown as character codes
            return self._strings(md_array.ReadAsArray())
        if text or dt.GetClass() == gdal.GEDTC_STRING:
            values = md_array.Read()
            if isinstance(values, (bytes, bytearray)):
                values = [bytes(values)]
            return self._strings(values)
        return md_array.ReadAsArray()

    def __getitem__(self, key):
        from osgeo import gdal

        comps = self._components()
        if comps is None:
            data = self._read_plain(self._a)
        else:
            data = np.empty(self.shape, dtype=self.dtype)
            offsets = [c.GetOffset() for c in comps] + [self._a.GetDataType().GetSize()]
            for i, c in enumerate(comps):
                t = c.GetType()
                if t.GetClass() == gdal.GEDTC_NUMERIC and t.GetSize() == 1:
                    if offsets[i + 1] - offsets[i] > 1:
                        _note(
                            f"GDAL {gdal.__version__} returns only the first "
                            f"character of the text field '{c.GetName()}' in "
                            f"{self.name}; names read from it are incomplete."
                        )
            text = _text_components(self._a.GetDataType())
            for c in comps:
                name = c.GetName()
                view = self._a.GetView(f'["{name}"]')
                data[name] = self._read_plain(view, text=name in text)
        if key == () or key is Ellipsis:
            return data
        return data[key]


class GdalH5Group:
    def __init__(self, group, name: str = "/"):
        self._g = group
        self.name = name

    def _groups(self) -> list:
        return list(self._g.GetGroupNames() or [])

    def _arrays(self) -> list:
        return list(self._g.GetMDArrayNames() or [])

    def keys(self) -> list:
        return sorted(self._groups() + self._arrays())

    def __iter__(self):
        return iter(self.keys())

    def _child(self, part: str):
        if part in self._groups():
            return GdalH5Group(self._g.OpenGroup(part), f"{self.name}/{part}")
        if part in self._arrays():
            return GdalH5Dataset(self._g.OpenMDArray(part), f"{self.name}/{part}")
        raise KeyError(part)

    def __getitem__(self, path: str):
        node = self
        for part in [p for p in str(path).split("/") if p]:
            if not isinstance(node, GdalH5Group):
                raise KeyError(path)
            node = node._child(part)
        return node

    def __contains__(self, path: str) -> bool:
        try:
            self[path]
            return True
        except KeyError:
            return False


class GdalH5File(GdalH5Group):
    def __init__(self, path: str):
        from osgeo import gdal

        gdal.UseExceptions()
        try:
            self._ds = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER | gdal.OF_READONLY)
        except RuntimeError as e:
            raise OSError(
                f"Could not open {path} as an HDF5 file through GDAL ({e}). "
                f"h5py could not be loaded either ({H5PY_ERROR})."
            ) from e
        if self._ds is None:
            raise OSError(f"Could not open {path} as an HDF5 file through GDAL.")
        super().__init__(self._ds.GetRootGroup(), "")

    def close(self):
        self._g = None
        self._ds = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
