#!/usr/bin/env python3
"""Portable, read-only inspection of a local Mach-O GUIDE reference executable."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile


def inspect(path):
    raw = path.read_bytes()
    if raw[:4] != b"\xcf\xfa\xed\xfe":
        raise ValueError("expected little-endian 64-bit Mach-O")
    header = struct.unpack_from("<IiiIIIII", raw)
    offset = 32
    dylibs = []
    sections = []
    version = None
    symtab = None
    for _ in range(header[4]):
        cmd, size = struct.unpack_from("<II", raw, offset)
        if size < 8 or offset + size > len(raw):
            raise ValueError("invalid load command")
        if cmd == 2:
            symtab = struct.unpack_from("<IIII", raw, offset + 8)
        elif cmd == 12:
            name_offset = struct.unpack_from("<I", raw, offset + 8)[0]
            dylibs.append(raw[offset + name_offset:offset + size].split(b"\0")[0].decode())
        elif cmd == 0x32:
            minimum = struct.unpack_from("<I", raw, offset + 12)[0]
            version = f"{minimum >> 16}.{(minimum >> 8) & 255}.{minimum & 255}"
        elif cmd == 0x19:
            count = struct.unpack_from("<I", raw, offset + 64)[0]
            for j in range(count):
                start = offset + 72 + 80 * j
                name = raw[start:start + 16].split(b"\0")[0].decode()
                sections.append(name)
        offset += size
    symbols = []
    if symtab:
        symoff, nsyms, stroff, strsize = symtab
        strings = raw[stroff:stroff + strsize]
        for i in range(nsyms):
            strx, kind, section, _, address = struct.unpack_from("<IBBHQ", raw, symoff + 16 * i)
            name = strings[strx:].split(b"\0")[0].decode(errors="replace")
            if (kind & 0x0e) == 0x0e and section and sections[section - 1] == "__text":
                symbols.append({"name": name, "address": hex(address)})
    printable = [s.decode() for s in re.findall(rb"[\x20-\x7e]{4,}", raw)]
    modules = collections.Counter(s["name"].split("_MP_")[0].split("_IP_")[0]
                                  for s in symbols if "_MP_" in s["name"] or "_IP_" in s["name"])
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw), "cpu_type": header[1], "minimum_macos": version,
            "dylibs": dylibs, "symbol_count": symtab[1] if symtab else 0,
            "text_section_symbol_count": len(symbols), "module_symbol_counts": dict(modules),
            "source_filenames": sorted(set(s for s in printable if re.fullmatch(r"[\w]+\.(?:f90|f95|f)", s))),
            "banner": [s for s in printable if "Build date:" in s], "text_symbols": symbols}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=Path(__file__).resolve().parents[1] / ".reference/upstream/macos-arm64/guide")
    parser.add_argument("--probe", action="store_true", help="attempt execution in a temporary working directory")
    parser.add_argument("--timeout", type=float, default=15)
    args = parser.parse_args()
    if not args.path.is_file():
        parser.error(f"reference executable not found: {args.path}; supply its path explicitly (see docs/reference-upstream.md)")
    result = inspect(args.path.resolve())
    if args.probe:
        with tempfile.TemporaryDirectory(prefix="guide-reference-") as directory:
            try:
                p = subprocess.run([str(args.path.resolve())], input="0\n\n", text=True,
                                   capture_output=True, timeout=args.timeout, cwd=directory,
                                   env={**os.environ, "TMPDIR": directory})
                result["probe"] = {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
            except subprocess.TimeoutExpired:
                result["probe"] = {"timeout_seconds": args.timeout}
            except OSError as exc:
                result["probe"] = {"error": str(exc)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
