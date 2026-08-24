#!/usr/bin/env python3
"""
download_hallast_y43.py

Cross-platform (Windows/macOS/Linux) downloader for the 43 assembled human
Y chromosomes from:

    Hallast, Ebert, Loftus et al. (2023) "Assembly of 43 human Y chromosomes
    reveals extensive complexity and variation", Nature 621, 344-354.
    https://doi.org/10.1038/s41586-023-06425-6

Data host: 1000 Genomes / HGSVC3 FTP at EBI, reachable both over HTTPS and FTP:

    https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/HGSVC3/working/20230412_sigY_assembly_data/

This script walks the directory tree over FTP (no external dependencies,
standard library only) and mirrors every file locally. Safe to re-run:
files already present with the same size are skipped.

Usage:
    python download_hallast_y43.py [output_dir]

Default output_dir: ./hallast_y43_assemblies
"""

import os
import sys
from ftplib import FTP

HOST = "ftp.1000genomes.ebi.ac.uk"
ROOT = "/vol1/ftp/data_collections/HGSVC3/working/20230412_sigY_assembly_data"


def walk(ftp: FTP, remote_dir: str, local_dir: str) -> None:
    """Recursively mirror remote_dir into local_dir."""
    os.makedirs(local_dir, exist_ok=True)
    entries = []
    ftp.retrlines(f"LIST {remote_dir}", entries.append)

    for line in entries:
        parts = line.split(maxsplit=8)
        if len(parts) < 9:
            continue
        kind, size, name = parts[0][0], parts[4], parts[8]
        if name in (".", ".."):
            continue
        remote_path = f"{remote_dir}/{name}"
        local_path = os.path.join(local_dir, name)

        if kind == "d":
            walk(ftp, remote_path, local_path)
        elif kind == "-":
            expected = int(size)
            if os.path.exists(local_path) and os.path.getsize(local_path) == expected:
                print(f"  [skip] {name} (already complete)")
                continue
            print(f"  [get ] {remote_path}  ({expected / 1e6:.1f} MB)")
            with open(local_path, "wb") as fh:
                ftp.retrbinary(f"RETR {remote_path}", fh.write)


def main() -> None:
    outdir = sys.argv[1] if len(sys.argv) > 1 else "hallast_y43_assemblies"
    print(f"Connecting to {HOST} ...")
    ftp = FTP(HOST, timeout=120)
    ftp.login()  # anonymous
    print(f"Mirroring {ROOT}\n     into {os.path.abspath(outdir)}\n")
    walk(ftp, ROOT, outdir)
    ftp.quit()
    print("\nDone. Verify against any MD5/manifest files included in the download.")


if __name__ == "__main__":
    main()
