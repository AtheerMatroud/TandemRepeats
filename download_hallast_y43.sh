#!/usr/bin/env bash
# ============================================================================
# download_hallast_y43.sh
#
# Download the 43 assembled human Y chromosomes from:
#   Hallast, Ebert, Loftus et al. (2023) "Assembly of 43 human Y chromosomes
#   reveals extensive complexity and variation", Nature 621, 344-354.
#   https://doi.org/10.1038/s41586-023-06425-6
#
# Data host: 1000 Genomes / HGSVC3 FTP (EBI)
#   https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/HGSVC3/working/20230412_sigY_assembly_data/
#
# Related INSDC accessions (archived copies): PRJEB58376, PRJNA988114
#
# Usage:
#   bash download_hallast_y43.sh [output_dir]
#
# On Windows, run this inside WSL or Git Bash (wget required),
# or use the Python alternative: download_hallast_y43.py
# ============================================================================

set -euo pipefail

BASE_URL="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/HGSVC3/working/20230412_sigY_assembly_data/"
OUTDIR="${1:-hallast_y43_assemblies}"

command -v wget >/dev/null 2>&1 || {
    echo "ERROR: wget is required. Install it (e.g. 'sudo apt install wget') and retry." >&2
    exit 1
}

mkdir -p "$OUTDIR"
echo "Downloading Hallast et al. 2023 Y assemblies into: $OUTDIR"
echo "Source: $BASE_URL"
echo

# -r  : recursive
# -np : never ascend to the parent directory
# -nH : no host-name directory
# --cut-dirs=6 : strip vol1/ftp/data_collections/HGSVC3/working/20230412_sigY_assembly_data
# -c  : continue partially downloaded files (safe to re-run)
# -R  : skip FTP listing artefacts
wget -r -np -nH --cut-dirs=6 -c \
     -R "index.html*" \
     -P "$OUTDIR" \
     "$BASE_URL"

echo
echo "Done. Assembly files are under: $OUTDIR"
echo "Verify sizes/checksums against any MD5/manifest files included in the download."
