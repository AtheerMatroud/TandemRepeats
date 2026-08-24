# TandemRepeats

Code and documentation for tandem-repeat analysis of the human Y chromosome.

## Contents

- `download_hallast_y43.py` — cross-platform (Windows/macOS/Linux, standard library only) downloader for the 43 assembled human Y chromosomes from Hallast et al. 2023 (*Nature* 621, 344–354, [doi:10.1038/s41586-023-06425-6](https://doi.org/10.1038/s41586-023-06425-6)), mirrored from the HGSVC3 collection on the 1000 Genomes FTP.
- `download_hallast_y43.sh` — equivalent Bash/wget version (WSL, Git Bash, Linux, macOS).
- `project_log.tex` — LaTeX work log documenting each session's steps.

## Usage

```bash
python download_hallast_y43.py [output_dir]   # or
bash download_hallast_y43.sh [output_dir]
```

Default output directory: `hallast_y43_assemblies/`. The full dataset is tens of GB; both scripts are resumable and skip files already downloaded.

## Data sources

- HGSVC3 FTP: https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/HGSVC3/working/20230412_sigY_assembly_data/
- INSDC BioProjects: PRJEB58376, PRJNA988114 (assemblies); PRJEB41077, PRJEB39684, PRJEB39750 (reads)
