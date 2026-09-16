# Publishing these repositories and getting the DOIs

Both manuscripts promise a replication package with a persistent identifier.
This is how to produce it. Roughly 20 minutes for both.

## 1. Create the repositories

Two separate repositories, not one. Each paper cites its own DOI, and a shared
repository would give both papers the same identifier and make the version
history of one paper's data depend on edits to the other's.

Suggested names, already used in the CITATION files:

- `model-hub-emissions-disclosure`
- `quantization-inference-energy`

Make them **public** before connecting Zenodo. Zenodo cannot archive a private
repository.

## 2. Push

```bash
cd model-hub-emissions-disclosure
git init
git add .
git commit -m "Replication package for environmental disclosure audit"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/model-hub-emissions-disclosure.git
git push -u origin main
```

Then the same for the second repository.

Before pushing, replace `REPLACE-ME` in `CITATION.cff` with your GitHub
username. GitHub reads that file and shows a "Cite this repository" button, so
an error there is visible to everyone.

## 3. Connect Zenodo

1. Sign in to zenodo.org with your GitHub account.
2. Open the GitHub settings page in Zenodo and switch **on** the toggle for each
   of the two repositories.
3. Back on GitHub, create a release for each repository: Releases, then Draft a
   new release, tag `v1.0.0`, title "Replication package v1.0.0", publish.

Zenodo archives the repository at that tag and mints a DOI within a few minutes.
You will get two DOIs per repository: a version DOI for `v1.0.0` and a concept
DOI that always resolves to the latest version.

**Cite the concept DOI in the papers.** It stays valid if you release a
corrected version later.

## 4. Fill in the manuscripts

Each paper has a bracketed placeholder in the Availability of data and materials
declaration. Replace it with, for example:

> The dataset supporting the conclusions of this article is available in the
> Zenodo repository, https://doi.org/10.5281/zenodo.XXXXXXX

Then add the same DOI to the `preferred-citation` block in `CITATION.cff` and
push the change. That does not require a new release.

## 5. Order of operations matters

Do this **before** you submit, not after. Reviewers at SpringerOpen journals
routinely check that the availability statement resolves, and a statement that
points at nothing is a common cause of a first-round complaint.

If you would rather not make the code public before acceptance, Zenodo supports
restricted records with a reviewer access link, and you can open it at
acceptance. In that case put the access link in the cover letter and say in the
declaration that the archive will be made public on acceptance.

## Before you push: a short checklist

- No API tokens or credentials anywhere in the notebooks. The harness reads a
  Hugging Face token from a variable that is `None` by default; confirm it is
  still `None` in the committed copy.
- Notebook outputs are cleared in these files. If you re-run and re-commit,
  clear them again, or the repository will grow quickly.
- The `.gitignore` excludes `raw_runs.csv` and `summary_runs.csv` so that a
  local re-run does not overwrite the released snapshot in `data/`. If you
  deliberately want to publish a new run, add it under `data/` with a new name.
- Check that `data/runs_48.csv` has 48 rows and
  `data/hf_gov_cache_export.zip` unzips to 12 files.

## Optional, but it helps reuse

Add a short description and topic tags on each repository: `green-ai`,
`quantization`, `energy-measurement`, `model-cards`, `reproducibility`. People
find replication packages through topics far more often than through search.
