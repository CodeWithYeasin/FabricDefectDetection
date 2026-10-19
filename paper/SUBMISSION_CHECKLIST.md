# Before submitting

The manuscript is a full draft, but it cannot be submitted until the experiments have run. Work
through this list in order.

## 1. Run the experiments
- [ ] `notebooks/run_experiments_colab.ipynb` end to end (or `scripts/run_all.sh`)
- [ ] Open `data/dedup_report.json` and look at 30 random rows of `data/duplicate_pairs.csv` by eye.
      If many pairs are not real duplicates, lower `--phash-thr` and rerun `prepare_data.py`
      **before** training anything.
- [ ] Open `data/name_label_conflicts.csv` (e.g. `line_*` files in the *hole* folder), look at each
      image and decide hole or lines. If you relabel, move the files in a copy of the dataset,
      rerun `prepare_data.py`, and say so in the paper (Section 3.1)
- [ ] `results/shortcut_check.json`: metadata-only macro-F1 should be high, protocol-input macro-F1 low
- [ ] `results/summary.json` contains every experiment with 5 runs each
- [ ] Optional but strongly recommended: one external dataset with `scripts/cross_dataset.py`

## 2. Check the story against the numbers
The method section makes predictions. Check each one honestly and rewrite the text if the data
disagree.
- [ ] SpecTexNet beats its own backbone without the spectral branch (`abl_no_spectral`)
- [ ] The gain is largest on hole / horizontal / vertical and smallest on stain
- [ ] OA-Rot beats both flips-only and naive rotation
- [ ] McNemar p < 0.05 against the strongest baseline. If not, write "comparable to" and lean on
      efficiency, calibration or explainability instead.

If SpecTexNet does not beat the strongest fine-tuned baseline, the paper can still be publishable
as an evaluation-protocol paper (leakage, metrics, the v1 drop) with OA-Rot as the method
contribution. Do not change the protocol after seeing test results to rescue a claim.

## 3. Fill the manuscript
- [ ] Every red `\todo{}` in `main.tex` is resolved (search for `TODO`)
- [ ] No red `XX` remains (run `aggregate.py` and recompile)
- [ ] Related work: add 5 to 8 fabric-defect papers from 2023 to 2026, ideally from the target journal
- [ ] Verify every reference against the publisher page (authors, pages, DOI)
- [ ] Authors, affiliations, CRediT statement, GitHub URL
- [ ] Download `elsarticle` from the journal page if your TeX install lacks it (Overleaf has it)

## 4. Target journals (Q1, applied AI / industrial vision)
Pick one and follow its guide for authors: Engineering Applications of Artificial Intelligence,
Expert Systems with Applications, Computers in Industry, Journal of Intelligent Manufacturing,
Textile Research Journal. Check the current quartile in Scimago for your field before choosing.
