# Earlier versions

These notebooks are kept as a record of how the project developed. **Don't use them as a reference.** The final pipeline is
[`../mental_health_bert.ipynb`](../mental_health_bert.ipynb). Both ran locally (Apple Silicon / MPS).

| Notebook | What it shows | Reported result |
|---|---|---|
| `v0_leaky_oversampling.ipynb` | `RandomOverSampler` applied to the whole dataset **before** `train_test_split`, so copies of the same rows end up in both train and test | 0.97 accuracy (inflated by leakage) |
| `v1_split_before_oversampling.ipynb` | Same pipeline, but split first and oversample only the training set | 0.75 accuracy, 0.72 macro-F1 |

Notes, for transparency:

- To keep the files small, I cleared the outputs of three cells in each notebook: the `pip install` log and two cells
  that print raw token-ID lists. No other code or output was changed.
- In v0, the execution counts show that the evaluation cell ran **before** the training cell in that session, using a model loaded
  from `SAVE_DIR`. So the 0.97 report comes from a model trained earlier with the same leaky pipeline, not from the training run
  shown in the notebook. The training run that is shown points the same way: its lowest validation loss is 0.22, compared with 0.69 in v1.
- v0's split has no `random_state`, so it can't be reproduced exactly.
