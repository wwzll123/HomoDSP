# PWM error estimation

Each fold bundle contains the calibrated sample-level and position-level
regressors for the matching `template-dna` checkpoint. Each branch averages
five calibration partitions and three seeds (17, 29, 43). The position-level
branch first averages its estimates across scored positions on both strands.
The two branches receive equal weight. The CLI reports the mean over the five
base folds as `predicted_mean_fold_mae`.

The bundles include the frozen query-only HomoDSP (last epoch) and standard
DeepPBS (best epoch) parameters used to calculate auxiliary predictions.
DeepPBS fold order follows the upstream run suffixes 828, 529, 173, 898 and
820. Its protein-feature scaler reproduces the original external evaluation:
it was computed from the 33 candidate structures and is fixed for new inputs.
The standardization of these features uses protein atoms only. The UQ heads
were trained on the CV benchmark. [DeepPBS models and citation](https://github.com/timkartar/DeepPBS)

`template_iptm.tsv.gz` contains the template identifiers and iPTM scores used
for the coverage and confidence features. A query with zero retrieved templates
uses the learned preprocessing for missing template evidence.
