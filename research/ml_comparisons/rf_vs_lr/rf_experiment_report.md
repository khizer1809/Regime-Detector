# Random Forest vs Logistic Regression -- HMM-4 Downstream Classifier Comparison

RF params: `{'n_estimators': 300, 'max_depth': 8, 'min_samples_leaf': 50, 'min_samples_split': 100, 'max_features': 'sqrt', 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1}`

Runtime: 386.1s (estimated 297.6s from a single-fold benchmark before running).

## A. Did RF outperform LR? (OOS, identical eligible rows)

   metric           LR           RF
 log_loss     0.691607     0.698391
    brier     0.249223     0.252518
  roc_auc     0.522800     0.504400
   pr_auc     0.486601     0.470855
 accuracy     0.530028     0.497770
precision     0.622951     0.470920
   recall     0.000842     0.552879
       f1     0.001682     0.508619
        n 95978.000000 95978.000000

Delta OOS ROC-AUC (RF - LR): -0.0184

## B. Is RF overfitting?

Mean per-fold train AUC: 0.6016616447725901
Mean per-fold OOS AUC: 0.5033526719961473
Train-OOS gap: 0.0983089727764428

## C. HMM-4 vs HMM-6

NOT run automatically -- no existing walk-forward HMM-6 cache; would require a new, comparably expensive Stage-A build. Flagged as an opt-in follow-up, not silently skipped.

## Feature importance

       feature  importance  rank  permutation_importance_mean  permutation_importance_std
     stay_prob    0.534445     1                     0.029253                    0.009610
log1p_duration    0.230537     2                     0.021542                    0.009896
        margin    0.123813     3                     0.007154                    0.009167
    confidence    0.111205     4                    -0.003577                    0.008560

## Threshold analysis

model  threshold  n_signals  hit_rate  recall_of_positives
   LR       0.50         61  0.622951             0.000842
   LR       0.55          0       NaN                  NaN
   LR       0.60          0       NaN                  NaN
   LR       0.65          0       NaN                  NaN
   LR       0.70          0       NaN                  NaN
   LR       0.75          0       NaN                  NaN
   RF       0.50      52975  0.470920             0.552879
   RF       0.55      10683  0.469812             0.111232
   RF       0.60       3853  0.445886             0.038075
   RF       0.65       1648  0.448422             0.016378
   RF       0.70        342  0.409357             0.003103
   RF       0.75         40  0.350000             0.000310

## Leakage audit

All checks pass: True

- `hmm_unchanged`: {'note': 'Data/v2_cache/fold_XXX/ never opened for writing in this script', 'pass': True}
- `lr_unchanged`: {'note': 'lr_probability column reused verbatim from the frozen master file, LR never re-fit', 'pass': True}
- `target_unchanged`: {'note': 'label/usable/epsilon columns reused verbatim, never recomputed', 'pass': True}
- `rf_pool_train_only`: {'note': 'RF fit only on pool_usable (folds strictly before the current fold), predictions made only after fit completes', 'pass': True}
- `no_scaler_fit_on_test`: {'note': 'RF uses no scaler (tree-invariant); N/A', 'pass': True}
- `same_walkforward_folds_as_lr`: {'note': 'same `fold`/`usable`/`is_trending_t` columns as the LR pipeline, identical test periods by construction', 'pass': True}
- `feature_importance_not_used_for_selection`: {'note': 'importance computed AFTER all walk-forward folds completed, never fed back into fold training', 'pass': True}
- `permutation_importance_isolated`: {'note': "computed only on fold 100's own eligible OOS rows, using the model already fit on the full training pool -- descriptive only, not used to alter any fold's training", 'pass': True}
