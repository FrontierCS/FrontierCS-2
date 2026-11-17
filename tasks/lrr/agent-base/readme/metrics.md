- `train_vulnerability_elimination_rate`: Share of train regexes whose repair has a completed
  detector result with no successful attack, regardless of semantics.
- `train_repair_rate`: Share of train regexes with a credited repair.
- `train_well_formed_rate`: Share of train regexes with a well-formed repair.
- `train_mean_rli`: Mean relative length increase of the repairs over all train regexes; lower
  is better.
- `train_mean_nls`: Mean normalized Levenshtein similarity of the repairs over all train
  regexes.
- `train_mean_jaccard`: Mean Xeger Jaccard estimate of the repairs over all train regexes.
- `train_mean_precision`: Mean Xeger precision estimate of the repairs over all train regexes.
- `train_mean_recall`: Mean Xeger recall estimate of the repairs over all train regexes.
