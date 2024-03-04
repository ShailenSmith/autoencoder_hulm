def perm_test(sep_preds, dsep_preds, labels):
    # variables: sep_preds, dsep_preds, labels
    # HYP: dsep is better than sep, i.e., dsep_ppl < sep_ppl
    # NULL HYP: dsep_ppl >= sep_ppl

    NUM_TRIALS=1000
    diffs = []
    for i in range(NUM_TRIALS):
        idxs = np.sample(n=len(sep_preds), with_replacement=True)
        sep_ppl = calc_ppl(sep_preds[idxs], labels[idxs])
        dsep_ppl = calc_ppl(dsep_preds[idxs], labels[idxs])
        diffs.append(dsep_ppl - sep_ppl)

    p = sum([1 if d>=0 for d in diffs])/NUM_TRIALS
    return p