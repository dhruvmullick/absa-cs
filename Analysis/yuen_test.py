from Results.AmbiguousDataset8_ALSC.topk_alsc import *

import scipy.stats

t, p = scipy.stats.ttest_ind(result_baseline, result_commongen_0_1, equal_var=False, trim=0.1)
print(p)