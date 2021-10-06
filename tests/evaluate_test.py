import evaluate

overlap = evaluate.check_for_unequal_word_overlap("food", "food")
if overlap is True:
    raise AssertionError("check for food vs food failed")

overlap = evaluate.check_for_unequal_word_overlap("indian food", "food")
if overlap is False:
    raise AssertionError("check for indian food vs food failed")

overlap = evaluate.check_for_unequal_word_overlap("food", "indian food")
if overlap is False:
    raise AssertionError("food vs indian food failed")

overlap = evaluate.check_for_unequal_word_overlap("american food items", "indian food")
if overlap is False:
    raise AssertionError("american food items vs indian food failed")