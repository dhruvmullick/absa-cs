import numpy as np
import matplotlib.pyplot as plt

BASELINE_EPOCH_RESULTS_FILE_PATH = '/Users/dhruvmullick/Projects/absa-cs/Results/AmbiguousDataset8_ALSC/by_epoch/baseline_epochs.csv'
COMMONGEN_0_1_EPOCH_RESULTS_FILE_PATH = '/Users/dhruvmullick/Projects/absa-cs/Results/AmbiguousDataset8_ALSC/by_epoch/commongen_0.1_epochs.csv'
COMMONGEN_0_2_EPOCH_RESULTS_FILE_PATH = '/Users/dhruvmullick/Projects/absa-cs/Results/AmbiguousDataset8_ALSC/by_epoch/commongen_0.2_epochs.csv'
COMMONGEN_0_5_EPOCH_RESULTS_FILE_PATH = '/Users/dhruvmullick/Projects/absa-cs/Results/AmbiguousDataset8_ALSC/by_epoch/commongen_0.5_epochs.csv'
COMMONGEN_1_EPOCH_RESULTS_FILE_PATH = '/Users/dhruvmullick/Projects/absa-cs/Results/AmbiguousDataset8_ALSC/by_epoch/commongen_1.0_epochs.csv'
WIKITEXT_0_1_EPOCH_RESULTS_FILE_PATH = '/Users/dhruvmullick/Projects/absa-cs/Results/AmbiguousDataset8_ALSC/by_epoch/wikitext_0.1_epochs.csv'
COSMOS_0_1_EPOCH_RESULTS_FILE_PATH = '/Users/dhruvmullick/Projects/absa-cs/Results/AmbiguousDataset8_ALSC/by_epoch/cosmos_0.1_epochs.csv'
COSMOS_0_2_EPOCH_RESULTS_FILE_PATH = '/Users/dhruvmullick/Projects/absa-cs/Results/AmbiguousDataset8_ALSC/by_epoch/cosmos_0.2_epochs.csv'
COSMOS_0_5_EPOCH_RESULTS_FILE_PATH = '/Users/dhruvmullick/Projects/absa-cs/Results/AmbiguousDataset8_ALSC/by_epoch/cosmos_0.5_epochs.csv'
COSMOS_1_EPOCH_RESULTS_FILE_PATH = '/Users/dhruvmullick/Projects/absa-cs/Results/AmbiguousDataset8_ALSC/by_epoch/cosmos_1.0_epochs.csv'
SQUAD_0_5_EPOCH_RESULTS_FILE_PATH = '/Users/dhruvmullick/Projects/absa-cs/Results/AmbiguousDataset8_ALSC/by_epoch/squad_0.5_epochs.csv'

MAX_EPOCH = 20

def display_results(file_path, colour, label):
    epoch_results_file = open(file_path, 'r')
    results = {}
    for line in epoch_results_file:
        seed, epoch, f1 = line.split(',')
        f1 = float(f1.strip())
        if epoch not in results:
            results[epoch] = []
        results[epoch].append(f1)
    means = []
    stddev = []
    for epoch in results:
        f1_list = np.array(results[epoch])
        means.append(np.mean(f1_list))
        stddev.append(np.std(f1_list))
    means = np.array(means)
    stddev = np.array(stddev)

    means = means[:MAX_EPOCH]
    stddev = stddev[:MAX_EPOCH]
    epochs = list(results.keys())[:MAX_EPOCH]

    plt.plot(epochs, means, color=colour, label=label)
    plt.fill_between(epochs, (means - stddev), (means + stddev), color=colour, alpha=0.1)


### Best Ones
# display_results(BASELINE_EPOCH_RESULTS_FILE_PATH, 'blue', 'Baseline')
# display_results(COMMONGEN_0_1_EPOCH_RESULTS_FILE_PATH, 'red', 'Commongen 0.1')
# display_results(COSMOS_1_EPOCH_RESULTS_FILE_PATH, 'purple', 'Cosmos 1.0')
# display_results(WIKITEXT_0_1_EPOCH_RESULTS_FILE_PATH, 'black', 'Wikitext 0.1')
# display_results(SQUAD_0_5_EPOCH_RESULTS_FILE_PATH, 'magenta', 'Squad 0.5')

### Cosmos
# display_results(COSMOS_0_1_EPOCH_RESULTS_FILE_PATH, 'black', 'Cosmos 0.1')
# display_results(COSMOS_0_2_EPOCH_RESULTS_FILE_PATH, 'blue', 'Cosmos 0.2')
# display_results(COSMOS_0_5_EPOCH_RESULTS_FILE_PATH, 'red', 'Cosmos 0.5')
display_results(COSMOS_1_EPOCH_RESULTS_FILE_PATH, 'purple', 'Cosmos 1.0')

### Commongen
# display_results(COMMONGEN_0_1_EPOCH_RESULTS_FILE_PATH, 'black', 'Commongen 0.1')
# display_results(COMMONGEN_0_2_EPOCH_RESULTS_FILE_PATH, 'blue', 'Commongen 0.2')
# display_results(COMMONGEN_0_5_EPOCH_RESULTS_FILE_PATH, 'red', 'Commongen 0.5')
# display_results(COMMONGEN_1_EPOCH_RESULTS_FILE_PATH, 'purple', 'Commongen 1.0')


plt.legend(loc='lower right')
plt.xticks(rotation=90)
plt.show()
