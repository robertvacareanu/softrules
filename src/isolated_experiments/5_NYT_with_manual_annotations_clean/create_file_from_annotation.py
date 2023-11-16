"""

"""

import glob
import pandas as pd

if __name__ == "__main__":
    tsv_files = glob.glob('src/isolated_experiments/5_NYT_with_manual_annotations_clean/data/*.tsv')
    all_data = []
    for f in tsv_files:
        data = pd.read_csv(f, sep='\t')
        data = data[data['MS'] == 1]
        data['relation'] = f.split('/')[-1].split('.tsv')[0]
        all_data.append(data)

    data = pd.concat(all_data)