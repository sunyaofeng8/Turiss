import pandas as pd

csvfile = pd.read_csv('test_set.csv')
csvfile.to_csv('test_set.tsv', sep='\t', index=None)
