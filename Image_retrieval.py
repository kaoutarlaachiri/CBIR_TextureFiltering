import numpy as np
import csv
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler


def run_search(query_features ):
	limit=10
	results = dict()
	path = 'static/features/gabor_features.csv'
	with open(path) as f:
		reader = csv.reader(f)
		for row in reader:
				#separiting the the image Name from features, and computing the chi-squared distance.
				features = [float(x) for x in row[1:]]
				query_features = [float(x) for x in row[1:]]
				dist = euclidean(query_features,features)

				results[row[0]] = dist
		f.close()
			
		#dictionarry sort
		results = sorted(
			[(v,k) for (k,v) in results.items()]
		)
		return results[:limit]

