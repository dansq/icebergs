import pandas as pd
import numpy as np
from skimage import io, filters
from sklearn.svm import SVR

def main():
	train_df = pd.read_json('train.json')
	print('passou treino')

	x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75)
						for band in train_df["band_1"]])
	x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75)
						for band in train_df["band_2"]])
	X_train = np.concatenate([x_band1[:, :, :, np.newaxis],
							  x_band2[:, :, :, np.newaxis]], axis=-1)
	Y_train = np.array(train_df["is_iceberg"])
	vars = X_train.shape
	d2_train = X_train.reshape(vars[0], vars[1] * vars[2] * vars[3])

	print("Xtrain:", X_train.shape)
	print("Ytrain:", Y_train.shape)
	print(Y_train)
	
	filtered_images = np.empty(x_band1.shape)
	i = 0
	for band in x_band1:
		gaussian = filters.gaussian(band, preserve_range=True)
		threshold = filters.threshold_otsu(gaussian)
		mask = gaussian < threshold
		filtered_images[i] = mask
		i += 1

	vars = filtered_images.shape
	d2_train = filtered_images.reshape(vars[0], vars[1] * vars[2])
	clf = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2,
			  gamma='auto', kernel='rbf', max_iter=-1, shrinking=True,
			  tol=0.001, verbose=False)

	clf.fit(d2_train, Y_train)
	
	print('treinou')

	#
	response = huge_predict('test2.json', clf, chunksize=337)

	print()

	print(response)
	print(response.shape)
	
	np.savetxt('resultados.csv', response, delimiter=',')

def huge_predict(filepath, clf, chunksize=4):
    reader = pd.read_json(filepath, lines=True, chunksize=chunksize)
    prediction = np.array([])
    #i = 0
    for chunk in reader:
        x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75)
                            for band in chunk["band_1"]])
        x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75)
                            for band in chunk["band_2"]])
        X_test = np.concatenate([x_band1[:, :, :, np.newaxis],
                                 x_band2[:, :, :, np.newaxis]], axis=-1)
        test_vars = X_test.shape
        d2_test = X_test.reshape(test_vars[0], test_vars[1] *
                                 test_vars[2] * test_vars[3])
        filtered_images = np.empty(x_band1.shape)
        i = 0
        for band in x_band1:
            gaussian = filters.gaussian(band, preserve_range=True)
            threshold = filters.threshold_otsu(gaussian)
            mask = gaussian < threshold
            filtered_images[i] = mask
            i += 1
			
        vars = filtered_images.shape
        d2_test = filtered_images.reshape(vars[0], vars[1] * vars[2])
        response = clf.predict(d2_test)
        prediction = np.concatenate((prediction, response))
        #i = i + 1
        #if i > 1:
        #    break
    return prediction
	
main()
