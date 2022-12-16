import cv2 as cv
import numpy as np
from pathlib import Path
import pandas as pd
import os


from extract_features import gabor_extractor



if __name__ == "__main__":

    output_file = 'static/features/gabor_features.csv'
    c = 1
    all_files = os.listdir('static/images/') 
    

	#loop through images and extract features
    for img_path in sorted(Path("./static/images").glob('*.png')):
        #imageId = img_path[img_path.rfind("/")+1:]
        image = cv.imread(img_path.as_posix())

        features = gabor_extractor(image)
        features = [str(f) for f in features]
		# print("c = {}".format(c))
        c += 1
        with open(output_file, 'a', encoding="utf8") as f:
            f.write("%s,%s\n" % (img_path, ",".join(features)))
            f.close()
    
    '''    #read images from database
    feature_df=pd.DataFrame(columns=create_labels())
    #print(feature_df.columns)
    for img_path in sorted(Path("./static/images").glob('*.png')):
        features = apply_filter(img=cv.imread(img_path.as_posix()))
        #features.insert(0,cv.imread(img_path.as_posix()))

        
        features.insert(0,cv.imread(img_path.as_posix()))

        feature_df.loc[len(feature_df)] = features
    
    print(feature_df.iloc[1,1])'''
    
        




    


           