import time
import joblib
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

feat_to_use = []     # Indices of the features to use. If n is the number of features, from 0 to n-1

def load_features(filepath):
    ''' Load the features indices from a .txt file
       
        Attributes:
            filepath (string)   :  Path to the .txt file
    '''
    with open(filepath, 'r') as f:
        for line_index, line in enumerate(f.readlines()):
            tokens = line.strip().split(' ')
            if line_index == 0:
                global feat_to_use
                feat_to_use = [int(t) for t in tokens]

def read_model(filepath):
    ''' Read the Random Forest model from a .pkl file

        Attributes:
            filepath (string)   :   Path to the .pkl file
    '''
    with open(filepath, 'rb') as f:
        E = joblib.load(f)
        f.close()
    return E

def read_data(filepath):
    ''' Read the point cloud to classify from a .txt file

        Attributes:
            filepath (string)   :   Path to the .txt file
        
        Return:
            X (np.array)   :    Point cloud and features
    '''
    X = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split(',')
            if 'nan' not in tokens:   
                X.append([float(t) for t_index, t in enumerate(tokens)])
    return np.asarray(X, dtype=np.float32)

def write_classification(X, Y, filename):
    ''' Write a classified point cloud

        Attributes:
            X (np.array)        :   Point cloud and features
            Y (np.array)        :   Classes
            filename (string)   :   Output file path
    '''
    classes = {5: [0,251,20],4:[226,175,165],0:[222,150,200],1:[209,109,106],3:[212,212,212],2:[178,167,211],6:[212,231,205],7:[50,205,50],8:[172,238,238],9:[244,199,131],10:[252,225,198]}
    with open('{}.txt'.format(filename), 'w') as out:
        X = X.tolist()
        Y = Y.tolist()
        red = [classes[id][0] for id in Y]
        green = [classes[id][1] for id in Y]
        blue = [classes[id][2] for id in Y]
        for index, x in enumerate(X):
            x0 = str(x[0])
            x1 = str(x[1])
            x2 = str(x[2])
            Yg = str(int(x[-1]))
            out.write('{} {} {} {} {} {} {} {}\n'.format(x0, x1, x2, str(red[index]),  str(green[index]),  str(blue[index]), Yg, str(int(Y[index]))))

def main():
    parser = argparse.ArgumentParser(description='Classify a point cloud with a random forest model.')
    parser.add_argument('--features_filepath', default='featurefile.txt' ,help='Path to the file containing the index of the features and the class index')
    parser.add_argument('--model', default="D:\随机森林原始\新建文件夹\\ne1_msl1_mss2_md23_acc0.5696741561346023.pkl",help='Path to .pkl file containing the trained model.')
    parser.add_argument('--point_cloud', default='refined.txt',help='Path to .txt file containing the point cloud to classify.')
    parser.add_argument('--output_name', default='scanpred1treesmd23',help='Name of the predicted test file')
    args = parser.parse_args()

    start = time.time() 
    print ('Loading data ...')
    load_features(args.features_filepath)                   # Load feature indices
    model = read_model(args.model)                          # Load trained model
    X = read_data(args.point_cloud)                         # Load data to classify
    
    print ('Classifying the dataset ...')
    
    Y_pred = model.predict(X[:, feat_to_use])               # Classify the data

    print ('Saving ...')
    write_classification(X, Y_pred, args.output_name)       # Output classification
    end = time.time()
    print('Data classified in: {}'.format(end - start))


if __name__== '__main__':
    main()