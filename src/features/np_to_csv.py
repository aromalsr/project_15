import numpy as np
import os
import pandas as pd
import glob



def data_extraction(filepath: str) -> None:
    """
            A function that calculates various statistical features from the data stored in .npy files in the given folder path.

            Parameters:
                - filepath: A string representing the path to the folder containing .npy files.

            Returns:
                - No direct return value. Saves the calculated features and labels to a CSV file.
    """
    
    files = list(filter(os.path.isfile, glob.glob(filepath + "/**/*.npy")))
    
    #(filepath + "/**/**/*.npy")

    
    features = []
    labels = []
    
    
    # Loop through all the .npy files in the folder
    for file in files:
        if file.endswith('.npy'):

            # Load the numpy file
            data = np.load(file)
            
            # split the data into intensity and intensity diff
            sample= file.split('\\')[-1]
            sample=sample.split('_')[0]
            
            print(sample)
            
            # Add the label to the features based on the sample name wheather it is plastic (P) or food (F)
            
            food_samples = ['S001','S002','S003','S004','S005','S006','S007','S008','S009','S024']
            
            if sample in food_samples:
                cato = 'F'
            else:
                cato = 'P'
            
            print(file)
            data = data[1, :, :]
            
            # Calculate features (mean, median, standard deviation, variance, range, IQR)
            name =  sample
            mean = np.mean(data)
            median = np.median(data)
            std_dev = np.std(data)
            variance = np.var(data)
            data_range = np.ptp(data)
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            
            # Append the features and label to the lists
            features.append([name, mean, median, std_dev, variance, data_range, iqr])
            labels.append("plastic" if cato == 'P' else "food")
         
    # Convert features and labels to NumPy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    # Scale the features
    # scaler = StandardScaler()
    # features_scaled = scaler.fit_transform(features)
            
    # Save the features and labels to a CSV file
    df = pd.DataFrame(features, columns=["name", "mean", "median", "std_dev", "variance", "range", "iqr"])
    
    df["label"] = labels
    
    # get name of csv file
    csv_name  = input('enter there name of csv file : ')
    
    # save to csv or xlsx
    df.to_csv("data/processed/"+csv_name+".csv", index=False)      
        
    #df.to_excel("project_marvel/data/processed/"+CsvName+".xlsx", index=False)
