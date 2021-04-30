
#Feature selection class to eliminate multicollinearity
class MCE():
    #Class Constructor
    def __init__(self):
        self.pd = __import__("pandas")

    #Method to create and return the feature correlation matrix dataframe
    def createCorrelationMatrix(self, X, y, include_target=False):
        #Checking we should include the target in the correlation matrix
        df_temp = X
        if (include_target == True):
            df_temp = self.pd.concat([X, y], axis=1)
        
        #Setting method to Pearson to prevent issues in case the default method for df.corr() gets changed
        #Setting min_period to 30 for the sample size to be statistically significant (normal) according to 
        #central limit theorem
        corrMatrix = df_temp.corr(method="pearson", min_periods=30).abs()

        #Creating the required dataframe, then dropping the target row 
        #and sorting by the value of correlation with target (in asceding order)
        if(include_target == True):
            corrMatrix = self.pd.DataFrame(corrMatrix.loc[:, y.name]).drop([y.name], axis = 0).sort_values(by=y.name)
        return (corrMatrix)

    #Method to create and return the list of correlated features
    def createCorrelatedFeaturesList(self, X, y, threshold):
        #Obtaining the correlation matrix of the dataframe (without the target)
        corrMatrix = self.createCorrelationMatrix(X, y, include_target=False)
        colCorr = []
        #Iterating through the columns of the correlation matrix dataframe
        for column in corrMatrix.columns:
            #Iterating through the values (row wise) of the correlation matrix dataframe
            for idx, row in corrMatrix.iterrows():
                if(row[column] > threshold) and (row[column] < 1):
                    #Adding the features that are not already in the list of correlated features
                    if (idx not in colCorr):
                        colCorr.append(idx)
                    if (column not in colCorr):
                        colCorr.append(column)
        #print(colCorr, '\n')
        return colCorr

    #Method to eliminate the least important features from the list of correlated features
    def deleteFeatures(self, X, y, colCorr):
        #Obtaining the feature to target correlation matrix dataframe
        corrWithTarget = self.createCorrelationMatrix(X, y, include_target=True)
        for idx, row in corrWithTarget.iterrows():
            #print(idx, '\n')
            if (idx in colCorr):
                X = X.drop(idx, axis=1)
                break
        return (X)

    #Method to run automatically eliminate multicollinearity
    def autoEliminate(self, X, y, threshold):
        #Obtaining the list of correlated features
        colCorr = self.createCorrelatedFeaturesList(X, y, threshold)
        while colCorr != []:
            #Obtaining the dataframe after deleting the feature (from the list of correlated features) 
            #that is least correlated with the target
            X = self.deleteFeatures(X, y, colCorr)
            #Obtaining the list of correlated features
            colCorr = self.createCorrelatedFeaturesList(X, y, threshold)   
        return (X)