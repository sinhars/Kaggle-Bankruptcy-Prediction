import sklearn


class Modelling():
    #Class Constructor
    def __init__(self, vis):
        self.np = __import__("numpy")
        self.sklearn = __import__("sklearn")
        self.metrics = sklearn.metrics
        self.vis = vis

    #Method to plot counts/ histogram
    def getModelPerformance(self, trueVals, preds, figSize, plotTitle, targetNames):
        confMatrix = self.metrics.confusion_matrix(trueVals, preds, normalize="true")
        report = self.metrics.classification_report(trueVals, preds, target_names=targetNames)
        print(report)
        self.vis.plotConfusionMatrix(confMatrix, labels=targetNames, figSize=figSize, plotTitle=plotTitle)