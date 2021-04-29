#Plotter class for various data exploration techniques
class Visualizations():
    #Class Constructor
    def __init__(self, theme="white", fontSize=12):
        self.np = __import__("numpy")
        self.mpl = __import__("matplotlib")
        self.sns = __import__("seaborn")
        self.plt = self.mpl.pyplot
        self.sns.set_theme(style=theme)
        self.fontSize = fontSize
        self.cmap = self.sns.diverging_palette(10, 250, s=90, l=33, as_cmap=True)

    #Method to plot counts/ histogram
    def plotCounts(self, xData, figSize, plotTitle):
        self.plt.figure(figsize=figSize)
        self.sns.countplot(x=xData)
        self.plt.title(plotTitle, fontsize=self.fontSize)
        self.plt.show()

    def plotHistograms(self, data, figSize, bins=50):
        data.hist(figsize=figSize, bins=bins)
        self.plt.show()

    def plotBoxPlots(self, data, figSize, plotTitle):
        self.plt.figure(figsize=figSize)
        ax = self.sns.boxplot(data=data, orient="h")
        ax.set_title(plotTitle, fontsize=self.fontSize)
        ax.set(xscale="log")
        self.plt.show()

    def plotCorrelationMatrix(self, data, figSize, plotTitle):
        corrMatrix = data.corr()
        maskMatrix = self.np.triu(self.np.ones_like(corrMatrix, dtype=bool))
        self.plt.figure(figsize=figSize)
        self.sns.heatmap(corrMatrix, mask=maskMatrix, cmap=self.cmap, annot=False)
        self.plt.title(plotTitle, fontsize=self.fontSize)
        self.plt.show()
    
    def plotConfusionMatrix(self, data, labels, figSize, plotTitle):
        self.plt.figure(figsize=figSize)
        self.sns.heatmap(data, cmap=self.cmap, xticklabels=labels, yticklabels=labels, annot=True, fmt=".1%")
        self.plt.title(plotTitle, fontsize=self.fontSize)
        self.plt.show()
