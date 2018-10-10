from sklearn import datasets
from rgf.sklearn import RGFClassifier

iris = datasets.load_iris()

clf = RGFClassifier()
clf.fit(iris.data, iris.target)
