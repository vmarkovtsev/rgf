from sklearn import datasets
from sklearn.utils.validation import check_random_state
from rgf.sklearn import RGFClassifier

iris = datasets.load_iris()
rng = check_random_state(0)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

clf = RGFClassifier()
clf.fit(iris.data, iris.target)
#score = clf.score(iris.data, iris.target)
