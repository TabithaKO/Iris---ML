# These are all the libraries that we'll need
# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# load the dataset in csv form or from an online source
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
url = "~/Downloads/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)


# Get the summary of the dataset
# 1. Dimensions of the dataset.
#2. Peek at the data itself.
#3. Statistical summary of all attributes.
#4. Breakdown of the data by the class variable.


# Dimensions of a dataset	
# shape
print(dataset.shape)


# This allows you to see the first 20 rows of the dataset
# head
print(dataset.head(20))

# This includes the count, mean, max values of the various features
# descriptions
print(dataset.describe())

# We get to see how the dataset is distributed across the various classes
# class distribution
print(dataset.groupby('class').size())


# There are numerous ways to visualize the data
# The method below is using a box and whiskers plot
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# # Alternatively I can use a histogram 
# Ive edited this specific code a little since the original wasn't working out
dataset.plot(kind='hist', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# I can also use a scatter matrix and figure out the correlations between the various features. Diagonal grouping of some pairs suggests a high correlation and predictable relationship.
# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

# I need to do a lot more research on how to display the data using scipy software

# ***********************************************************
# So now we're working with algorithms. 
# Step1: Split our data into train(80%) and validation(20%)
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


# Let's try out some learning algorithms
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

