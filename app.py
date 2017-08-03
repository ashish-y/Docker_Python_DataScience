from sklearn import tree
from sklearn import neighbors

#[size,weight, texture]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['apple', 'apple', 'orange', 'orange', 'apple', 'apple', 'orange', 'orange',
     'orange', 'apple', 'apple']

#classifier - DecisionTreeClassifier
clf_tree = tree.DecisionTreeClassifier();
clf_tree = clf_tree.fit(X,Y);

#classifier - neighbour
clf_neighbors = neighbors.KNeighborsClassifier();
clf_neighbors = clf_neighbors.fit(X,Y);

#test_data
test_data = [[190,70,42],[172,64,39],[182,80,42]];

#prediction
prediction_tree = clf_tree.predict(test_data);
prediction_neighbors = clf_neighbors.predict(test_data);

print("prediction of DecisionTreeClassifier:",prediction_tree);

print("prediction of Neighour:",prediction_neighbors);
