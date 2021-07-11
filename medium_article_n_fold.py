def train_model(clf,x,y, n_fold = 5):
    x = x.values
    y = y.values.ravel()
    skf = StratifiedKFold(n_splits= n_fold)


    for train_index, test_index in skf.split(x, y):
        #print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test)
        acc = accuracy_score(y_test, y_hat)

        p_score = precision_score(y_test, y_hat,average='macro')
        r_score = recall_score(y_test, y_hat,average='macro')
        f1_s = f1_score(y_test, y_hat,average='macro')
        print("accuracy score",acc)
        print("precision is {}| recall is {}| f1 score is {}".format(p_score,r_score,f1_s))
    return clf, X_test, y_test
