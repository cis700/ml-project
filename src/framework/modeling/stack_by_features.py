import time
import pandas as pd
from numpy.random.mtrand import RandomState
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class StackByFeatures:

    def __init__(self, X, y, top_n=3, test_size=0.3):
        # Instantiate ensemble algorithms for use in following sections
        self.rng = RandomState()
        self.AB = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, learning_rate=1.0)
        self.RF = RandomForestClassifier(n_estimators=10, criterion='entropy', max_features='auto', bootstrap=True)
        self.ET = ExtraTreesClassifier(n_estimators=10, criterion='gini', max_features='auto', bootstrap=False)
        self.GB = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=200, max_features='auto')
        self.DT = DecisionTreeClassifier()
        self.test_size = test_size
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.rng)
        self.top_n = top_n
        self.train_me()

    def train_me(self):
        print("Training Size: {:.2f}%  with Test Size: {:.2f}%".format((1-self.test_size)*100, self.test_size*100))
        print("--------------------")
        # Start training timer
        training_start = time.time()

        # ----- AdaBoost
        start = time.time()

        self.AB.fit(self.x_train, self.y_train)

        ab_feature = self.AB.feature_importances_
        print("ab_feature: {}".format(ab_feature))

        ab_score = self.AB.score(self.x_test, self.y_test)
        ab_predict = self.AB.predict(self.x_test)

        print("\n--------------------------------------------")
        print("AdaBoost Prediction Results: ")

        ab_accuracy_score = accuracy_score(ab_predict, self.y_test)
        ab_precision_score = precision_score(ab_predict, self.y_test, average='weighted')
        ab_recall_score = recall_score(ab_predict, self.y_test, average='weighted')
        ab_f1score = f1_score(ab_predict, self.y_test, average='weighted')

        runtime = time.time() - start

        print('Accuracy: \t\t\t{:.4f}'.format(ab_accuracy_score))
        print('Precision: \t\t\t{:.4f}'.format(ab_precision_score))
        print('Recall: \t\t\t{:.4f}'.format(ab_recall_score))
        print('F1: \t\t\t\t{:.4f}'.format(ab_f1score))

        # Record total modeling time for AdaBoost
        print('Initial - AdaBoost: \t\t{:.2f}% in {:.2f} seconds'.format(ab_score * 100, runtime))

        # ----- RandomForest, Base Set
        start = time.time()

        self.RF.fit(self.x_train, self.y_train)

        rf_feature = self.RF.feature_importances_
        print("self.rf_feature: {}".format(rf_feature))

        rf_score = self.RF.score(self.x_test, self.y_test)

        # RandomForest Prediction
        print("\n--------------------------------------------")
        print("Random Forest Prediction Results: ")
        rf_predict = self.RF.predict(self.x_test)

        rf_accuracy_score = accuracy_score(rf_predict, self.y_test)
        rf_precision_score = precision_score(rf_predict, self.y_test, average='weighted')
        rf_recall_score = recall_score(rf_predict, self.y_test, average='weighted')
        rf_f1score = f1_score(rf_predict, self.y_test, average='weighted')

        runtime = time.time() - start

        print('Accuracy: \t\t\t{:.4f}'.format(rf_accuracy_score))
        print('Precision: \t\t\t{:.4f}'.format(rf_precision_score))
        print('Recall: \t\t\t{:.4f}'.format(rf_recall_score))
        print('F1: \t\t\t\t{:.4f}'.format(rf_f1score))

        print('Initial - RandomForest: \t{:.2f}% in {:.2f} seconds'.format(rf_score * 100, runtime))


        # ----- ExtraTrees, Base Set
        start = time.time()

        self.ET.fit(self.x_train, self.y_train)

        et_feature = self.ET.feature_importances_
        print("et_feature: {}".format(et_feature))

        et_score = self.ET.score(self.x_test, self.y_test)
        et_predict = self.ET.predict(self.x_test)

        print("\n--------------------------------------------")
        print("ExtraTrees Prediction Results: ")

        et_accuracy_score = accuracy_score(et_predict, self.y_test)
        et_precision_score = precision_score(et_predict, self.y_test, average='weighted')
        et_recall_score = recall_score(et_predict, self.y_test, average='weighted')
        et_f1score = f1_score(et_predict, self.y_test, average='weighted')
        runtime = time.time() - start

        print('Accuracy: \t\t\t{:.4f}'.format(et_accuracy_score))
        print('Precision: \t\t\t{:.4f}'.format(et_precision_score))
        print('Recall: \t\t\t{:.4f}'.format(et_recall_score))
        print('F1: \t\t\t\t{:.4f}'.format(et_f1score))

        print('Initial - ExtraTrees: \t\t{:.2f}% in {:.2f} seconds'.format(et_score * 100, runtime))

        # -- DT, Base Set
        start = time.time()

        self.DT.fit(self.x_train, self.y_train)

        dt_feature = self.DT.feature_importances_
        print("dt_feature: {}".format(et_feature))
        dt_score = self.DT.score(self.x_test, self.y_test)
        dt_predict = self.DT.predict(self.x_test)

        print("\n--------------------------------------------")
        print("DecisionTree Prediction Results: ")

        dt_accuracy_score = accuracy_score(dt_predict, self.y_test)
        dt_precision_score = precision_score(dt_predict, self.y_test, average='weighted')
        dt_recall_score = recall_score(dt_predict, self.y_test, average='weighted')
        dt_f1score = f1_score(dt_predict, self.y_test, average='weighted')

        print('Accuracy: \t\t\t{:.4f}'.format(dt_accuracy_score))
        print('Precision: \t\t\t{:.4f}'.format(dt_precision_score))
        print('Recall: \t\t\t{:.4f}'.format(dt_recall_score))
        print('F1: \t\t\t\t{:.4f}'.format(dt_f1score))

        runtime = time.time() - start
        print('Initial - DecisionTree: \t\t{:.2f}% in {:.2f} seconds'.format(et_score * 100, runtime))

        # ----- GradientBoost, Base Set
        start = time.time()

        self.GB.fit(self.x_train, self.y_train)

        gb_feature = self.GB.feature_importances_
        print("GB_feature: {}".format(gb_feature))

        gb_score = self.GB.score(self.x_test, self.y_test)
        gb_predict = self.GB.predict(self.x_test)

        print("\n--------------------------------------------")
        print("GradientBoost Prediction Results: ")

        gb_accuracy_score = accuracy_score(gb_predict, self.y_test)
        gb_precision_score = precision_score(gb_predict, self.y_test, average='weighted')
        gb_recall_score = recall_score(gb_predict, self.y_test, average='weighted')
        gb_f1score = f1_score(gb_predict, self.y_test, average='weighted')

        print('Accuracy: \t\t\t{:.4f}'.format(gb_accuracy_score))
        print('Precision: \t\t\t{:.4f}'.format(gb_precision_score))
        print('Recall: \t\t\t{:.4f}'.format(gb_recall_score))
        print('F1: \t\t\t\t{:.4f}'.format(gb_f1score))

        runtime = time.time() - start

        print('Initial - GradientBoost: \t{:.2f}% in {:.2f} seconds'.format(gb_score * 100, runtime))

        # Create collection of all features from tests
        cols = self.x_train.columns.values
        feature_df = pd.DataFrame({'features': cols,
                                   'AdaBoost': ab_feature,
                                   'RandomForest': rf_feature,
                                   'ExtraTree': et_feature,
                                   'GradientBoost': gb_feature
                                   })
        print(feature_df.head(8))

        graph = feature_df.plot.bar(figsize=(18, 10), title='Feature distribution', grid=True, legend=True, fontsize=15,
                                    xticks=feature_df.index)
        graph.set_xticklabels(feature_df.features, rotation=80)

        top_n = self.top_n

        # Grab the nlargest features (by score) from each ensemble group
        a_f = feature_df.nlargest(top_n, 'AdaBoost')
        e_f = feature_df.nlargest(top_n, 'ExtraTree')
        g_f = feature_df.nlargest(top_n, 'GradientBoost')
        r_f = feature_df.nlargest(top_n, 'RandomForest')

        print("AdaBoost Highest: {}".format(a_f))
        print("ExtraTree Highest: {}".format(e_f))
        print("GradientBoost Highest: {}".format(g_f))
        print("RandomForest Highest: {}".format(r_f))

        # Concat the top nlargest scores from all groups into one list
        result = pd.concat([a_f, e_f, g_f, r_f])

        # Drop duplicate fields from the list
        result = result.drop_duplicates()
        print("Dropped Duplicate Fields: {}".format(result))

        # selected_features contains the ensemble results for best features
        selected_features = result['features'].values.tolist()
        print("Selected Features: {}".format(selected_features))

        # Load new datasets for use in correlation analysis
        x_train_ens = self.x_train[selected_features]

        x_test_ens = self.x_test[selected_features]

        # Record total training time
        training_runtime = time.time() - training_start
        print('Total Training Time: \t\t{:.2f} seconds'.format(training_runtime))
        print("--------------------")

        # ****************************************************************
        # 3. Modeling
        # ****************************************************************
        # Run final models based on dataset produced by correlation
        # analysis.

        print("Modeling")
        print("--------------------")

        # Start modeling timer
        model_start1 = time.time()

        # ----- DecisionTree, Final
        self.DT.fit(x_train_ens, self.y_train)

        dt_predict = self.DT.predict(x_test_ens)
        dt_accuracy_score = accuracy_score(dt_predict, self.y_test)
        dt_precision_score = precision_score(dt_predict, self.y_test, average='weighted')
        dt_recall_score = recall_score(dt_predict, self.y_test, average='weighted')
        dt_f1score = f1_score(dt_predict, self.y_test, average='weighted')

        print('Accuracy: \t\t\t{:.4f}'.format(dt_accuracy_score))
        print('Precision: \t\t\t{:.4f}'.format(dt_precision_score))
        print('Recall: \t\t\t{:.4f}'.format(dt_recall_score))
        print('F1: \t\t\t\t{:.4f}'.format(dt_f1score))

        # Record total modeling time
        dt_model_runtime = time.time() - model_start1
        print('DT Total Modeling Time: \t\t{:.2f} seconds\n'.format(dt_model_runtime))
        print("--------------------")

        # Start modeling timer
        model_start = time.time()

        # ----- AdaBoostClassifier, Final
        self.AB.fit(x_train_ens, self.y_train)

        ab_predict = self.AB.predict(x_test_ens)
        ab_accuracy_score = accuracy_score(gb_predict, self.y_test)
        ab_precision_score = precision_score(gb_predict, self.y_test, average='weighted')
        ab_recall_score = recall_score(gb_predict, self.y_test, average='weighted')
        ab_f1score = f1_score(gb_predict, self.y_test, average='weighted')

        print('Accuracy: \t\t\t{:.4f}'.format(ab_accuracy_score))
        print('Precision: \t\t\t{:.4f}'.format(ab_precision_score))
        print('Recall: \t\t\t{:.4f}'.format(ab_recall_score))
        print('F1: \t\t\t\t{:.4f}'.format(ab_f1score))

        # Record total modeling time
        ab_model_runtime = time.time() - model_start
        print('AdaBoost Total Modeling Time: \t\t{:.2f} seconds\n'.format(ab_model_runtime))
        print("--------------------")

        # Start modeling timer
        model_start = time.time()

        # ----- ExtraTrees Classifier, Final
        start = time.time()

        self.ET.fit(x_train_ens, self.y_train)

        et_predict = self.ET.predict(x_test_ens)
        et_accuracy_score = accuracy_score(et_predict, self.y_test)
        et_precision_score = precision_score(et_predict, self.y_test, average='weighted')
        et_recall_score = recall_score(et_predict, self.y_test, average='weighted')
        et_f1score = f1_score(et_predict, self.y_test, average='weighted')

        print('Accuracy: \t\t\t{:.4f}'.format(et_accuracy_score))
        print('Precision: \t\t\t{:.4f}'.format(et_precision_score))
        print('Recall: \t\t\t{:.4f}'.format(et_recall_score))
        print('F1: \t\t\t\t{:.4f}'.format(et_f1score))

        # Record total modeling time
        et_model_runtime = time.time() - model_start
        print('ExtraTrees Total Modeling Time: \t\t{:.2f} seconds\n'.format(et_model_runtime))
        print("--------------------")

        # Start modeling timer
        model_start = time.time()

        # ----- GradientBoostClassifier, Final
        self.GB.fit(x_train_ens, self.y_train)

        gb_predict = self.GB.predict(x_test_ens)
        gb_accuracy_score = accuracy_score(gb_predict, self.y_test)
        gb_precision_score = precision_score(gb_predict, self.y_test, average='weighted')
        gb_recall_score = recall_score(gb_predict, self.y_test, average='weighted')
        gb_f1score = f1_score(gb_predict, self.y_test, average='weighted')

        print('Accuracy: \t\t\t{:.4f}'.format(gb_accuracy_score))
        print('Precision: \t\t\t{:.4f}'.format(gb_precision_score))
        print('Recall: \t\t\t{:.4f}'.format(gb_recall_score))
        print('F1: \t\t\t\t{:.4f}'.format(gb_f1score))

        # Record total modeling time
        gb_model_runtime = time.time() - model_start
        print('GB Total Modeling Time: \t\t{:.2f} seconds\n'.format(gb_model_runtime))
        print("--------------------")

        # Start modeling timer
        model_start = time.time()

        # ----- GradientBoostClassifier, Final
        self.RF.fit(x_train_ens, self.y_train)

        rf_predict = self.RF.predict(x_test_ens)
        rf_accuracy_score = accuracy_score(rf_predict, self.y_test)
        rf_precision_score = precision_score(rf_predict, self.y_test, average='weighted')
        rf_recall_score = recall_score(rf_predict, self.y_test, average='weighted')
        rf_f1score = f1_score(rf_predict, self.y_test, average='weighted')

        print('Accuracy: \t\t\t{:.4f}'.format(rf_accuracy_score))
        print('Precision: \t\t\t{:.4f}'.format(rf_precision_score))
        print('Recall: \t\t\t{:.4f}'.format(rf_recall_score))
        print('F1: \t\t\t\t{:.4f}'.format(rf_f1score))

        # Record total modeling time
        rf_model_runtime = time.time() - model_start
        print('self.RF Total Modeling Time: \t\t{:.2f} seconds'.format(rf_model_runtime))
        print("--------------------")

