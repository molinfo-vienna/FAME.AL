"""
classifiers

Created by: Martin Sicho
On: 3/24/17, 11:30 AM

revised by: Ya Chen
- change parameters for new version 2022
- add MLP classifier
"""

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier

class ThresholdExtraTreesClassifier(ExtraTreesClassifier):
    def __init__(self,
                 n_estimators=100,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0.0, 
                 max_samples=None,
                 decision_threshold=0.5):
        super(ThresholdExtraTreesClassifier, self).__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples
        )
        self.decision_threshold = decision_threshold

    def predict(self, X):
        predictions = super(ThresholdExtraTreesClassifier, self).predict(X)
        probabilities = self.predict_proba(X)[:, 1]

        pos_class = self.classes_[1]
        neg_class = self.classes_[0]
        for idx, proba in enumerate(probabilities):
            if proba >= self.decision_threshold:
                predictions[idx] = pos_class
            else:
                predictions[idx] = neg_class

        return predictions

class ThresholdRandomForestClassifier(RandomForestClassifier):
    def __init__(self,
                 n_estimators=100,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="sqrt",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None, #means 1 unless in a joblib.parallel_backend context. -1 means using all processors
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0.0,
                 max_samples=None,
                 decision_threshold=0.5):
        super(ThresholdRandomForestClassifier, self).__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples
        )
        self.decision_threshold = decision_threshold

    def predict(self, X):
        predictions = super(ThresholdRandomForestClassifier, self).predict(X)
        probabilities = self.predict_proba(X)[:, 1]

        pos_class = self.classes_[1]
        neg_class = self.classes_[0]
        for idx, proba in enumerate(probabilities):
            if proba >= self.decision_threshold:
                predictions[idx] = pos_class
            else:
                predictions[idx] = neg_class

        return predictions

class ThresholdMLPClassifier(MLPClassifier):
    def __init__(self,
                hidden_layer_sizes=(100,), 
                activation='relu', 
                solver='adam', 
                alpha=0.0001, 
                batch_size='auto', 
                learning_rate='constant', 
                learning_rate_init=0.001, 
                power_t=0.5, 
                max_iter=200, 
                shuffle=True, 
                random_state=None, 
                tol=0.0001, 
                verbose=False, 
                warm_start=False, 
                momentum=0.9, 
                nesterovs_momentum=True, 
                early_stopping=False, 
                validation_fraction=0.1, 
                beta_1=0.9, 
                beta_2=0.999, 
                epsilon=1e-08, 
                n_iter_no_change=10, 
                max_fun=15000,
                decision_threshold=0.5):
        super(ThresholdMLPClassifier, self).__init__(
            hidden_layer_sizes=hidden_layer_sizes, 
            activation=activation, 
            solver=solver, 
            alpha=alpha, 
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            learning_rate_init=learning_rate_init, 
            power_t=power_t, 
            max_iter=max_iter, 
            shuffle=shuffle, 
            random_state=random_state, 
            tol=tol, 
            verbose=verbose, 
            warm_start=warm_start, 
            momentum=momentum, 
            nesterovs_momentum=nesterovs_momentum, 
            early_stopping=early_stopping, 
            validation_fraction=validation_fraction, 
            beta_1=beta_1, 
            beta_2=beta_2, 
            epsilon=epsilon, 
            n_iter_no_change=n_iter_no_change, 
            max_fun=max_fun,
        )
        self.decision_threshold = decision_threshold

    def predict(self, X):
        predictions = super(ThresholdMLPClassifier, self).predict(X)
        probabilities = self.predict_proba(X)[:, 1]

        pos_class = self.classes_[1]
        neg_class = self.classes_[0]
        for idx, proba in enumerate(probabilities):
            if proba >= self.decision_threshold:
                predictions[idx] = pos_class
            else:
                predictions[idx] = neg_class

        return predictions