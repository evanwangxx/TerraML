# -*- coding: utf-8 -*-
# Copyright 2019, Tencent Inc.
# All rights reserved
#
# Create by Neptunewang on 2019/01/30
#
# Feature Selection
# Spark ML Model Training

from pyspark.ml.stat import ChiSquareTest
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier

from transformer import FeatureName


class FeatureSelection(FeatureName):

    def __init__(self):
        FeatureName.__init__(self)

    def chi_square_test(self, data, feature_col):
        test = ChiSquareTest.test(data, feature_col, self.label_index_name)
        return test.statistics


class Classification(FeatureName):

    def __init__(self):
        FeatureName.__init__(self)
        self._model = None

    @property
    def model(self):
        if not self._model:
            raise ValueError("ERROR | model not set yet")
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def logistic_regression(self, elastic_param, reg_param, family):
        lr = LogisticRegression()\
            .setLabelCol(self.label_index_name)\
            .setPredictionCol(self.prediction_name)\
            .setFeaturesCol(self.feature_name)\
            .setElasticNetParam(elastic_param)\
            .setRegParam(reg_param)\
            .setFamily(family)

        return lr

    def random_forest(self, max_depth, max_num_tree):
        rf = RandomForestClassifier()\
            .setLabelCol(self.label_index_name)\
            .setPredictionCol(self.prediction_name)\
            .setFeaturesCol(self.feature_name)\
            .setMaxDepth(max_depth)\
            .setNumTrees(max_num_tree)

        return rf

    def gbdt(self, max_depth, max_bins=32):
        gbdt = GBTClassifier()\
            .setLabelCol(self.label_index_name)\
            .setPredictionCol(self.prediction_name)\
            .setFeaturesCol(self.feature_name)\
            .setMaxDepth(max_depth)\
            .setMaxBins(max_bins)

        return gbdt

    def train(self, data, option, param_map):
        """
        train a Spark ML model
        Args:
            data: as name
            option: model option
            param_map: parameter map

        Returns: SparkML model
        """
        if option == "lr":
            md = self.logistic_regression(elastic_param=param_map["elastic_param"],
                                          reg_param=param_map["reg_param"],
                                          family=param_map["family"])
        elif option == "rf":
            md = self.random_forest(max_depth=param_map["max_depth"],
                                    max_num_tree=param_map["max_num_tree"])
        elif option == "gbdt":
            md = self.gbdt(max_depth=param_map["max_depth"],
                           max_bins=param_map["max_bins"])
        else:
            raise ValueError("ERROR | model %s does not support yet" % option)

        self.model = md.fit(data)
        return self.model

    def predict(self, data, only_lp=True):
        """
        Predict by a SparkML model
        Args:
            data: as name
            only_lp: only select label, prediction and corresponding probability column

        Returns: result dataframe
        """
        df = self.model.transpose(data)
        if only_lp:
            df = df.select(self.label_index_name, self.prediction_name, self.probability)
        return df

    def save(self, path):
        self.model.write().overwrite().save(path)
