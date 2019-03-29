#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019, Tencent Inc.
# All rights reserved
#
# Author: Neptunewang
# Create: 2019/01/30
#
# Main Function

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from transformer import Pipeline, FeatureName
from model_training import Classification
from evaluation import BinaryEvaluation, MultiEvaluation
from terraml_conf import config


class InputChecker(object):
    def __init__(self):
        self.feature = config["feature"]
        self.option = config["option"]
        self.path = config["path"]
        self.feature = config["feature"]
        self.model_option = config["model_option"]
        self.param_map = config["param_map"]

        self._columns_values = None
        self._label = None
        self._feature = None
        self._one_hot = None
        self._min_max = None
        self._z_score = None

    def _check_column_option_exist(self, columns):
        # check true or false certain column exists in df
        for c in columns:
            if c not in self.columns_values:
                raise ValueError("ERROR | column %s not found in data" % c)

    @property
    def columns_values(self):
        return self._columns_values

    @columns_values.setter
    def columns_values(self, column_list):
        assert column_list, "ERROR | Dataframe-column-value is empty"
        self._columns_values = column_list

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._check_column_option_exist([label])
        self._label = label

    @property
    def feature_columns(self):
        return self._feature

    @feature_columns.setter
    def feature_columns(self, column_list):
        self._check_column_option_exist(column_list)
        self._feature = column_list

    @property
    def one_hot_columns(self):
        return self._one_hot

    @one_hot_columns.setter
    def one_hot_columns(self, column_list):
        self._check_column_option_exist(column_list)
        self._one_hot = column_list

    @property
    def min_max_columns(self):
        return self._min_max

    @min_max_columns.setter
    def min_max_columns(self, column_list):
        self._check_column_option_exist(column_list)
        self._min_max = column_list

    @property
    def z_score_columns(self):
        return self._z_score

    @z_score_columns.setter
    def z_score_columns(self, column_list):
        self._check_column_option_exist(column_list)
        self._z_score = column_list

    def _check_path(self, option, path):
        assert (self.option[option] and self.path[path]) or (not self.option[option]), \
            "ERROR | if save %s, must have a path to write" % option

    def _check_type(self, option, path):
        assert (isinstance(self.option[option], bool) and isinstance(self.path[path], str)) \
               or (not self.option[option]), "ERROR | %s path is not valid" % option

    def check_config(self):
        option_list = ["save_feature", "save_pipeline", "save_model"]
        path_list = ["df_path", "pip_path", "model_path"]
        for option, path in zip(option_list, path_list):
            self._check_path(option, path)
            self._check_type(option, path)

    def init_columns(self, df):
        self.columns_values = df.columns
        self.feature_columns = self.feature["feature_all"]
        self.label = self.feature["label"]
        self.one_hot_columns = self.feature["one_hot"]
        self.z_score_columns = self.feature["z_score"]
        self.min_max_columns = self.feature["max_min"]


class Manager(InputChecker):
    def __init__(self):
        InputChecker.__init__(self)
        self.check_config()

    # mainly for test run
    def load_raw_data(self, spark, path):
        return spark.read.csv(path, inferSchema="true")

    @staticmethod
    def split_train_test(df, weights):
        for w in weights:
            if w < 0.0:
                raise ValueError("ERROR | Weights must be positive. Found weight value: %s" % w)
        return df.randomSplit(weights)

    def sample_pos_neg(self, df, pos_ratio, neg_ratio):
        fn = FeatureName()
        pos = df.where(col(fn.label_index_name) == 1.0).sample(False, pos_ratio)
        neg = df.where(col(fn.label_index_name) == 0.0).sample(False, neg_ratio)
        print("INFO | pos count: ", pos.count(), " | neg count: ", neg.count())
        return pos, neg

    def train_feature_pipeline(self, data):
        pip = Pipeline()
        pipeline, feature_name = pip\
            .pipeline_builder(feature_name=self.feature_columns,
                              label_name=self.label,
                              one_hot_name=self.one_hot_columns,
                              z_score_name=self.z_score_columns,
                              max_min_name=self.min_max_columns)
        pipeline_model = pipeline.fit(data)

        if self.option["save_pipeline"]:
            pipeline_model.save(self.path["pip_path"])

        return pipeline_model, feature_name

    def train_model(self, feature):
        model_option = self.model_option
        param_map = self.param_map[model_option]

        cls = Classification()
        model = cls.train(feature, model_option, param_map)

        if self.option["save_model"]:
            cls.save(self.path["model_path"])

        return model

    def evaluation(self, data, model, spark):
        threshold = config["threshold"]
        model_type = config["type"]
        label_and_probability = model.transform(data).select(self.label, "probability")

        if model_type == "binary" or model_type == "b":
            metric, auc, apr = BinaryEvaluation(spark)\
                .evaluation(label_and_probability, threshold)
            print("INFO | AUC: %s | APR: %s" % (auc, apr))
            metric.show()
            return metric, auc, apr
        elif model_type == "multi" or model_type == "m":
            cm, precision_label, recall_label = MultiEvaluation(spark)\
                .evaluation(label_and_probability)
            # TODO: no print, save log
            print(cm)
            print(precision_label)
            print(recall_label)
            return cm, precision_label, recall_label
        else:
            raise ValueError("ERROR | Only support b-binary/m-multi evaluation")

    def main(self):
        sc = SparkContext(appName="ml_model_training")
        spark = SparkSession.builder\
            .appName("ml_model_training") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()

        data_path = self.path["input_path"]
        data = self.load_raw_data(spark, data_path).persist()
        self.init_columns(data)

        pipeline_model, feature_name = self.train_feature_pipeline(data)
        feature = pipeline_model.transform(data).select(feature_name)
        pos, neg = self.sample_pos_neg(feature,
                                       config["label_sample_ratio"][0],
                                       config["label_sample_ratio"][1])
        df = pos.union(neg).persist()
        data.unpersist()

        train, validation, test = self.split_train_test(df, config["train_validation_test"])
        ml_model = self.train_model(train)

        self.evaluation(train, ml_model, spark)
        self.evaluation(validation, ml_model, spark)
        self.evaluation(test, ml_model, spark)

        sc.stop()


if __name__ == "__main__":
    Manager().main()
