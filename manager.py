#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Neptunewang
# Create: 2019/01/30
#
# Main Function

from pyspark import SparkContext
from pyspark.sql import SparkSession

from transformer import Pipeline
from model_training import Classification
from evaluation import BinaryEvaluation, MultiEvaluation
from terraml_conf import config


class Manager:
    def __init__(self):
        self.option = config["option"]
        self.path = config["path"]
        self.feature = config["feature"]
        self.model_option = config["model_option"]
        self.param_map = config["param_map"]

        # defence
        self._check_config()

    def _check_path(self, option, path, text):
        assert (self.option[option] and self.path[path]) or \
               (not self.option[option]), \
            "ERROR | if save %s, must have a path to write" % text

    def _check_type(self, option, path, text):
        assert (isinstance(self.option[option], bool) and
                isinstance(self.path[path], str)) or \
               (not self.option[option]), \
            "ERROR | $s path is not vaild" % text

    def _check_feature_save(self):
        option = "save_feature"
        path = "df_path"
        text = "feature"
        self._check_path(option, path, text)
        self._check_type(option, path, text)

    def _check_pipeline_save(self):
        option = "save_pipeline"
        path = "pip_path"
        text = "pipeline"
        self._check_path(option, path, text)
        self._check_type(option, path, text)

    def _check_model_save(self):
        option = "save_model"
        path = "model_path"
        text = "model"
        self._check_path(option, path, text)
        self._check_type(option, path, text)

    def _check_config(self):
        self._check_feature_save()
        self._check_pipeline_save()
        self._check_model_save()

    def load_raw_data(self, spark, path):
        return spark.read.csv(path, inferSchema="true")

    @staticmethod
    def split_train_test(df, weights):
        for w in weights:
            if w < 0.0:
                raise ValueError("ERROR | Weights must be positive. Found weight value: %s" % w)
        return df.randomSplit(weights)

    def train_feature_pipeline(self, data):
        feature_all = self.feature["feature_all"]
        label = self.feature["label"]
        one_hot = self.feature["one_hot"]
        z_score = self.feature["z_score"]
        max_min = self.feature["max_min"]

        pip = Pipeline()
        pipeline, feature_name = pip\
            .pipeline_builder(feature_name=feature_all, label_name=label,
                              one_hot_name=one_hot, z_score_name=z_score, max_min_name=max_min)
        pipeline_model = pip.train(data)

        if self.option["save_pipeline"]:
            pip.save(self.path["pip_path"])

        return pipeline_model, feature_name

    def train_model(self, feature):
        model_option = self.model_option
        param_map = self.param_map[model_option]

        cls = Classification()
        model = cls.train(feature, model_option, param_map)

        if self.option["save_model"]:
            cls.save(self.path["model_path"])

        return model

    def evaluation(self, feature, model, spark):
        threshold = config["threshold"]
        model_type = config["type"]

        if model_type == "binary" or model_type == "b":
            metric, auc, apr = BinaryEvaluation(model, spark).evaluation(feature, threshold)
            print("INFO | AUC: %s | APR: %s" % (auc, apr))
            metric.show()
            return metric, auc, apr
        elif model_type == "multi" or model_type == "m":
            cm, precision_label, recall_label = MultiEvaluation(model, spark).evaluation(feature)
            # TODO
            print(cm)
            print(precision_label)
            print(recall_label)
            return cm, precision_label, recall_label
        else:
            raise ValueError("ERROR | Only support b-binary/m-multi evaluation")

    def main(self):
        sc = SparkContext(appName="3c_variables_make")
        spark = SparkSession.builder\
            .master("local") \
            .appName("3c_model_training") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()

        data_path = self.path["input_path"]
        data = self.load_raw_data(spark, data_path)

        pipeline_model, feature_name = self.train_feature_pipeline(data)
        feature = pipeline_model.transform(data).select(feature_name)

        train, validation, test = self.split_train_test(feature, config["train_validation_test"])
        ml_model = self.train_model(train)

        self.evaluation(train, ml_model, spark)
        self.evaluation(validation, ml_model, spark)
        self.evaluation(test, ml_model, spark)

        sc.stop()


TERRA_ML = """
 _____                              __  
/__   \___ _ __ _ __ __ _  /\/\    / /  
  / /\/ _ \ '__| '__/ _` |/    \  / /   
 / / |  __/ |  | | | (_| / /\/\ \/ /___ 
 \/   \___|_|  |_|  \__,_\/    \/\____/ 

"""

if __name__ == "__main__":
    print(TERRA_ML)
    Manager().main()
