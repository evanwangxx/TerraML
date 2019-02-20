#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Neptunewang
# Create: 2019/01/30
#
# Main Function

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from transformer import Pipeline, FeatureName
from model_training import Classification, FeatureSelection
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
        for col in columns:
            if col in self.columns_values:
                pass
            else:
                raise ValueError("ERROR | column %s not found in data" % col)

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

    def _check_path(self, option, path, text):
        assert (self.option[option] and self.path[path]) or (not self.option[option]), \
            "ERROR | if save %s, must have a path to write" % text

    def _check_type(self, option, path, text):
        assert (isinstance(self.option[option], bool) and isinstance(self.path[path], str)) \
               or (not self.option[option]), "ERROR | %s path is not valid" % text

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

    def check_config(self):
        self._check_feature_save()
        self._check_pipeline_save()
        self._check_model_save()

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
        sc = SparkContext(appName="3c_variables_make")
        spark = SparkSession.builder\
            .master("local") \
            .appName("3c_model_training") \
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
        df = pos.unionAll(neg).persist()
        data.unpersist()

        FeatureSelection.chi_square_test(df, "feature", "label_indexed")

        train, validation, test = self.split_train_test(df, config["train_validation_test"])
        ml_model = self.train_model(train)

        self.evaluation(train, ml_model, spark)

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
