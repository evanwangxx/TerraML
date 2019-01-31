#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Neptunewang
# Create: 2018/08/16
#
# Data transformer for pipeline building
#

from pyspark.ml.pipeline import Pipeline as SparkPipeline
from pyspark.ml.feature import OneHotEncoder, StandardScaler, MinMaxScaler
from pyspark.ml.feature import PCA, Imputer
from pyspark.ml.feature import VectorAssembler, StringIndexer


def since(version):
    """
    A decorator that annotates a function to append the version of Spark the function was added.
    """
    import re
    indent_p = re.compile(r'\n( +)')

    def deco(f):
        indents = indent_p.findall(f.__doc__)
        indent = ' ' * (min(len(m) for m in indents) if indents else 0)
        f.__doc__ = f.__doc__.rstrip() + "\n\n%s.. versionadded:: %s" % (indent, version)
        return f
    return deco


class FeatureName(object):

    def __init__(self):
        self._one_hot_footer = "_one_hot"
        self._z_score_footer = "_z_score"
        self._standard_footer = "_standard"
        self._min_max_footer = "_max_min"

        self.label_index_name = "label_indexed"
        self.prediction_name = "prediction"
        self.probability = "probability"
        self.feature_name = "feature"


class Transformer(FeatureName):

    def __init__(self):
        FeatureName.__init__(self)

    # def one_hot_encoder(input_one_hot_feature_names):
    #     """ Attention: this function only available for Spark version over 2.3.0
    #         args:
    #              input_one_hot_feature_names: List(String), feature names
    #         returns:
    #              one_hot_pipeline_stage: List(PipelineStage),one hot pipeline stages
    #                                      and modified one hot column names
    #              feature_name: List(String), feature names
    #     """
    #
    #     output_name = [name + "_one_hot" for name in input_one_hot_feature_names]
    #     encoder = OneHotEncoderEstimator(inputCols=input_one_hot_feature_names,
    #                                      outputCols=output_name)
    #
    #     one_hot_pipeline_stage = Pipeline().setStages(encoder)
    #
    #     return one_hot_pipeline_stage, output_name

    def one_hot_encoder(self, column_names, invalid="keep"):
        """
        Attention: OneHotEncoder has been deprecated in 2.3.0 and will be removed in 3.0.0
        Args:
            column_names: List(String), feature names
            invalid: handle invalid situation, default keep

        Returns:
            list of pipeline stage
        """
        stages = list()
        footer = self._one_hot_footer

        for name in column_names:
            stages.append(StringIndexer()
                          .setInputCol(name)
                          .setOutputCol(name + "_indexed")
                          .setHandleInvalid(invalid))

            stages.append(OneHotEncoder()
                          .setInputCol(name + "_indexed")
                          .setOutputCol(name + footer))
        return stages

    def z_score_encoder(self, column_names, mean=True, sd=True):
        """
        Normalize features in column_names
        Args:
            column_names: List(String), feature names for scale columns
            mean: Boolean, default True
            sd: Boolean, default True

        Returns:
            list of pipeline stage
        """

        stages = list()
        footer = self._z_score_footer
        feature_name = [(name, name + footer) for name in column_names]

        for name in feature_name:
            inner_name = name[0]
            output_name = name[1]

            vector = VectorAssembler().setInputCols([inner_name])\
                .setOutputCol(inner_name + "_vector")

            z_score = StandardScaler().setInputCol(inner_name+"_vector")\
                .setOutputCol(output_name)\
                .setWithMean(mean)\
                .setWithStd(sd)

            stages.extend([vector, z_score])

        return stages

    def standard_encoder(self, column_name, mean=True, sd=True):
        """
        Normalize a single feature
        Args:
            column_name: List(String), feature names for scale columns
            mean: Boolean, default True
            sd: Boolean, default True

        Returns:
            pipeline model
        """
        footer = self._standard_footer

        vector = VectorAssembler()\
            .setInputCols(column_name)\
            .setOutputCol(column_name[0] + "_vector")

        standard = StandardScaler().setInputCol(column_name[0] + "_vector")\
            .setOutputCol(column_name[0] + footer)\
            .setWithMean(mean).setWithStd(sd)

        pipe = SparkPipeline().setStages([vector, standard])

        return pipe

    def min_max_encoder(self, column_names, _max=1.0, _min=0.0, footer=None):
        """
        Scale features by max-min
        Args:
            column_names: List(String), feature names for max-min scale columns
            _max: float, default 1.0
            _min: float, default 0.0
            footer: String, footnote for modified column names

        Returns:
            list of pipeline stage
        """
        stages = list()
        footer = self._min_max_footer if not footer else footer
        feature_name = [(name, name + footer) for name in column_names]

        for name in feature_name:
            inner_name = name[0]
            output_name = name[1]

            vector = VectorAssembler().setInputCols([inner_name])\
                .setOutputCol(inner_name + "_vector")

            max_min = MinMaxScaler().setInputCol(inner_name+"_vector")\
                .setOutputCol(output_name).setMax(_max).setMin(_min)

            stages.extend([vector, max_min])

        return stages

    @staticmethod
    def pca_encoder(k, features_name, output_name):
        pca_model = PCA()\
            .setK(k)\
            .setInputCol(features_name)\
            .setOutputCol(output_name)

        return [pca_model]

    @ staticmethod
    def pca(data, k, features_name, output_name):

        pca_model = Transformer().pca_encoder(k, features_name, output_name)

        return pca_model[0].fit(data)

    @ staticmethod
    def imputer(features_name, strategy="mean", missing_value=None, footer="_imputer"):
        """
        Spark experiment method
        Args:
            features_name:
            strategy:
            missing_value:
            footer:

        Returns:

        """
        output_names = [name+footer for name in features_name]

        imputer = Imputer() \
            .setInputCols(features_name) \
            .setOutputCols(output_names)\
            .setStrategy(strategy)

        if missing_value:
            imputer.setMissingValue(missing_value)

        return imputer

    def feature_column_name(self, feature_list, one_hot_list, z_score_list, max_min_list):
        """
        collect feature names for VectorAssembler
        Args:
            feature_list: feature_list: List(String), original feature names
            one_hot_list: List(String), features needs to do one hot
            z_score_list: List(String), features needs to do z-score scale
            max_min_list: List(String), features needs to do max-min scale

        Returns:
            feature names
        """
        footer_one_hot = self._one_hot_footer
        footer_z_score = self._z_score_footer
        footer_max_min = self._min_max_footer

        output_name = [name for name in feature_list if name not in one_hot_list
                       and name not in z_score_list]

        if one_hot_list:
            output_name.extend([name + footer_one_hot for name in one_hot_list])

        if z_score_list:
            output_name.extend([name + footer_z_score for name in z_score_list])

        if max_min_list:
            output_name.extend([name + footer_max_min for name in max_min_list])

        return output_name


class Pipeline(FeatureName):

    def __init__(self):
        FeatureName.__init__(self)
        self._pipeline = None
        self._pipeline_model = None

    @property
    def pipeline(self):
        if not self._pipeline:
            raise ValueError("ERROR | pipeline not set yet")
        return self._pipeline

    @pipeline.setter
    def pipeline(self, pipeline):
        self._pipeline = pipeline

    @property
    def pipeline_model(self):
        if not self._pipeline_model:
            raise ValueError("ERROR | pipeline model not set yet")
        return self._pipeline_model

    @pipeline_model.setter
    def pipeline_model(self, pip_model):
        self._pipeline_model = pip_model

    def pipeline_builder(self, feature_name,
                         label_name=None,
                         one_hot_name=None,
                         z_score_name=None,
                         max_min_name=None):
        """
        Build a Pipeline Stage
        Args:
            label_name: label column name in data
            feature_name: all features' name
            one_hot_name: features need to be transformed to one-hot
            z_score_name: features need to be transformed to z-score
            max_min_name: features need to be transformed by max-min scale

        Returns:
            pipeline: pipeline for data transformation
            feature_column: modified feature columns' name
        """
        stage = list()
        features = [self.feature_name]

        if one_hot_name:
            one_hot_stage = Transformer().one_hot_encoder(one_hot_name)
            stage.extend(one_hot_stage)
        else:
            one_hot_name = []

        if z_score_name:
            z_score_stage = Transformer().z_score_encoder(z_score_name)
            stage.extend(z_score_stage)
        else:
            z_score_name = []

        if max_min_name:
            max_min_stage = Transformer().min_max_encoder(max_min_name)
            stage.extend(max_min_stage)
        else:
            max_min_name = []

        feature_column = Transformer()\
            .feature_column_name(feature_name, one_hot_name, z_score_name, max_min_name)
        if label_name:
            label_index_stage = StringIndexer()\
                .setInputCol(label_name)\
                .setOutputCol(self.label_index_name)
            stage.append(label_index_stage)
            features.extend([label_name, self.label_index_name])

        vector_assembler = VectorAssembler()\
            .setInputCols(feature_column)\
            .setOutputCol(self.feature_name)
        stage.append(vector_assembler)

        self._pipeline = SparkPipeline().setStages(stage)
        features.extend(feature_column)

        return self._pipeline, features

    def train(self, df):
        self.pipeline_model = self.pipeline.fit(df)
        return self.pipeline_model

    def save(self, path):
        self.pipeline_model.write().overwrite().save(path)
