# -*- coding: utf-8 -*-
# Copyright 2019, Tencent Inc.
# All rights reserved
#
# Author: Neptunewang
# Create: 2018/08/29
#
# model evaluation for multi-class classification model

from operator import add

import math
import numpy as np

from pyspark.sql.types import StructType, StructField, DoubleType


class Evaluation(object):

    def __init__(self, spark, label=None, prediction=None):
        self.spark = spark
        self._label = label if label else "label_indexed"
        self._prediction = prediction if label else "prediction"

    @staticmethod
    def calculate_accuracy(label_and_prediction, length=None):
        """
        Calculate the accuracy for a certain model
        Args:
            label_and_prediction: rdd with label and prediction
            length: data length

        Returns:
            accuracy value
        """
        if not length:
            length = label_and_prediction.count()

        accuracy = float(label_and_prediction.filter(
            lambda r: r[0] == r[1]).count()) / float(length)
        return accuracy

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, name):
        self._label = name

    @property
    def prediction(self):
        return self._prediction

    @prediction.setter
    def prediction(self, name):
        self._prediction = name


class BinaryEvaluation(Evaluation):

    def __init__(self, spark):
        """
        Create a new BinaryClassificationEvaluation configuration
        Args:
            spark: SparkSession
        """
        super(BinaryEvaluation, self).__init__(spark=spark)
        self.auc = None
        self.apr = None
        self.matrix = None
        self.likelihood = None

    def evaluation(self, label_and_probability, threshold_list=None):
        """
        Evaluate the self.model performance on a certain data set
        Args:
            label_and_probability: data with true label and prediction
            threshold_list: thresholds, default [0.25, 0.50, 0.75]

        Returns:
            evaluation matrix, auc value, apr value
        """
        if not threshold_list:
            threshold_list = [0.25, 0.50, 0.75]

        # TODO: defaultDictionary
        output_order = ["threshold", "Accuracy",
                        "TN", "TP", "FN", "FP",
                        "precision", "recall", "TPR", "FPR", "F1"]
        result = {key: [] for key in output_order}
        data_length = float(label_and_probability.count())

        self.likelihood = label_and_probability.rdd.map(lambda x: (x[0], x[1][1])).persist()

        # TODO: O(n)*c to O(c)*n, although both in O(n), second one should be faster since less IO
        for threshold in threshold_list:
            label_and_predict = self.likelihood \
                .map(lambda l, t=threshold: (l[0], 1.0 if l[1] >= t else 0.0)).cache()

            accuracy = Evaluation.calculate_accuracy(label_and_predict, data_length)
            tn, tp, fn, fp = self.construct_confusion_matrix(label_and_predict)
            precision, recall, tpr, fpr, f1 = self.evaluate_confusion_matrix(tn, tp, fn, fp)

            result["threshold"].append(threshold)
            result["Accuracy"].append(accuracy)
            result["TN"].append(tn)
            result["TP"].append(tp)
            result["FN"].append(fn)
            result["FP"].append(fp)
            result["precision"].append(precision)
            result["recall"].append(recall)
            result["TPR"].append(tpr)
            result["FPR"].append(fpr)
            result["F1"].append(f1)

            label_and_predict.unpersist()

        self.matrix = self.convert_dictionary_to_dataframe(result, self.spark, output_order)
        roc_fpr, roc_tpr = self._convert_roc_axis(result["FPR"], result["TPR"])
        self.auc = self.calculate_trapezoid_area(roc_fpr, roc_tpr)
        self.apr = self.calculate_trapezoid_area(result["recall"], result["precision"])

        return self.matrix, self.auc, self.apr

    @staticmethod
    def construct_confusion_matrix(label_and_prediction):
        """
        Construct a confusion matrix from true label and the corresponding prediction
        Args:
            label_and_prediction: rdd with label and prediction

        Returns:
            tn: True positive
            tp: True negative
            fn: False negative
            fp: False positive
        """
        tfpn = label_and_prediction \
            .map(lambda r: ('tp', 1.0) if (r[0] == r[1]) and (r[0] == 1.0)
                 else (('tn', 1.0) if (r[0] == r[1]) and (r[0] == 0.0)
                       else (('fn', 1.0) if (r[0] != r[1]) and (r[0] == 1.0) else ('fp', 1.0)))) \
            .reduceByKey(add).collect()

        tfpn = dict(tfpn)
        flow_matrix = {k: tfpn[k] if k in tfpn.keys() else 0.0 for k in ["tn", "tp", "fn", "fp"]}

        return flow_matrix["tn"], flow_matrix["tp"], flow_matrix["fn"], flow_matrix["fp"]

    @staticmethod
    def evaluate_confusion_matrix(tn, tp, fn, fp):
        """
        Using confusion matrix to calculate model performance index
        Args:
            tn: True positive
            tp: True negative
            fn: False negative
            fp: False positive

        Returns:
            precision, recall, tpr ,fpr ,f1
        """
        tpr = tp / (tp + fn) if tp + fn != 0.0 else 0.0
        fpr = fp / (fp + tn) if fp + tn != 0.0 else 0.0
        recall = tp / (tp + fn) if tp + fn != 0.0 else 0.0
        precision = tp / (tp + fp) if tp + fp != 0.0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0.0 else 0.0

        return precision, recall, tpr, fpr, f1

    @staticmethod
    def convert_dictionary_to_dataframe(dic, spark, output_order=None):
        """
        Helper method for converting the result Dictionary to DataFrame(Spark)
        Args:
            dic: result dictionary
            spark: Spark Session
            output_order: output column order

        Returns: result DataFrame(Spark)
        """
        if not output_order:
            output_order = dic.keys()

        schema_ls = []
        for key in output_order:
            schema_ls.append(StructField(key, DoubleType(), True))
        schema = StructType(schema_ls)

        row_ls = [[] for _ in range(len(list(dic.values())[0]))]
        i = 0
        for row in row_ls:
            for k in output_order:
                row.append(dic[k][i])
            i += 1

        return spark.createDataFrame(row_ls, schema)

    @classmethod
    def _convert_roc_axis(cls, fpr, tpr):
        """
        Add (0, 0) and (1, 1) into ROC plot
        Args:
            fpr: FPR list in order
            tpr: TPR list in order

        Returns: FPR list and the corresponding TPR list
        """
        if len(tpr) != len(fpr):
            raise ValueError("ERROR | TPR and FPR length mismatch")

        fpr_plot = fpr
        fpr_plot.append(0.0)
        fpr_plot.insert(0, 1.0)

        tpr_plot = tpr
        tpr_plot.append(0.0)
        tpr_plot.insert(0, 1.0)

        return fpr_plot, tpr_plot

    @classmethod
    def calculate_trapezoid_area(cls, x, y):
        """
        Estimate the area under points (x, y) by using trapezoid area calculation
        Args:
            x: x-axis
            y: y-axis

        Returns: area
        """
        if len(x) != len(y):
            raise ValueError("ERROR | x and y length mismatch")
        else:
            area = 0
            for i in range(1, len(x)):
                tmp_area = math.fabs(x[i] - x[i - 1]) * (y[i - 1] + y[i]) * 0.5
                area += tmp_area
            return area


class MultiEvaluation(Evaluation):
    _predict_all_header = "predict_all_"
    _true_all_header = "true_all_"
    _probability_column_name = "probability"

    def __init__(self, spark):
        super(MultiEvaluation, self).__init__(spark=spark)
        self.accuracy = None
        self.distinct_label_name = None
        self.confusion_matrix = None  # confusion matrix by label
        self.precision = None  # precision dictionary by label
        self.recall = None  # recall dictionary by label

    def evaluation(self, label_and_probability):
        """
        Evaluate the self.model performance on a certain data-set
        Args:
            label_and_probability: label and probability in DataFrame
        """
        self._set_distinct_label_name(label_and_probability, self.label)
        label_and_predict = label_and_probability.rdd\
            .map(lambda l: (l[0], float(l[1].argmax()))).cache()

        self.accuracy = Evaluation.calculate_accuracy(label_and_predict)
        cm, true_sum, predict_sum, confusion_matrix = self\
            .construct_confusion_matrix(label_and_predict)
        self.precision, self.recall = self.evaluate_confusion_matrix(cm, true_sum, predict_sum)

        self.confusion_matrix = np.array(confusion_matrix)

        return self.confusion_matrix, self.precision, self.recall

    def construct_confusion_matrix(self, label_and_predict):
        """
        Private method, calculate confusion matrix
        Args:
            label_and_predict: rdd with label and prediction
        """
        cm_dict = {}
        true_sum = {}
        predict_sum = {}
        confusion_matrix = []

        for label in self.distinct_label_name:
            true_tmp_sum = 0.0
            tmp_confusion_matrix = []
            for predict in self.distinct_label_name:
                value = float(label_and_predict.filter(
                    lambda r, lb=label, p=predict: r[0] == lb and r[1] == p).count())

                cm_dict[str(label) + "_" + str(predict)] = value
                tmp_confusion_matrix.append(value)

                if self._predict_all_header + str(predict) not in predict_sum.keys():
                    predict_sum[self._predict_all_header + str(predict)] = value
                else:
                    predict_sum[self._predict_all_header + str(predict)] += value
                true_tmp_sum += value

            confusion_matrix.append(tmp_confusion_matrix)
            true_sum[self._true_all_header + str(label)] = true_tmp_sum

        return cm_dict, true_sum, predict_sum, confusion_matrix

    def evaluate_confusion_matrix(self, confusion_matrix, true_sum, predict_sum):
        """
        Calculate Precision by label and Recall by label
        Args:
            confusion_matrix: confusion matrix
            true_sum: true label row sum dictionary
            predict_sum: predicted label column sum dictionary

        Returns:
            precision and recall matrix by label
        """
        precision = {}
        recall = {}
        for label in self.distinct_label_name:
            label_str = str(label)

            value = confusion_matrix[label_str + "_" + label_str]
            tmp_true_sum = true_sum[self._true_all_header + label_str]
            tmp_predict_sum = predict_sum[self._predict_all_header + label_str]

            precision["precision_" + label_str] = value / tmp_predict_sum \
                if tmp_predict_sum != 0.0 else None
            recall["recall_" + label_str] = value / tmp_true_sum \
                if tmp_true_sum != 0.0 else None

        return precision, recall

    def _set_distinct_label_name(self, data, label_name=None):
        """
        Set distinct value for label column
        Args:
            data: data in DataFrame
            label_name: Label column name, default LABEL_NAME

        Returns: None
        """
        if not label_name:
            label_name = self.label

        distinct_label_name = [u[label_name] for u in data.select(label_name).distinct().collect()]

        distinct_label_name.sort()
        self.distinct_label_name = distinct_label_name
