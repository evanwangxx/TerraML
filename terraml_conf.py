# -*- coding: utf-8 -*-
#
# Author: Neptunewang
# Create: 2019/01/30
#
# config for TerraML

config = {
    # Running Option
    "option": {
        "run_transformation": True,
        "run_model_training": True,
        "run_evaluation": True,
        "save_feature": False,
        "save_pipeline": True,
        "save_model": True,
    },
    # Path Options
    "path": {
        "input_path": "hdfs://ss-cdg-3-v2/data/MAPREDUCE/CDG/g_sng_gdt_gdt_targeting/halfyear/neptunewang/ml_data/adult.txt",
        "df_path": "hdfs://ss-cdg-3-v2/data/MAPREDUCE/CDG/g_sng_gdt_gdt_targeting/halfyear/neptunewang/terraml/df/",
        "pip_path": "hdfs://ss-cdg-3-v2/data/MAPREDUCE/CDG/g_sng_gdt_gdt_targeting/halfyear/neptunewang/terraml/pipeline/",
        "model_path": "hdfs://ss-cdg-3-v2/data/MAPREDUCE/CDG/g_sng_gdt_gdt_targeting/halfyear/neptunewang/terraml/model/",
        "predict_df_path": "path"
    },
    # Feature Transformation
    "feature": {
        # - all features: List[String]
        "feature_all": "_c1|_c2|_c3|_c5|_c12|_c13".split('|'),
        # - label column: String
        "label": "_c14",
        # - one-hot columns: List[String]
        "one_hot": "_c1|_c3|_c5|_c13".split('|'),
        # - z-score columns: List[String]
        "z_score": [],
        # - max-min columns: List[String]
        "max_min": "_c2|_c12".split('|')
    },

    # Model Training
    "train_validation_test": [0.7, 0.2, 0.1],
    "model_option": "lr",
    "param_map": {
        "lr": {
            # - elastic parameter: Float
            "elastic_param": 0.5,
            # - regularization parameter: Float
            "reg_param": 0.1,
            # - classification family: binary/multi : String
            "family": 'auto'
        },
        "rf": {
            # - max depth
            "max_depth": 10,
            # - max number of tree
            "max_num_tree": 50
        },
        "gbdt": {
            # - max depth
            "max_depth": 10,
            # - max bin size
            "max_bins": 32
        }
    },

    # Model Evaluation
    # - model threshold
    "threshold": [0.50],
    # - evaluation type: support b: binary || m: multi
    "type": "b"
}
