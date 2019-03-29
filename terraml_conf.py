# -*- coding: utf-8 -*-
# Copyright 2019, Tencent Inc.
# All rights reserved
#
# Author: Neptunewang
# Create: 2019/01/30
#
# config for TerraML
# specify for 3c-phone-prediction

PATH = "hdfs://ss-cdg-3-v2/data/MAPREDUCE/CDG/g_sng_gdt_gdt_targeting/halfyear/neptunewang/"

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
        "input_path": PATH + "/3c_device_prediction/features/test/change",
        "df_path": PATH + "terraml/df/",
        "pip_path": PATH + "terraml/pipeline/",
        "model_path": PATH + "terraml/model/",
        "predict_df_path": "path"
    },
    # Feature Transformation
    "feature": {
        # - all features: List[String]
        "feature_all": "brand,device_alias,model_in_use_month,price,age,gender,education," \
                       "common_country_id_4ad,common_province_id_4ad,common_city_id_4ad," \
                       "open_card_num,pay_banks,screen_pixel,cam_back,cam_front,cam_type," \
                       "core_cnt,cpu_model,cpu_frequency,ram,rom,battery_size,weight," \
                       "simcard,isdualcard,exposed_year,exposed_month,buy_year,buy_month," \
                       "date_from_expose,resolution_length,resolution_width," \
                       "model_size_length,model_size_width,model_size_thick".split(","),
        # - label column: String
        "label": "label",
        # - one-hot columns: List[String]
        "one_hot": ["brand", "device_alias", "gender", "education",
                    "common_country_id_4ad", "common_province_id_4ad", "common_city_id_4ad",
                    "cam_back", "cam_front", "cam_type", "core_cnt", "cpu_model", "cpu_frequency",
                    "ram", "rom",
                    "simcard", "isdualcard"],
        # - z-score columns: List[String]
        "z_score": [],
        # - max-min columns: List[String]
        "max_min": ['model_in_use_month', 'price', 'age', 'open_card_num',
                    'pay_banks', 'screen_pixel', 'battery_size', 'weight',
                    'exposed_year', 'exposed_month', 'buy_year', 'buy_month',
                    'date_from_expose', 'resolution_length', 'resolution_width',
                    'model_size_length', 'model_size_width', 'model_size_thick']
    },

    # Model Training
    "train_validation_test": [0.7, 0.2, 0.1],
    "label_sample_ratio": [1.0, 0.25],
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
    "threshold": [0.25, 0.50, 0.75],
    # - evaluation type: support b: binary || m: multi
    "type": "b"
}
