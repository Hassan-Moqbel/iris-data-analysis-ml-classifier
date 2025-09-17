"""
حزمة src لمشروع تصنيف زهور Iris
"""
from .data_loading import load_iris_data, create_iris_dataframe, explore_dataset
from .model_training import train_models, get_best_model, prepare_data
from .model_evaluation import evaluate_model, plot_confusion_matrix, plot_feature_importance
from .utils import save_model, load_model, predict_new_sample
from .web_interface import run_web_interface

__version__ = "0.1.0"
__author__ = "Your Name" 
