import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datasets import load_dataset
from sklearn.metrics import mean_squared_error, accuracy_score

# --- CONFIGURATION ---
REDNESS_MILD = 0.10
REDNESS_MODERATE = 0.15
REDNESS_SEVERE = 0.19

FATIGUE_MILD = 130
FATIGUE_MODERATE = 110
FATIGUE_SEVERE = 90

SYMMETRY_TOLERANCE = 0.02

# --- LOAD DATASET ---
dataset = load_dataset("falah/eye-disease-dataset")["train"]  # use train split
print(dataset)

