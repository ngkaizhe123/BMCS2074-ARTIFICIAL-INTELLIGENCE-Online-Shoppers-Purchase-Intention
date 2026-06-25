import pandas as pd
from sklearn.metrics import classification_report


def print_evaluation_report(y_true, y_pred, model_name):
    print(f"\n======================================")
    print(f"[REPORT] {model_name} model classification report")
    print(f"======================================")
    report = classification_report(y_true, y_pred)
    print(report)
    print(f"======================================\n")


def get_evaluation_report_df(y_true, y_pred):
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    return df.round(4)
