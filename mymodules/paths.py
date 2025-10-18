import os

class Paths:
    def __init__(self, base="open_dev", *, model_name: str="lgbm"):
        self.base = base
        os.makedirs(base, exist_ok=True)

        data_base  = os.path.join(base, "data")
        os.makedirs(data_base, exist_ok=True)
        os.makedirs(os.path.join(data_base, "train"), exist_ok=True)
        os.makedirs(os.path.join(data_base, "test"), exist_ok=True)
        self.train_meta = os.path.join(data_base, f"train.csv")
        self.train_A    = os.path.join(data_base, "train", f"A.csv")
        self.train_B    = os.path.join(data_base, "train", f"B.csv")
        self.test_meta  = os.path.join(data_base, f"test.csv")
        self.test_A     = os.path.join(data_base, "test", f"A.csv")
        self.test_B     = os.path.join(data_base, "test", f"B.csv")

        model_base = os.path.join(base, "model")
        os.makedirs(model_base, exist_ok=True)
        self.model_A = os.path.join(model_base, f"{model_name}_A.pkl")
        self.model_B = os.path.join(model_base, f"{model_name}_B.pkl")

        output_base = os.path.join(base, "output")
        os.makedirs(output_base, exist_ok=True)
        self.submission        = os.path.join(output_base, f"submission.csv")

    def __str__(self):
        return self.base
