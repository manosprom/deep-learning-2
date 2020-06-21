class MuraLoader(object):
    def __init__(self, name="MURA-v.1.1"):
        import os

        self._name=name
        self._train_image_paths_filename = "train_image_paths.csv"
        self._train_labeled_studies_filename = "train_labeled_studies.csv"
        self._valid_image_path_filename = "valid_image_paths.csv"
        self._valid_labeled_studies_filename = "valid_labeled_studies.csv"

        self._mura_path = os.path.abspath(os.path.dirname(__file__) + f"/../{self._name}/")
        self._train_image_paths_filepath = self._mura_path + self._train_image_paths_filename
        self._train_labeled_studies_filepath = self._mura_path + self._train_labeled_studies_filename
        self._valid_image_paths_filepath = self._mura_path + self._valid_image_path_filename
        self._valid_labeled_studies_filepath = self._mura_path + self._valid_labeled_studies_filename

    def load_dataset(self):
        import pandas as pd
        p
