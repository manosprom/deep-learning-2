from enum import Enum
from src.seeded import tf, seed


class MuraLoader(object):
    class BodyPart(Enum):
        XR_SHOULDER = 'XR_SHOULDER'
        XR_HUMERUS = 'XR_HUMERUS'
        XR_FINGER = 'XR_FINGER'
        XR_ELBOW = 'XR_ELBOW'
        XR_WRIST = 'XR_WRIST'
        XR_FOREARM = 'XR_FOREARM'
        XR_HAND = 'XR_HAND'

    class Category(Enum):
        POSITIVE = 1
        NEGATIVE = 0

    def __init__(self, folder="data", name="MURA-v1.1"):
        import os

        self._name = name
        self.folder = folder
        train_image_paths_filename = "train_image_paths.csv"
        train_labeled_studies_filename = "train_labeled_studies.csv"
        valid_image_paths_filename = "valid_image_paths.csv"
        valid_labeled_studies_filename = "valid_labeled_studies.csv"

        self._mura_path = os.path.abspath(os.path.dirname(__file__) + f"/../{self.folder}/{self._name}/")
        self._train_image_paths_filepath = "/".join([self._mura_path, train_image_paths_filename])
        self._train_labeled_studies_filepath = "/".join([self._mura_path, train_labeled_studies_filename])
        self._valid_image_paths_filepath = "/".join([self._mura_path, valid_image_paths_filename])
        self._valid_labeled_studies_filepath = "/".join([self._mura_path, valid_labeled_studies_filename])

        self.imageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255
        )

        self.__load()

    def __load(self):
        train_image_paths, train_labeled_studies, valid_image_paths, valid_labeled_studies \
            = self.__load_initial(
            self._train_image_paths_filepath,
            self._train_labeled_studies_filepath,
            self._valid_image_paths_filepath,
            self._valid_labeled_studies_filepath
        )

        self._train_set = self.__expand_imagepath_dataset(train_image_paths)
        self._test_set = self.__expand_imagepath_dataset(valid_image_paths)

    def __load_initial(self, train_image_paths_filepath, train_labeled_studies_filepath, valid_image_paths_filepath, valid_labeled_studies_filepath):
        import pandas as pd
        train_image_paths = pd.read_csv(filepath_or_buffer=train_image_paths_filepath, names=['InitialImagePath'])
        train_labeled_studies = pd.read_csv(filepath_or_buffer=train_labeled_studies_filepath, names=['InitialStudyPath'], sep=",")

        valid_image_paths = pd.read_csv(filepath_or_buffer=valid_image_paths_filepath, names=['InitialImagePath'])
        valid_labeled_studies = pd.read_csv(filepath_or_buffer=valid_labeled_studies_filepath, names=['InitialStudyPath'], sep=",")

        return train_image_paths, train_labeled_studies, valid_image_paths, valid_labeled_studies

    def __expand_imagepath_dataset(self, dataFrame):
        import re
        dataFrame = dataFrame.copy()
        temp = dataFrame.InitialImagePath.str.split("/", expand=True)

        name = re.escape(self._name + "/")
        dataFrame['ImagePath'] = dataFrame.InitialImagePath.str.replace(rf"^{name}", "", n=1, regex=True)
        dataFrame['Dataset'] = temp[0]
        dataFrame['Type'] = temp[1]
        dataFrame['BodyPart'] = temp[2]
        dataFrame['PatientNo'] = temp[3]
        dataFrame['StudyNo'] = temp[4].str.split("_", expand=True)[0]
        dataFrame['StudyLabelStr'] = temp[4].str.split("_", expand=True)[1]
        dataFrame['StudyLabel'] = dataFrame['StudyLabelStr'].apply(lambda study_label_str: 1 if study_label_str == 'positive' else 0).astype(int)
        dataFrame['ImageFilename'] = temp[5]
        return dataFrame

    def get_initial_sets(self):
        return self._train_set, self._test_set

    def get_sets(self, split=0.2, body_part: str = None):
        from sklearn.model_selection import train_test_split

        if body_part:
            train_set = self._train_set[self._train_set['BodyPart'] == body_part]
            test_set = self._test_set[self._test_set['BodyPart'] == body_part]
        else:
            train_set = self._train_set
            test_set = self._test_set

        train_set, validation_set = train_test_split(train_set, stratify=train_set['StudyLabel'], random_state=seed, test_size=split)
        return train_set, validation_set, test_set

    def get_generators(self, train_set, validation_set, test_set, batch_size=32):
        # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
        # https://medium.com/datadriveninvestor/keras-imagedatagenerator-methods-an-easy-guide-550ecd3c0a92

        print(f"Creating train generator")
        train_generator = self.get_generator(train_set, shuffle=True, batch_size=batch_size)

        print(f"Creating validation generator")
        validation_generator = self.get_generator(validation_set, shuffle=False, batch_size=batch_size)

        print(f"Creating test generator")
        test_generator = self.get_generator(test_set, shuffle=False, batch_size=batch_size)

        return train_generator, validation_generator, test_generator

    def get_generator(self, dataset, shuffle=False, batch_size=32, target_size=(256, 256), color_mode="rgb"):
        generator = self.imageDataGenerator.flow_from_dataframe(
            dataset,
            directory=self._mura_path,
            x_col='ImagePath',
            y_col='StudyLabel',
            weight_col=None,
            target_size=target_size,
            color_mode=color_mode,
            classes=None,
            class_mode='raw',
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=None,
            save_prefix='',
            save_format='png',
            subset=None,
            interpolation='nearest',
            validate_filenames=True
        )
        return generator
