import unittest
from DecisionTreeClassifier import *
from Encoder import *

class TestDecisionTreeClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_file = "test_encoder.csv"
        cls.dt_classifier = DecisionTreeClassifier(cls.test_file, "info gain", 1)
        cls.en = Encoder(cls.dt_classifier.data, ["Gender", "Color", "Sport", "Dominant Hand", "Home State", "Allergy", "Food"])
        #cls.dt_classifier.data = cls.en.encode()

    def test_calculate_entropy(self):
        data = self.dt_classifier.data
        self.assertAlmostEqual(self.dt_classifier.calculate_entropy(data), -1 * (0.3 * math.log2(0.3) + 0.6 * math.log2(0.6) + 0.1 * math.log2(0.1)))

    def test_calculate_gini_index(self):
        data = self.dt_classifier.data
        self.assertAlmostEqual(self.dt_classifier.calculate_gini_index(data), 1-(0.3**2 + 0.6**2 + 0.1**2))

    def test_get_majority_class(self):
        data = self.dt_classifier.data
        self.assertEqual(self.dt_classifier.get_majority_class(data), "B")

    def test_calculate_split_entropy_Gender(self):
        data = self.dt_classifier.data
        left_data = data.loc[data["Gender"] == 0, :]
        right_data = data.loc[data["Gender"] == 1, :]
        self.assertEqual(self.dt_classifier.calculate_split_entropy(self.dt_classifier.data, left_data, right_data), 1)

    def test_calculate_split_entropy_Color(self):
        data = self.dt_classifier.data
        left_data = data.loc[data["Color"] == 0, :]
        right_data = data.loc[data["Color"] == 1, :]
        self.assertEqual(self.dt_classifier.calculate_split_entropy(self.dt_classifier.data, left_data, right_data), -1 * (0.7 * math.log2(0.7) + 0.3 * math.log2(0.3)))

    def test_calculate_split_entropy_Sport(self):
        data = self.dt_classifier.data
        left_data = data.loc[data["Sport"] == 0, :]
        right_data = data.loc[data["Sport"] == 1, :]
        self.assertEqual(self.dt_classifier.calculate_split_entropy(self.dt_classifier.data, left_data, right_data),
                         -1 * (0.6 * math.log2(0.6) + 0.4 * math.log2(0.4)))

    def test_calculate_split_gini_index_Gender(self):
        data = self.dt_classifier.data
        left_data = data.loc[data["Gender"] == 0, :]
        right_data = data.loc[data["Gender"] == 1, :]
        self.assertEqual(self.dt_classifier.calculate_split_gini_index(self.dt_classifier.data, left_data, right_data),
                         1 - (0.5**2 + 0.5**2))

    def test_calculate_split_gini_index_f2(self):
        data = self.dt_classifier.data
        left_data = data.loc[data["Color"] == 0, :]
        right_data = data.loc[data["Color"] == 1, :]
        self.assertEqual(self.dt_classifier.calculate_split_gini_index(self.dt_classifier.data, left_data, right_data),
                         1 - (0.7**2 + 0.3**2))

    def test_calculate_split_gini_index_f3(self):
        data = self.dt_classifier.data
        left_data = data.loc[data["Sport"] == 0, :]
        right_data = data.loc[data["Sport"] == 1, :]
        self.assertEqual(self.dt_classifier.calculate_split_gini_index(self.dt_classifier.data, left_data, right_data),
                         1 - (0.6**2 + 0.4**2))

    def test_calculate_info_gain_f1(self):
        data = self.dt_classifier.data
        self.assertAlmostEqual(self.dt_classifier.calculate_info_gain(self.dt_classifier.calculate_entropy(data), data, "Gender")[0], 0.1245112498)

    def test_calculate_info_gain_f5(self):
        data = self.dt_classifier.data
        self.assertAlmostEqual(self.dt_classifier.calculate_info_gain(self.dt_classifier.calculate_entropy(data), data, "Home State")[0], 0.1735337494)

    def test_calculate_gini_gain_f2(self):
        data = self.dt_classifier.data
        self.assertAlmostEqual(self.dt_classifier.calculate_gini_gain(self.dt_classifier.calculate_gini_index(data), data, "Color")[0], 0.0542857143)

    def test_calculate_gini_gain_f7(self):
        data = self.dt_classifier.data
        self.assertAlmostEqual(self.dt_classifier.calculate_gini_gain(self.dt_classifier.calculate_gini_index(data), data, "Food")[0], 0.123333333333)

    def test_train(self):
        self.dt_classifier.train()
        self.assertEqual(self.dt_classifier.predict(["Male", "Yellow", "Football", "Left Handed", "Maryland", "No Allergies", 5.43]), "B")
        self.assertEqual(self.dt_classifier.predict([0, 1, 0, 1, 1, "Allergies", 6.8]), "A")


