import unittest
from DecisionTreeClassifier import *


class TestDecisionTreeClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_file = "test_decision_tree.csv"
        cls.dt_classifier = DecisionTreeClassifier(cls.test_file, "info gain", 1, 2)

    def test_calculate_entropy(self):
        data = self.dt_classifier.data
        self.assertAlmostEqual(self.dt_classifier.calculate_entropy(data), -1 * (0.3 * math.log2(0.3) + 0.6 * math.log2(0.6) + 0.1 * math.log2(0.1)))

    def test_calculate_gini_index(self):
        data = self.dt_classifier.data
        self.assertAlmostEqual(self.dt_classifier.calculate_gini_index(data), 1-(0.3**2 + 0.6**2 + 0.1**2))

    def test_get_majority_class(self):
        data = self.dt_classifier.data
        self.assertEqual(self.dt_classifier.get_majority_class(data), "B")

    def test_calculate_split_entropy_f1(self):
        data = self.dt_classifier.data
        left_data = data.loc[data["f1"] == 0, :]
        right_data = data.loc[data["f1"] == 1, :]
        self.assertEqual(self.dt_classifier.calculate_split_entropy(self.dt_classifier.data, left_data, right_data), 1)

    def test_calculate_split_entropy_f2(self):
        data = self.dt_classifier.data
        left_data = data.loc[data["f2"] == 0, :]
        right_data = data.loc[data["f2"] == 1, :]
        self.assertEqual(self.dt_classifier.calculate_split_entropy(self.dt_classifier.data, left_data, right_data), -1 * (0.7 * math.log2(0.7) + 0.3 * math.log2(0.3)))

    def test_calculate_split_entropy_f3(self):
        data = self.dt_classifier.data
        left_data = data.loc[data["f3"] == 0, :]
        right_data = data.loc[data["f3"] == 1, :]
        self.assertEqual(self.dt_classifier.calculate_split_entropy(self.dt_classifier.data, left_data, right_data),
                         -1 * (0.6 * math.log2(0.6) + 0.4 * math.log2(0.4)))

    def test_calculate_split_gini_index_f1(self):
        data = self.dt_classifier.data
        left_data = data.loc[data["f1"] == 0, :]
        right_data = data.loc[data["f1"] == 1, :]
        self.assertEqual(self.dt_classifier.calculate_split_gini_index(self.dt_classifier.data, left_data, right_data),
                         1 - (0.5**2 + 0.5**2))

    def test_calculate_split_gini_index_f2(self):
        data = self.dt_classifier.data
        left_data = data.loc[data["f2"] == 0, :]
        right_data = data.loc[data["f2"] == 1, :]
        self.assertEqual(self.dt_classifier.calculate_split_gini_index(self.dt_classifier.data, left_data, right_data),
                         1 - (0.7**2 + 0.3**2))

    def test_calculate_split_gini_index_f3(self):
        data = self.dt_classifier.data
        left_data = data.loc[data["f3"] == 0, :]
        right_data = data.loc[data["f3"] == 1, :]
        self.assertEqual(self.dt_classifier.calculate_split_gini_index(self.dt_classifier.data, left_data, right_data),
                         1 - (0.6**2 + 0.4**2))

    def test_calculate_info_gain_f1(self):
        data = self.dt_classifier.data
        self.assertAlmostEqual(self.dt_classifier.calculate_info_gain(self.dt_classifier.calculate_entropy(data), data, "f1"), 0.1245112498)

    def test_calculate_info_gain_f5(self):
        data = self.dt_classifier.data
        self.assertAlmostEqual(self.dt_classifier.calculate_info_gain(self.dt_classifier.calculate_entropy(data), data, "f5"), 0.1735337494)

    def test_calculate_gini_gain_f2(self):
        data = self.dt_classifier.data
        self.assertAlmostEqual(self.dt_classifier.calculate_gini_gain(self.dt_classifier.calculate_gini_index(data), data, "f2"), 0.0542857143)

    def test_calculate_gini_gain_f7(self):
        data = self.dt_classifier.data
        self.assertAlmostEqual(self.dt_classifier.calculate_gini_gain(self.dt_classifier.calculate_gini_index(data), data, "f7"), 0.123333333333)

    def test_train(self):
        self.dt_classifier.train()
        self.assertEqual(self.dt_classifier.predict([0, 1, 0, 1, 1, 1, 0]), "B")
        self.assertEqual(self.dt_classifier.predict([0, 1, 0, 1, 1, 0, 0]), "A")


