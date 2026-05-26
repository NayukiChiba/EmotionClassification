"""
评估指标单元测试

测试极端值（完美预测、随机预测、全错预测）下各项指标的合理性。
"""

import numpy as np

from src.evaluate.metrics import compute_metrics


class TestMetrics:
    """评估指标测试"""

    def test_perfect_predictions(self):
        """测试完美预测时指标应为 1.0"""
        y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        report = compute_metrics(y_true, y_pred, y_prob)

        assert report.accuracy == 1.0
        assert report.precision == 1.0
        assert report.recall == 1.0
        assert report.f1_score == 1.0
        assert report.auc == 1.0

    def test_all_wrong_predictions(self):
        """测试全错预测时指标应为 0.0（或接近 0）"""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

        report = compute_metrics(y_true, y_pred, y_prob)

        assert report.accuracy == 0.0
        assert report.precision == 0.0
        assert report.recall == 0.0
        assert report.f1_score == 0.0
        # AUC：正类概率全 0、负类概率全 1 -> 预测方向完全反转 -> AUC = 0.0
        assert report.auc == 0.0

    def test_balanced_random_predictions(self):
        """测试随机预测时指标应在合理范围内"""
        np.random.seed(42)
        n = 200
        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.randint(0, 2, n)
        y_prob = np.random.rand(n)

        report = compute_metrics(y_true, y_pred, y_prob)

        # 随机预测下指标应接近 0.5
        assert 0.3 < report.accuracy < 0.7
        assert 0.3 < report.precision < 0.7
        assert 0.3 < report.recall < 0.7
        # AUC 对于随机概率也是接近 0.5
        assert 0.2 < report.auc < 0.8

    def test_all_positive_predictions(self):
        """测试全部预测为正类时的指标"""
        y_true = np.array([1, 1, 1, 0, 0, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.9, 0.8, 0.7, 0.6])

        report = compute_metrics(y_true, y_pred, y_prob)

        # 全正预测下，准确率 = 正类占比 = 4/8 = 0.5
        assert report.accuracy == 0.5
        # precision = TP/(TP+FP) = 4/8 = 0.5
        assert report.precision == 0.5
        # recall = TP/(TP+FN) = 4/4 = 1.0
        assert report.recall == 1.0
        # f1 = 2 * 0.5 * 1.0 / (0.5 + 1.0) = 1.0 / 1.5 ≈ 0.6667
        assert abs(report.f1_score - 0.666667) < 0.01
        # AUC for perfect separation inside this group should be valid
        assert 0.0 <= report.auc <= 1.0

    def test_all_negative_predictions(self):
        """测试全部预测为负类时的指标"""
        y_true = np.array([1, 1, 0, 0, 1, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0.4])

        report = compute_metrics(y_true, y_pred, y_prob)

        # 全负预测下，准确率 = 负类占比 = 5/8
        assert report.accuracy == 5 / 8
        # precision = TP/(TP+FP) = 0/0 -> zero_division=0
        assert report.precision == 0.0
        # recall = TP/(TP+FN) = 0/3 = 0.0
        assert report.recall == 0.0
        assert report.f1_score == 0.0

    def test_auc_binary_extremes(self):
        """测试 AUC 在二分类极端条件的边界值"""
        # 正类概率 > 负类概率 -> 完美排序
        y_true = np.array([1, 1, 0, 0])
        y_prob_perfect = np.array([0.9, 0.8, 0.3, 0.2])
        report = compute_metrics(
            y_true, (y_prob_perfect >= 0.5).astype(int), y_prob_perfect
        )
        assert report.auc == 1.0

        # 正类概率 < 负类概率 -> 完全反转
        y_prob_reversed = np.array([0.1, 0.2, 0.8, 0.9])
        report = compute_metrics(
            y_true, (y_prob_reversed >= 0.5).astype(int), y_prob_reversed
        )
        assert report.auc == 0.0

    def test_classification_report_not_empty(self):
        """测试分类报告字符串非空"""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 1])
        y_prob = np.array([0.8, 0.2, 0.7, 0.6])

        report = compute_metrics(y_true, y_pred, y_prob)
        assert len(report.classification_report) > 0
        assert "precision" in report.classification_report.lower()
        assert "recall" in report.classification_report.lower()
