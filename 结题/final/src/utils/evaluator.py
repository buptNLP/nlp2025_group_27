import re

class Evaluator:
    def __init__(self):
        # 匹配 "#### 数字" 格式的答案
        self.ans_re = re.compile(r"####\s*([\d\.]+)")
        # 匹配步骤中的数字和运算符
        self.step_re = re.compile(r"[\d\.]+[+\-*/=]+[\d\.]+")

    def exact_match(self, pred, gt):
        """
        判断预测答案和真实答案是否精确匹配。
        """
        pred_match = self.ans_re.search(pred)
        gt_match = self.ans_re.search(gt)

        if pred_match and gt_match:
            pred_num = pred_match.group(1)
            gt_num = gt_match.group(1)
            return int(pred_num == gt_num)
        else:
            return 0  # 如果没有找到匹配的答案，则返回 0

    def stepwise_score(self, pred_steps, gt_steps):
        """
        计算预测步骤和真实步骤的分步得分。
        """
        matched = 0
        for gt_step in gt_steps:
            # 使用正则表达式匹配步骤中的关键信息
            gt_step_match = self.step_re.findall(gt_step)
            if not gt_step_match:
                continue  # 如果真实步骤无法解析，则跳过

            for pred_step in pred_steps:
                pred_step_match = self.step_re.findall(pred_step)
                if not pred_step_match:
                    continue  # 如果预测步骤无法解析，则跳过

                # 比较预测步骤和真实步骤的关键信息
                if set(gt_step_match).issubset(set(pred_step_match)):
                    matched += 1
                    break  # 找到匹配的预测步骤，则跳出内层循环

        return matched / len(gt_steps) if gt_steps else 0 