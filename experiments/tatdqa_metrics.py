"""Secondary TAT-DQA metrics for ONE string span with empty gold/predicted scale.

Formatting helpers are copied from the creator-endorsed Doc2SoarGraph evaluator,
commit 71a02715849942b79c1f74934a139e4b56fc6826. The wrapper specializes its
single-span path, preserving normalization quirks, literal-empty prediction
handling, numeric conversion and NumPy's two-decimal F1 rounding. It does not
supply a gold scale to the prediction and does not implement multi-span,
arithmetic/count, date, or dataset-level evaluation.

Upstream license (applies to the copied helpers):

MIT License

Copyright (c) 2023 Fengbin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import re
import string
from typing import List, Set

import numpy as np

__all__ = ["score_official", "UPSTREAM_COMMIT", "UPSTREAM_SHA256"]
UPSTREAM_COMMIT = "71a02715849942b79c1f74934a139e4b56fc6826"
UPSTREAM_SHA256 = {
    "tatqa_metric.py": "dfca76b510de264ecd0bbad70356ee4aabef95014cb50cd273c8cc3189962d88",
    "tatqa_utils.py": "a3af9c8110ec800b4247e54c0bbaebe3443ba0eeab75203d500bcb1c3a2b69e5",
    "LICENSE": "966940980b5ba990bd61791094bc3aece4928b65c87236d576dc76a9d061f2bf",
}

# Begin unchanged upstream helper bodies (unused dependencies are omitted).
def scale_to_num(scale):
    scale = scale.lower()
    num = 1
    if 'hundred' in scale:  # hundred
        num = 100
    elif 'thousand' in scale:  # thousand
        num = 1000
    elif 'million' in scale:  # million
        num = 1000000
    elif 'billion' in scale:  # billion
        num = 1000000000
    elif 'percent' in scale:  # percent
        num = 0.01
    return num

def extract_one_num_from_str(s):
    s = _clean_num(s)
    r_num = r"([+-]?\d+(\.\d+)?)|([+-]?\.\d+)"
    groups = re.findall(r_num, s)
    if len(groups) == 0:
        return None
    num = groups[0][0]
    if num == '':
        return None
    if '.' in num:
        return float(num)
    return int(num)

EXCLUDE_IN_NUM = "'\"\\$€£¥%(),[]"

def _clean_num(text:str):
    return "".join([ch for ch in str(text) if ch not in EXCLUDE_IN_NUM])

def is_number(text: str) -> bool:
    try:
        words = " ".join([_clean_num(w) for w in text.split()]).split()
        if len(words) == 0:
            """1023 or 1 million"""
            return False
        num = float(words[0])
        if np.isnan(num):
            return False
        if len(words) >= 2:
            if scale_to_num(words[1]) == 1:
                return False
        return True
    except ValueError:
        return False

def negative_num_handle(x):
    """
    :param x:  transform (134) -> -134
    :return:
    """
    all = re.findall('(\([\d.\s%]+\))', x.strip())
    if len(all) > 0:
        return -1
    return 1

def percent_num_handle(x):
    """
    :param x:  transform 12% -> 12/100
    :return:
    """
    all = re.findall('([\d.\s]+%)', x.strip())
    if len(all) > 0:
        return 0.01
    return 1

def word_scale_handle(x):
    """
    :param x: 1 million = 1,000,000
    :return:
    """
    iter = re.finditer('([\d.]+\s?[a-zA-Z]+)', x)
    for one in iter:
        text = one.group(0).lower()
        scale_val = scale_to_num(text)
        return scale_val
    return 1

def to_number(text:str) -> float:
    num = extract_one_num_from_str(text)
    scale_val = word_scale_handle(text)
    negative_flag = negative_num_handle(text)
    percent_flag = percent_num_handle(text)
    if num is not None:
        return round(num * scale_val * negative_flag * percent_flag, 4)
    return None

def remove_articles(text: str) -> str:
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)

def white_space_fix(text: str) -> str:
    return ' '.join(text.split())

EXCLUDE = set(string.punctuation)

def remove_punc(text: str) -> str:
    if not is_number(text):
        return ''.join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text

def lower(text: str) -> str:
    return text.lower()

def tokenize(text: str) -> List[str]:
    return re.split(" ", text)

def normalize_number(text: str) -> str:
    if is_number(text):
        return str(to_number(text))
    else:
        return text

def normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    parts = [white_space_fix(remove_articles(normalize_number(remove_punc(lower(token)))))
             for token in tokenize(text)]
    parts = [part for part in parts if part.strip()]
    normalized = ' '.join(parts).strip()
    return normalized

def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (2 * precision * recall) / (precision + recall) if not (precision == 0.0 and recall == 0.0) else 0.0
    return f1

def get_answer_str(answers: list, scale: str):
    """
    :param ans_type:  span, multi-span, arithmetic, count
    :param ans_list:
    :param scale: "", thousand, million, billion, percent
    :param mode:
    :return:

    """
    try:
        sorted_ans = sorted(answers, key=lambda x:str(x))
        ans_temp = []
        for ans in sorted_ans:
            ans_str = str(ans)
            if is_number(ans_str):
                ans_num = to_number(ans_str)
                if ans_num is None:
                    if scale:
                        ans_str = ans_str + " " + str(scale)
                else:
                    if '%' in ans_str:  # has been handled the answer itself is a percentage
                        ans_str = '%.4f' % ans_num
                    else:
                        ans_str = '%.4f' % (round(ans_num, 2) * scale_to_num(scale))
            else:
                if scale:
                    ans_str = ans_str + " " + str(scale)
            ans_temp.append(ans_str)
        return [" ".join(ans_temp)]
    except Exception as e:
        print(f'get answer str error:{e}')

    return [""]

# End upstream helpers.
def score_official(prediction: str, target: str) -> dict[str, float]:
    """Score one span using the pinned author's empty-scale EM/F1 semantics.

    Both arguments must be strings. Callers must select single-span examples
    whose gold scale is empty BEFORE calling this function. The function has
    no scale argument and never infers a prediction scale from the target.
    It preserves upstream empty/whitespace behavior; invalid model output
    should be rejected by the caller and represented as an empty prediction.
    """
    if not isinstance(prediction, str) or not isinstance(target, str):
        raise TypeError("prediction and target must be single-span strings")
    if prediction == "":
        return {"official_em": 0.0, "official_f1": 0.0}
    predicted = normalize_answer(get_answer_str([prediction], "")[0])
    gold = normalize_answer(get_answer_str([target], "")[0])
    f1 = _compute_f1(set(predicted.split()), set(gold.split()))
    return {
        "official_em": float(predicted == gold),
        "official_f1": float(np.round(f1, 2)),
    }
