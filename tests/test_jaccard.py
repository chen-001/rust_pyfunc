import sys
import os

# Import the Rust module
from rust_pyfunc.rust_pyfunc import jaccard_similarity

def test_jaccard_similarity():
    # 测试用例1：完全相同的句子
    text1 = "Hello World!"
    text2 = "Hello World!"
    assert jaccard_similarity(text1, text2) == 1.0
    print(f"Test 1 passed: {text1} vs {text2} = {jaccard_similarity(text1, text2)}")

    # 测试用例2：不同大小写和标点符号
    text1 = "Hello, World!"
    text2 = "HELLO WORLD"
    assert jaccard_similarity(text1, text2) == 1.0
    print(f"Test 2 passed: {text1} vs {text2} = {jaccard_similarity(text1, text2)}")

    # 测试用例3：部分重叠的句子
    text1 = "The quick brown fox"
    text2 = "The lazy brown dog"
    sim = jaccard_similarity(text1, text2)
    print(f"Test 3: {text1} vs {text2} = {sim}")
    assert 0 < sim < 1

    # 测试用例4：完全不同的句子
    text1 = "Hello world"
    text2 = "Goodbye universe"
    sim = jaccard_similarity(text1, text2)
    print(f"Test 4: {text1} vs {text2} = {sim}")
    assert sim == 0.0

    # 测试用例5：包含标点符号和多余空格的句子
    text1 = "Hello,   World!!!"
    text2 = "hello... world"
    assert jaccard_similarity(text1, text2) == 1.0
    print(f"Test 5 passed: {text1} vs {text2} = {jaccard_similarity(text1, text2)}")

if __name__ == "__main__":
    test_jaccard_similarity()
    print("All tests passed successfully!")


import Levenshtein
Levenshtein.distance