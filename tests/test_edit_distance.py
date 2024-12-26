from rust_pyfunc.rust_pyfunc import min_word_edit_distance

def test_min_word_edit_distance():
    # 测试用例1：从DA到DB（添加一个单词）
    da = "We expect demand to increase"
    db = "We expect worldwide demand to increase"
    dist = min_word_edit_distance(da, db)
    print(f"Distance from DA to DB: {dist}")  # 应该是1（添加"worldwide"）
    assert dist == 1

    # 测试用例2：从DA到DC（删除3个词，添加3个词）
    dc = "We expect weakness in sales"
    dist = min_word_edit_distance(da, dc)
    print(f"Distance from DA to DC: {dist}")  # 应该是6（删除3个词，添加3个词）
    assert dist == 6

    # 测试用例3：相同的句子
    dist = min_word_edit_distance(da, da)
    print(f"Distance between identical sentences: {dist}")  # 应该是0
    assert dist == 0

    # 测试用例4：大小写和标点符号不同
    text1 = "Hello, World!"
    text2 = "hello world"
    dist = min_word_edit_distance(text1, text2)
    print(f"Distance with different case and punctuation: {dist}")  # 应该是0
    assert dist == 0

    # 测试用例5：完全不同的句子
    text1 = "The quick brown fox"
    text2 = "jumps over lazy dog"
    dist = min_word_edit_distance(text1, text2)
    print(f"Distance between completely different sentences: {dist}")  # 所有词都不同
    assert dist == 8  # 4个词需要删除，4个词需要添加

if __name__ == "__main__":
    test_min_word_edit_distance()
    print("All tests passed successfully!")
