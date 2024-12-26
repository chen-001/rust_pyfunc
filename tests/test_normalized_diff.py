from rust_pyfunc.rust_pyfunc import normalized_diff

def test_normalized_diff():
    # 测试用例1：原始例子中的DA和DB
    da = "We expect demand to increase"
    db = "We expect worldwide demand to increase"
    score = normalized_diff(da, db)
    print(f"Score between DA and DB: {score}")  # 应该是一个较小的值，因为只添加了一个词
    # 1个添加，总词数平均是(5+6)/2=5.5，所以分数应该约为0.182
    assert abs(score - 0.182) < 0.001

    # 测试用例2：原始例子中的DA和DC
    dc = "We expect weakness in sales"
    score = normalized_diff(da, db)
    print(f"Score between DA and DC: {score}")  # 应该是一个较大的值，因为有多个更改

    # 测试用例3：完全相同的文档
    text1 = "The quick brown fox jumps over the lazy dog"
    score = normalized_diff(text1, text1)
    print(f"Score for identical documents: {score}")  # 应该是0
    assert score == 0.0

    # 测试用例4：大小写和标点符号不同
    text1 = "Hello, World!"
    text2 = "hello world"
    score = normalized_diff(text1, text2)
    print(f"Score with different case and punctuation: {score}")  # 应该是0
    assert score == 0.0

    # 测试用例5：完全不同的文档
    text1 = "The quick brown fox"
    text2 = "jumps over lazy dog"
    score = normalized_diff(text1, text2)
    print(f"Score for completely different documents: {score}")  # 应该接近2.0
    # 4个删除，4个添加，总词数平均是4，所以分数应该是2.0
    assert abs(score - 2.0) < 0.001

    # 测试用例6：空文档
    score = normalized_diff("", "")
    print(f"Score for empty documents: {score}")  # 应该是0
    assert score == 0.0

if __name__ == "__main__":
    test_normalized_diff()
    print("All tests passed successfully!")
