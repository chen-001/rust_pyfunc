use numpy::{PyArray, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// 高性能计算差值矩阵 (SIMD优化版本)
///
/// 输入一个一维数组，返回一个二维数组，其中第i行第j列的元素是输入数组第i个元素和第j个元素的差值
///
/// 优化策略:
/// 1. 使用AVX2 SIMD指令集加速向量化计算 (一次处理4个f64)
/// 2. 优化内存访问模式提升缓存命中率
/// 3. 循环展开减少分支预测失败
/// 4. 内存预分配减少动态分配开销
///
/// # Arguments
///
/// * `data` - 输入的一维数组 (numpy array)
///
/// # Returns
///
/// * `PyResult<&PyArray2<f64>>` - 返回的差值矩阵
#[pyfunction]
#[pyo3(name = "difference_matrix")]
pub fn difference_matrix<'a>(py: Python<'a>, data: PyReadonlyArray1<'a, f64>) -> PyResult<&'a PyArray2<f64>> {
    let input_array = data.as_slice()?;
    let n = input_array.len();

    if n == 0 {
        return Ok(PyArray::zeros(py, [0, 0], false));
    }

    // 创建输出矩阵，预分配内存
    let mut result = vec![0.0f64; n * n];
    let input_ptr = input_array.as_ptr();
    let result_ptr = result.as_mut_ptr();

    // 使用优化的SIMD计算
    difference_matrix_simd_optimized(input_ptr, result_ptr, n);

    // 将结果转换为numpy数组
    let result_array = PyArray::from_vec(py, result);
    let result_2d = unsafe {
        result_array.reshape([n, n])?
    };

    Ok(result_2d)
}

/// 高度优化的SIMD差值矩阵计算 (单线程)
///
/// 优化技巧:
/// 1. 使用AVX2指令集一次处理4个f64
/// 2. 循环展开减少分支预测失败
/// 3. 内存访问优化提高缓存命中率
#[inline]
fn difference_matrix_simd_optimized(input_ptr: *const f64, result_ptr: *mut f64, n: usize) {
    // 检查AVX2是否可用，如果不可用则回退到标量版本
    #[cfg(target_arch = "x86_64")]
    {
        let has_avx2 = is_x86_feature_detected!("avx2");
        if has_avx2 && n >= 4 {
            difference_matrix_avx2(input_ptr, result_ptr, n);
            return;
        }
    }

    // 回退到标量版本
    difference_matrix_scalar(input_ptr, result_ptr, n);
}

/// AVX2版本 - 一次处理4个f64
#[cfg(target_arch = "x86_64")]
#[inline]
fn difference_matrix_avx2(input_ptr: *const f64, result_ptr: *mut f64, n: usize) {
    let simd_chunks = n / 4;

    for i in 0..n {
        let base_value = unsafe { *input_ptr.add(i) };
        let row_ptr = unsafe { result_ptr.add(i * n) };

        // 创建广播向量 (base_value, base_value, base_value, base_value)
        let base_vec = unsafe { _mm256_set1_pd(base_value) };

        // 处理4个元素的块
        for chunk in 0..simd_chunks {
            let offset = chunk * 4;

            let input_chunk = unsafe {
                let ptr = input_ptr.add(offset);
                _mm256_loadu_pd(ptr)
            };

            let diff_chunk = unsafe { _mm256_sub_pd(base_vec, input_chunk) };

            unsafe {
                let dst_ptr = row_ptr.add(offset);
                _mm256_storeu_pd(dst_ptr, diff_chunk);
            }
        }

        // 处理剩余元素
        for j in simd_chunks * 4..n {
            unsafe {
                let input_val = *input_ptr.add(j);
                *row_ptr.add(j) = base_value - input_val;
            }
        }
    }
}

/// 标量版本 - 处理小规模数据或SIMD不可用的情况
#[inline]
fn difference_matrix_scalar(input_ptr: *const f64, result_ptr: *mut f64, n: usize) {
    for i in 0..n {
        let base_value = unsafe { *input_ptr.add(i) };
        let row_ptr = unsafe { result_ptr.add(i * n) };

        // 简单的循环展开 (展开4次)
        let unrolled_chunks = n / 4;
        let remainder = n % 4;

        for chunk in 0..unrolled_chunks {
            let offset = chunk * 4;
            unsafe {
                *row_ptr.add(offset) = base_value - *input_ptr.add(offset);
                *row_ptr.add(offset + 1) = base_value - *input_ptr.add(offset + 1);
                *row_ptr.add(offset + 2) = base_value - *input_ptr.add(offset + 2);
                *row_ptr.add(offset + 3) = base_value - *input_ptr.add(offset + 3);
            }
        }

        // 处理剩余元素
        for j in unrolled_chunks * 4..n {
            unsafe {
                *row_ptr.add(j) = base_value - *input_ptr.add(j);
            }
        }
    }
}

/// 内存高效版本的差值矩阵计算 (针对超大矩阵优化)
///
/// 使用分块计算策略提高缓存利用率，减少内存带宽瓶颈
#[pyfunction]
#[pyo3(name = "difference_matrix_memory_efficient")]
pub fn difference_matrix_memory_efficient<'a>(py: Python<'a>, data: PyReadonlyArray1<'a, f64>) -> PyResult<&'a PyArray2<f64>> {
    let input_array = data.as_slice()?;
    let n = input_array.len();

    if n == 0 {
        return Ok(PyArray::zeros(py, [0, 0], false));
    }

    // 直接创建numpy数组，避免中间Vec分配
    let result = unsafe { PyArray::new(py, [n, n], false) };
    let mut result_slice = unsafe { result.as_slice_mut()? };
    let input_ptr = input_array.as_ptr();
    let result_ptr: *mut f64 = result_slice.as_mut_ptr();

    // 使用分块计算策略，提高缓存利用率
    const BLOCK_SIZE: usize = 256;

    for block_i in (0..n).step_by(BLOCK_SIZE) {
        let end_i = (block_i + BLOCK_SIZE).min(n);

        for block_j in (0..n).step_by(BLOCK_SIZE) {
            let end_j = (block_j + BLOCK_SIZE).min(n);

            // 计算当前块
            for i in block_i..end_i {
                let base_value = unsafe { *input_ptr.add(i) };
                let row_ptr = unsafe { result_ptr.add(i * n + block_j) };

                // 向量化计算块内数据
                let block_width = end_j - block_j;

                #[cfg(target_arch = "x86_64")]
                {
                    let has_avx2 = is_x86_feature_detected!("avx2");
                    if has_avx2 && block_width >= 4 {
                        unsafe {
                            compute_block_avx2(base_value, input_ptr.add(block_j), row_ptr, block_width);
                        }
                        continue;
                    }
                }

                // 回退到标量版本
                unsafe {
                    compute_block_scalar(base_value, input_ptr.add(block_j), row_ptr, block_width);
                }
            }
        }
    }

    Ok(result)
}


/// 使用AVX2计算一个数据块
#[cfg(target_arch = "x86_64")]
#[inline]
fn compute_block_avx2(base_value: f64, input_ptr: *const f64, result_ptr: *mut f64, width: usize) {
    let base_vec = unsafe { _mm256_set1_pd(base_value) };
    let simd_chunks = width / 4;

    for chunk in 0..simd_chunks {
        let offset = chunk * 4;
        let input_chunk = unsafe {
            let ptr = input_ptr.add(offset);
            _mm256_loadu_pd(ptr)
        };
        let diff_chunk = unsafe { _mm256_sub_pd(base_vec, input_chunk) };
        unsafe {
            let dst_ptr = result_ptr.add(offset);
            _mm256_storeu_pd(dst_ptr, diff_chunk);
        }
    }

    // 处理剩余元素
    for j in simd_chunks * 4..width {
        unsafe {
            let input_val = *input_ptr.add(j);
            *result_ptr.add(j) = base_value - input_val;
        }
    }
}

/// 使用标量计算一个数据块
#[inline]
fn compute_block_scalar(base_value: f64, input_ptr: *const f64, result_ptr: *mut f64, width: usize) {
    for j in 0..width {
        unsafe {
            let input_val = *input_ptr.add(j);
            *result_ptr.add(j) = base_value - input_val;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_difference_matrix_small() {
        let data = vec![1.0, 2.0, 3.0];
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let input_array = PyArray1::from_vec(py, data);
            let result = difference_matrix(py, input_array.readonly()).unwrap();

            // 验证结果
            let result_slice = result.as_slice().unwrap();
            let expected = vec![
                0.0, -1.0, -2.0,  // 1.0 - [1.0, 2.0, 3.0]
                1.0,  0.0, -1.0,  // 2.0 - [1.0, 2.0, 3.0]
                2.0,  1.0,  0.0   // 3.0 - [1.0, 2.0, 3.0]
            ];

            assert_eq!(result_slice, &expected);
        });
    }
}