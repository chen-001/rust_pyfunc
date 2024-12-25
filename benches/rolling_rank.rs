// use criterion::{black_box, criterion_group, criterion_main, Criterion};
// use pyfunc::pandas_ext::rolling_window_stat;
// use numpy::ndarray::Array1;

// fn bench_rolling_rank(c: &mut Criterion) {
//     let n = 10000;
//     let window_ns = 1000.0;
//     let mut times = Vec::with_capacity(n);
//     let mut values = Vec::with_capacity(n);
    
//     // 生成测试数据
//     for i in 0..n {
//         times.push(i as f64);
//         values.push(rand::random::<f64>());
//     }
    
//     c.bench_function("rolling_rank", |b| {
//         b.iter(|| {
//             rolling_window_stat(
//                 black_box(times.clone()),
//                 black_box(values.clone()),
//                 black_box(window_ns),
//                 black_box("rank"),
//                 black_box(false)
//             )
//         })
//     });
// }

// criterion_group!(benches, bench_rolling_rank);
// criterion_main!(benches);


// "rank" => {
//             // 对于 rank 统计，忽略 include_current 参数，始终包含当前值
//             // 预估窗口大小，避免频繁的内存分配
//             let window_size = (window_ns / (times[1] - times[0])).ceil() as usize;
//             let mut window_values: VecDeque<(f64, usize)> = VecDeque::with_capacity(window_size.min(n));
//             let mut window_end = 0;
            
//             for i in 0..n {
//                 // 移除窗口前面的值
//                 while !window_values.is_empty() && window_values.front().unwrap().1 < i {
//                     window_values.pop_front();
//                 }
                
//                 // 添加新的值到窗口
//                 let target_time = times[i] + window_ns;
//                 while window_end < n && times[window_end] <= target_time {
//                     let val = values[window_end];
//                     let pos = match window_values.binary_search_by(|(x, _)| x.partial_cmp(&val).unwrap_or(std::cmp::Ordering::Equal)) {
//                         Ok(pos) | Err(pos) => pos,
//                     };
//                     window_values.insert(pos, (val, window_end));
//                     window_end += 1;
//                 }
                
//                 // 计算排名
//                 let window_len = window_values.len();
//                 if window_len > 1 {
//                     let current_value = values[i];
                    
//                     // 使用二分查找找到当前值的位置
//                     match window_values.binary_search_by(|(x, _)| x.partial_cmp(&current_value).unwrap_or(std::cmp::Ordering::Equal)) {
//                         Ok(pos) => {
//                             // 处理相等值
//                             let mut equal_start = pos;
//                             let mut equal_end = pos;
                            
//                             // 向前搜索相等值
//                             while equal_start > 0 {
//                                 if (window_values[equal_start - 1].0 - current_value).abs() < 1e-10 {
//                                     equal_start -= 1;
//                                 } else {
//                                     break;
//                                 }
//                             }
                            
//                             // 向后搜索相等值
//                             while equal_end < window_len - 1 {
//                                 if (window_values[equal_end + 1].0 - current_value).abs() < 1e-10 {
//                                     equal_end += 1;
//                                 } else {
//                                     break;
//                                 }
//                             }
                            
//                             result[i] = (equal_start + equal_end) as f64 / (2.0 * (window_len - 1) as f64);
//                         },
//                         Err(pos) => {
//                             result[i] = pos as f64 / (window_len - 1) as f64;
//                         }
//                     }
//                 }
//             }
//         },