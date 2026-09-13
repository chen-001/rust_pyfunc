//! 最小 npy 读取器（只支持需要的 dtype + C 顺序），供 sandbox 基准从
//! Python 导出的 .npy 文件加载数据。与 numpy.save 的 v1.0 格式兼容。
use std::fs;

pub enum NpyArray {
    F32 { shape: Vec<usize>, data: Vec<f32> },
    F64 { shape: Vec<usize>, data: Vec<f64> },
    I32 { shape: Vec<usize>, data: Vec<i32> },
    I64 { shape: Vec<usize>, data: Vec<i64> },
}

pub fn load(path: &str) -> NpyArray {
    let bytes = fs::read(path).expect("read npy");
    assert_eq!(&bytes[..6], b"\x93NUMPY", "bad magic");
    let major = bytes[6];
    assert_eq!(major, 1, "only npy v1 supported");
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
    let descr = header
        .split("'descr': '")
        .nth(1)
        .unwrap()
        .split('\'')
        .next()
        .unwrap()
        .to_string();
    let fortran = header.contains("'fortran_order': True");
    let shape: Vec<usize> = header
        .split("'shape': (")
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap()
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().trim_end_matches('L').parse().unwrap())
        .collect();
    let data = &bytes[10 + header_len..];
    if fortran {
        // F 序 → C 序（仅支持 1D/2D）
        let n = data.len();
        if shape.len() == 2 {
            let (rows, cols) = (shape[0], shape[1]);
            let elem = match descr.as_str() {
                "<f4" | "|f4" => 4,
                "<f8" | "|f8" => 8,
                "<i4" => 4,
                "<i8" => 8,
                _ => 8,
            };
            let mut corder = vec![0u8; n];
            for i in 0..rows {
                for j in 0..cols {
                    let src = (j * rows + i) * elem;
                    let dst = (i * cols + j) * elem;
                    corder[dst..dst + elem].copy_from_slice(&data[src..src + elem]);
                }
            }
            return parse_payload(&corder, shape, &descr);
        }
    }
    parse_payload(data, shape, &descr)
}

fn parse_payload(data: &[u8], shape: Vec<usize>, descr: &str) -> NpyArray {
    match descr {
        "<f4" | "|f4" => NpyArray::F32 {
            data: data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            shape,
        },
        "<f8" | "|f8" => NpyArray::F64 {
            data: data
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            shape,
        },
        "<i4" => NpyArray::I32 {
            data: data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            shape,
        },
        "<i8" => NpyArray::I64 {
            data: data
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            shape,
        },
        other => panic!("unsupported dtype {other}"),
    }
}

pub fn as_f32_mat(a: NpyArray) -> ndarray::Array2<f32> {
    match a {
        NpyArray::F32 { shape, data } => {
            ndarray::Array2::from_shape_vec((shape[0], shape[1]), data).unwrap()
        }
        _ => panic!("not f32"),
    }
}

pub fn as_f32_vec1(a: NpyArray) -> Vec<f32> {
    match a {
        NpyArray::F32 { shape, data } => {
            assert_eq!(shape.len(), 1);
            data
        }
        _ => panic!("not f32 vec"),
    }
}

pub fn as_i32_vec1(a: NpyArray) -> Vec<i32> {
    match a {
        NpyArray::I32 { shape, data } => {
            assert_eq!(shape.len(), 1);
            data
        }
        _ => panic!("not i32 vec"),
    }
}

pub fn as_f64_3d(a: NpyArray) -> ndarray::Array3<f64> {
    match a {
        NpyArray::F64 { shape, data } => {
            ndarray::Array3::from_shape_vec((shape[0], shape[1], shape[2]), data).unwrap()
        }
        _ => panic!("not f64 3d"),
    }
}

pub fn as_f64_mat(a: NpyArray) -> ndarray::Array2<f64> {
    match a {
        NpyArray::F64 { shape, data } => {
            ndarray::Array2::from_shape_vec((shape[0], shape[1]), data).unwrap()
        }
        _ => panic!("not f64"),
    }
}
