fn main() {
    pkg_config::probe_library("openblas").unwrap();
} 