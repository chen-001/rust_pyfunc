use std::{env, path::PathBuf};
use yupei_refined::{
    compute::{self, CHAINS},
    input,
    model::Config,
};
fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    }
}
fn run() -> std::io::Result<()> {
    let mut config = Config {
        date: 0,
        jobs: 50,
        level2_root: env::var_os("RUST_PYFUNC_LEVEL2_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| "/ssd_data/stock".into()),
        basic_info_dir: env::var_os("RUST_PYFUNC_BASIC_INFO_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| "/ssd_data/data/basic_info".into()),
        vars_dir: env::var_os("RUST_PYFUNC_VARS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| "/ssd_data/data/vars".into()),
        history: Vec::new(),
        allow_incomplete_history: false,
        allow_uncomputable_corr: false,
        codes: None,
        chains: CHAINS.iter().map(|s| s.to_string()).collect(),
    };
    let mut output = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--date" => {
                config.date = args
                    .next()
                    .ok_or_else(|| input::bad("--date missing value"))?
                    .parse()
                    .map_err(|_| input::bad("invalid date"))?
            }
            "--jobs" => {
                config.jobs = args
                    .next()
                    .ok_or_else(|| input::bad("--jobs missing value"))?
                    .parse()
                    .map_err(|_| input::bad("invalid jobs"))?
            }
            "--output" => {
                output = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| input::bad("--output missing value"))?,
                ))
            }
            "--history" => config.history.push(PathBuf::from(
                args.next()
                    .ok_or_else(|| input::bad("--history missing value"))?,
            )),
            "--daily-only" | "--allow-incomplete-history" => config.allow_incomplete_history = true,
            "--allow-uncomputable-corr" => config.allow_uncomputable_corr = true,
            "--codes" => {
                config.codes = Some(
                    args.next()
                        .ok_or_else(|| input::bad("--codes missing value"))?
                        .split(',')
                        .map(str::to_owned)
                        .collect(),
                )
            }
            "--help" => {
                println!("yupei_refined --date YYYYMMDD --output result.json [--jobs 50] [--history previous.snapshot.json ...] [--daily-only]\nSingle-day sandbox, no submission or batch. --daily-only explicitly permits null historical columns. --codes is an explicit test subset. Data roots use RUST_PYFUNC_LEVEL2_PATH/BASIC_INFO_DIR/VARS_DIR.");
                return Ok(());
            }
            _ => return Err(input::bad(format!("unknown option {arg}"))),
        }
    }
    if config.date < 19900101 {
        return Err(input::bad("--date is required"));
    }
    if config.history.is_empty() && !config.allow_incomplete_history {
        return Err(input::bad("historical features require explicit --history snapshots; use --daily-only for single-day exploration with null historical columns"));
    }
    let output = output.ok_or_else(|| input::bad("--output is required"))?;
    compute::write_result(&config, &output)
}
