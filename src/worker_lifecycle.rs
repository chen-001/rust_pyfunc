//! Worker lifetime protection, configured once at startup, never in the compute loop.
//!
//! Register after exec, not in Command::pre_exec: pre_exec disables Rust's
//! posix_spawn fast path and can add fork/page-table costs for large Python jobs.

pub const PARENT_PID_ENV: &str = "RUST_PYFUNC_WORKER_PARENT_PID";

/// Always overwrite inherited values, including when a manager restarts a worker.
pub fn configure_parent(command: &mut std::process::Command) {
    #[cfg(target_os = "linux")]
    command.env(PARENT_PID_ENV, std::process::id().to_string());
    #[cfg(not(target_os = "linux"))]
    let _ = command;
}

/// Called before affinity setup, IPC buffers, and any factor computation.
/// None keeps standalone/older callers compatible; new Linux callers require
/// the protected-ready handshake and will reject an unprotected worker.
pub fn bind_to_parent() -> Result<Option<u32>, String> {
    #[cfg(target_os = "linux")]
    {
        let value = match std::env::var(PARENT_PID_ENV) {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => return Ok(None),
            Err(error) => return Err(format!("invalid {PARENT_PID_ENV}: {error}")),
        };
        let expected = value
            .parse::<libc::pid_t>()
            .ok()
            .filter(|pid| *pid > 0)
            .ok_or_else(|| format!("invalid {PARENT_PID_ENV}: {value:?}"))?;
        // Linux follows the thread that spawned the worker. Our dedicated
        // manager threads must remain alive until their worker has been reaped.
        let result = unsafe {
            libc::prctl(
                libc::PR_SET_PDEATHSIG,
                libc::SIGKILL as libc::c_ulong,
                0 as libc::c_ulong,
                0 as libc::c_ulong,
                0 as libc::c_ulong,
            )
        };
        if result != 0 {
            return Err(format!(
                "PR_SET_PDEATHSIG failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        // Parent may have died before registration. Use the PID supplied by
        // the spawning parent, not a getppid() snapshot taken after exec.
        let actual = unsafe { libc::getppid() };
        if actual != expected {
            return Err(format!(
                "parent exited during worker startup (expected {expected}, actual {actual})"
            ));
        }
        Ok(Some(expected as u32))
    }
    #[cfg(not(target_os = "linux"))]
    {
        Ok(None)
    }
}
