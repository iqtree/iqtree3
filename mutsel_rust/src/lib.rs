#![allow(non_snake_case)]

pub mod data;
pub mod felsenstein;
pub mod gamma;
pub mod io;
pub mod model;
mod optimization;
mod utils;

use std::{
    io::Write,
    mem::MaybeUninit,
    path::Path,
};

use candle_core::{IndexOp, Tensor};
use candle_nn::ops::sigmoid;

use crate::{gamma::log_gamma_pdf, utils::tensor_full};

#[unsafe(no_mangle)]
pub extern "C" fn rust_set_rayon_threads(num_threads: u32) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads as usize)
        .build_global()
        .unwrap_or_else(|_| println!("Rust Threadnumber has already been set. Ignoring."));
}

const STDOUT_FD: libc::c_int = 1;
const STDERR_FD: libc::c_int = 2;

unsafe fn os_dup(fd: libc::c_int) -> libc::c_int {
    unsafe { libc::dup(fd) }
}

unsafe fn os_dup2(src: libc::c_int, dst: libc::c_int) -> libc::c_int {
    unsafe { libc::dup2(src, dst) }
}


unsafe fn os_close(fd: libc::c_int) -> libc::c_int {
    unsafe { libc::close(fd) }
}

#[cfg(unix)]
unsafe fn os_read(fd: libc::c_int, buf: &mut [u8]) -> isize {
    unsafe { libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) as isize }
}

#[cfg(windows)]
unsafe fn os_read(fd: libc::c_int, buf: &mut [u8]) -> isize {
    unsafe { libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len() as libc::c_uint) as isize }
}

#[cfg(unix)]
unsafe fn os_write(fd: libc::c_int, buf: &[u8]) -> isize {
    unsafe { libc::write(fd, buf.as_ptr() as *const libc::c_void, buf.len()) as isize }
}

#[cfg(windows)]
unsafe fn os_write(fd: libc::c_int, buf: &[u8]) -> isize {
    unsafe { libc::write(fd, buf.as_ptr() as *const libc::c_void, buf.len() as libc::c_uint) as isize }
}

#[cfg(unix)]
fn create_pipe() -> std::io::Result<[libc::c_int; 2]> {
    let mut fds = [0; 2];
    let ret = unsafe { libc::pipe(fds.as_mut_ptr()) };
    if ret != 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(fds)
    }
}

#[cfg(windows)]
fn create_pipe() -> std::io::Result<[libc::c_int; 2]> {
    let mut fds = [0; 2];
    // libc::pipe on Windows maps to the CRT signature (fds, psize, textmode).
    let ret = unsafe { libc::pipe(fds.as_mut_ptr(), 4096, 0) };
    if ret != 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(fds)
    }
}

fn write_all_fd(fd: libc::c_int, mut buf: &[u8]) -> std::io::Result<()> {
    while !buf.is_empty() {
        let n = unsafe { os_write(fd, buf) };
        if n < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        let n = n as usize;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "write to redirected stdout returned 0",
            ));
        }
        buf = &buf[n..];
    }
    Ok(())
}

fn tee_to_file(path: &str) -> (i32, i32, std::thread::JoinHandle<()>) {
    let fds = create_pipe().unwrap_or_else(|e| panic!("tee_to_file: pipe() failed: {}", e));

    // Save original stdout
    let saved_stdout = unsafe { os_dup(STDOUT_FD) };
    if saved_stdout < 0 {
        panic!(
            "tee_to_file: dup(STDOUT_FILENO) failed: {}",
            std::io::Error::last_os_error()
        );
    }
    let saved_stderr = unsafe { os_dup(STDERR_FD) };
    if saved_stderr < 0 {
        panic!(
            "tee_to_file: dup(STDERR_FILENO) failed: {}",
            std::io::Error::last_os_error()
        );
    }

    // Dedicated copy for tee thread so saved_stdout can be restored/closed independently.
    let tee_stdout = unsafe { os_dup(saved_stdout) };
    if tee_stdout < 0 {
        panic!(
            "tee_to_file: dup(saved_stdout) failed: {}",
            std::io::Error::last_os_error()
        );
    }

    // Redirect stdout to pipe write end
    let ret = unsafe { os_dup2(fds[1], STDOUT_FD) };
    if ret < 0 {
        panic!(
            "tee_to_file: dup2(STDOUT_FILENO) failed: {}",
            std::io::Error::last_os_error()
        );
    }
    let ret = unsafe { os_dup2(fds[1], STDERR_FD) };
    if ret < 0 {
        panic!(
            "tee_to_file: dup2(STDERR_FILENO) failed: {}",
            std::io::Error::last_os_error()
        );
    }

    // Close the original write end — stdout/stderr now hold the only references
    let ret = unsafe { os_close(fds[1]) };
    if ret != 0 {
        panic!(
            "tee_to_file: close(pipe write end) failed: {}",
            std::io::Error::last_os_error()
        );
    }

    // Open log file
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(path)
        .unwrap_or_else(|e| panic!("tee_to_file: failed to open '{}': {}", path, e));

    // Spawn thread to read from pipe and write to both
    let handle = std::thread::spawn(move || {
        let mut buffer = [0u8; 1024];

        loop {
            let n = unsafe { os_read(fds[0], &mut buffer) };
            if n == 0 {
                break; // EOF
            }
            if n < 0 {
                let err = std::io::Error::last_os_error();
                if err.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                panic!("tee_to_file: read from pipe failed: {}", err);
            }

            let n = n as usize;
            // Write to original stdout
            write_all_fd(tee_stdout, &buffer[..n])
                .unwrap_or_else(|e| panic!("tee_to_file: write to stdout failed: {}", e));
            // Write to file
            file.write_all(&buffer[..n])
                .unwrap_or_else(|e| panic!("tee_to_file: write to log file failed: {}", e));
        }

        let ret = unsafe { os_close(fds[0]) };
        if ret != 0 {
            panic!(
                "tee_to_file: close(pipe read end) failed: {}",
                std::io::Error::last_os_error()
            );
        }
        let ret = unsafe { os_close(tee_stdout) };
        if ret != 0 {
            panic!(
                "tee_to_file: close(tee stdout) failed: {}",
                std::io::Error::last_os_error()
            );
        }

        file.flush()
            .unwrap_or_else(|e| panic!("tee_to_file: flush log file failed: {}", e));
    });

    (saved_stdout, saved_stderr, handle)
}

fn restore_stdout_stderr(
    saved_stdout: i32,
    saved_stderr: i32,
    handle: std::thread::JoinHandle<()>,
) {
    unsafe {
        // Restoring stdout/stderr drops the last write-end references to the pipe,
        // causing EOF on the read end so the tee thread can exit cleanly.
        os_dup2(saved_stdout, STDOUT_FD);
        os_dup2(saved_stderr, STDERR_FD);
    }
    // Wait for the tee thread to drain and flush all remaining output.
    handle
        .join()
        .unwrap_or_else(|_| panic!("restore_stdout_stderr: tee thread panicked"));
    unsafe {
        os_close(saved_stdout);
        os_close(saved_stderr);
    }
}

pub struct MutselParams {
    pi_reg: f64,
    Mu_reg: f64,
    branch_reg: f64,
}

fn parse_musel_str(model_str: &str) -> MutselParams {
    let model_str = model_str.trim();
    let model_upper = model_str.to_ascii_uppercase();

    let (pi_reg, Mu_reg, branch_reg) = if model_upper == "MUTSEL" {
        (0.65, 1.05, 1.86)
    } else if model_upper.starts_with("MUTSEL{") && model_str.ends_with('}') {
        let params_str = &model_str[7..model_str.len() - 1];
        let values = params_str
            .split('/')
            .map(|value| value.trim().parse::<f64>().unwrap())
            .collect::<Vec<_>>();
        assert!(
            values.len() == 3,
            "Invalid MUTSEL format: expected MUTSEL{{pi_reg/Mu_reg/branch_reg}}, got {}",
            model_str
        );
        (values[0], values[1], values[2])
    } else {
        panic!(
            "Invalid MUTSEL format: expected MUTSEL or MUTSEL{{pi_reg/Mu_reg/branch_reg}}, got {}",
            model_str
        );
    };

    MutselParams {
        pi_reg,
        Mu_reg,
        branch_reg,
    }
}

/// parents: [num_nodes]
/// branch_lengths: [num_nodes]
/// alignment: [num_sites * num_leaves] (row-major)
/// out_site_freq: [num_sites * 20] (row-major)
/// out_rate_matrix: [num_sites * 190] (row-major)
/// out_rate_para: [num_para] from the rate model 1 for gamma, 2 * num_cat for free rate
/// out variables do not need to be initialized
/// prior_R_file and prior_pi_file can be null pointers
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_mutsel(
    parents: *const i32,
    branch_lengths: *const f64,
    alignment: *const u8,
    num_sites: u32,
    num_leaves: u32,
    num_nodes: u32,
    _pi_reg: f64,
    _R_reg: f64,
    model_str: *const std::os::raw::c_char,
    prior_R_file: *const std::os::raw::c_char,
    prior_pi_file: *const std::os::raw::c_char,
    _: *const std::os::raw::c_char,
    verbose: u8,
    out_site_freq: *mut f64,
    out_rate_matrix: *mut f64,
    out_rate_para: *mut f64,
    output_prefix: *const std::os::raw::c_char,
) {
    let out_prefix = unsafe { std::ffi::CStr::from_ptr(output_prefix) }
        .to_str()
        .unwrap();
    let (saved_stdout, saved_stderr, tee_handle) = tee_to_file(&format!("{}.log", out_prefix));

    let parents = unsafe { std::slice::from_raw_parts(parents, num_nodes as usize) };
    let branch_lengths = unsafe { std::slice::from_raw_parts(branch_lengths, num_nodes as usize) };
    let alignment =
        unsafe { std::slice::from_raw_parts(alignment, (num_sites * num_leaves) as usize) };

    let out_site_freq = unsafe {
        std::slice::from_raw_parts_mut(
            out_site_freq as *mut MaybeUninit<f64>,
            (num_sites * 20) as usize,
        )
    };

    let out_rate_matrix = unsafe {
        std::slice::from_raw_parts_mut(
            out_rate_matrix as *mut MaybeUninit<f64>,
            (num_sites * 190) as usize,
        )
    };

    let model_cstr = unsafe { std::ffi::CStr::from_ptr(model_str) };
    let model_str = model_cstr.to_str().unwrap();

    println!("Starting mutsel optimization with {}", model_str);
    // Split at a + sign to separate the MUSTEL part and the rate model part
    let plus_pos = model_str.find("+");
    let (rate_model, mutsel_str) = if let Some(pos) = plus_pos {
        (&model_str[pos + 1..], &model_str[..pos])
    } else {
        ("G1", model_str) // default
    };

    let mutsel_params = parse_musel_str(mutsel_str);

    let rate_model = parse_rate_model(rate_model);

    let out_rate_para = unsafe {
        std::slice::from_raw_parts_mut(
            out_rate_para as *mut MaybeUninit<f64>,
            rate_model.num_para(),
        )
    };

    let felsenstein = io::create_felsenstein_tree(
        parents,
        branch_lengths,
        alignment,
        num_sites as usize,
        num_leaves as usize,
    );

    let prior_R_file = if prior_R_file.is_null() {
        None
    } else {
        let cstr = unsafe { std::ffi::CStr::from_ptr(prior_R_file) };
        Some(Path::new(cstr.to_str().unwrap()))
    };

    let prior_pi_file = if prior_pi_file.is_null() {
        None
    } else {
        let cstr = unsafe { std::ffi::CStr::from_ptr(prior_pi_file) };
        Some(Path::new(cstr.to_str().unwrap()))
    };

    let substitution_model = SubstitutionModel::MutSel;

    let aa_dist = io::get_site_specific_aa_distribution_iqtree(
        alignment,
        num_sites as usize,
        num_leaves as usize,
    );

    let (S, sqrt_pi, rate_para, _substitution_rates) = optimization::optimize_internal(
        felsenstein,
        branch_lengths,
        aa_dist,
        mutsel_params,
        rate_model,
        prior_R_file,
        prior_pi_file,
        substitution_model,
        crate::Verbosity::from_u8(verbose),
        out_prefix,
    )
    .unwrap();

    let (R, pi) = model::phylograd2iqtree_parametrization(&S, &sqrt_pi).unwrap();
    let site_freq = pi.to_vec2().unwrap();

    for site_index in 0..num_sites as usize {
        for aa_index in 0..20 {
            let value = site_freq[site_index][aa_index];
            out_site_freq[site_index * 20 + aa_index].write(value);
        }
    }

    for site_index in 0..num_sites as usize {
        let mut idx = 0;
        // Upper diagonal
        for i in 0..20 {
            for j in (i + 1)..20 {
                out_rate_matrix[site_index * 190 + idx]
                    .write(R.i((site_index, i, j)).unwrap().to_scalar().unwrap());
                idx += 1;
            }
        }
    }
    let rate_para_vec = rate_para.to_vec1().unwrap();
    for i in 0..rate_model.num_para() {
        out_rate_para[i].write(rate_para_vec[i]);
    }
    std::io::stdout().flush().unwrap();
    restore_stdout_stderr(saved_stdout, saved_stderr, tee_handle);
}

#[derive(Debug, Clone, Copy)]
enum RateModel {
    G(usize),
    R(usize),
    X(f64, SiteSpecificRateModel), // this is the strength of the prior
}

impl RateModel {
    fn num_para(&self) -> usize {
        match self {
            RateModel::G(_) => 1,
            RateModel::R(n) => 2 * n,
            RateModel::X(_, _) => 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum SiteSpecificRateModel {
    Triangle,
    Uniform,
    Gamma(f64),     // alpha parameter
    LogNormal(f64), // Mean
}

impl SiteSpecificRateModel {
    fn from_str(s: &str) -> SiteSpecificRateModel {
        let s = s.trim().to_ascii_lowercase();
        if s == "triangle" {
            SiteSpecificRateModel::Triangle
        } else if s == "uniform" {
            SiteSpecificRateModel::Uniform
        } else if let Some(alpha_str) = s.strip_prefix("gamma{").and_then(|s| s.strip_suffix('}')) {
            let alpha: f64 = alpha_str.parse().unwrap();
            SiteSpecificRateModel::Gamma(alpha)
        } else if let Some(mean_str) = s
            .strip_prefix("lognormal{")
            .and_then(|s| s.strip_suffix('}'))
        {
            let mean: f64 = mean_str.parse().unwrap();
            SiteSpecificRateModel::LogNormal(mean)
        } else {
            panic!("Unknown site-specific rate model: {}", s);
        }
    }

    fn penalty(&self, rate_para: &Tensor) -> Tensor {
        match self {
            SiteSpecificRateModel::Triangle => (1.0 - rate_para).unwrap().sum_all().unwrap(),
            SiteSpecificRateModel::Uniform => {
                return tensor_full(1.0, &[]);
            }
            SiteSpecificRateModel::Gamma(alpha) => {
                return log_gamma_pdf(rate_para, &tensor_full(*alpha, &[]))
                    .neg()
                    .unwrap();
            }
            SiteSpecificRateModel::LogNormal(mean) => {
                return (rate_para.log().unwrap() - mean.ln())
                    .unwrap()
                    .powf(2.0)
                    .unwrap();
            }
        }
    }

    fn rates_from_parameters(&self, rate_para: &Tensor) -> Tensor {
        match self {
            SiteSpecificRateModel::Triangle | SiteSpecificRateModel::Uniform => {
                sigmoid(rate_para).unwrap()
            }
            SiteSpecificRateModel::Gamma(_) => {
                rate_para.exp().unwrap() // this will be transformed to rates in the penalty function
            }
            SiteSpecificRateModel::LogNormal(_) => {
                rate_para.exp().unwrap() // this will be transformed to rates in the penalty function
            }
        }
    }

    fn init(&self, num_sites: usize) -> Tensor {
        match self {
            SiteSpecificRateModel::Triangle | SiteSpecificRateModel::Uniform => {
                tensor_full(0.0, &[num_sites])
            }
            SiteSpecificRateModel::Gamma(_) => tensor_full(0.0, &[num_sites]),
            SiteSpecificRateModel::LogNormal(mean) => tensor_full(mean.ln(), &[num_sites]),
        }
    }
}

fn parse_rate_model(rate_model: &str) -> RateModel {
    let mut chars = rate_model.chars();
    let model_type = chars.next().unwrap();
    match model_type {
        'G' => {
            let num_categories: usize = chars.collect::<String>().parse().unwrap();
            RateModel::G(num_categories)
        }
        'R' => {
            let num_categories: usize = chars.collect::<String>().parse().unwrap();
            RateModel::R(num_categories)
        }
        'X' => {
            // X{strength, site_specific_model}
            let params_str = chars.collect::<String>();
            let params_str = params_str.trim();
            assert!(
                params_str.starts_with('{') && params_str.ends_with('}'),
                "Invalid X model format: {}",
                params_str
            );
            let params_str = &params_str[1..params_str.len() - 1]; // remove { and }
            let mut params = params_str.splitn(2, '/');
            let strength: f64 = params.next().unwrap().trim().parse().unwrap();
            let site_specific_model_str = params.next().unwrap().trim();
            let site_specific_model = SiteSpecificRateModel::from_str(site_specific_model_str);
            RateModel::X(strength, site_specific_model)
        }
        _ => panic!("Unknown rate model"),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubstitutionModel {
    MutSel,
    MutSelApprox,
    RelaxPMSF,
}

#[derive(Debug, Clone, Copy)]
enum Verbosity {
    Quiet,
    Min,
    Med,
    Max,
    Debug,
}

impl Verbosity {
    fn from_u8(value: u8) -> Verbosity {
        match value {
            0 => Verbosity::Quiet,
            1 => Verbosity::Min,
            2 => Verbosity::Med,
            3 => Verbosity::Max,
            4 => Verbosity::Debug,
            _ => panic!("Invalid verbosity level"),
        }
    }

    fn should_print(&self, level: Verbosity) -> bool {
        match (self, level) {
            (Verbosity::Quiet, _) => false,
            (
                Verbosity::Min,
                Verbosity::Min | Verbosity::Med | Verbosity::Max | Verbosity::Debug,
            ) => true,
            (Verbosity::Med, Verbosity::Med | Verbosity::Max | Verbosity::Debug) => true,
            (Verbosity::Max, Verbosity::Max | Verbosity::Debug) => true,
            (Verbosity::Debug, _) => true,
            _ => false,
        }
    }
}

// optimize function called from the rust binary
pub fn optimize_rust_binary(
    newick: &Path,
    fasta: &Path,
    pi_reg: f64,
    R_reg: f64,
    rate_mode: &str,
    prior_R_file: Option<&Path>,
    prior_pi_file: Option<&Path>,
    substitution_model: SubstitutionModel,
) -> Result<(Tensor, Tensor, Tensor, Vec<f64>), candle_core::Error> {
    let sequences = io::read_alignment(fasta);

    let (felsenstein, distances) =
        io::process_newick_alignment(&std::fs::read_to_string(&newick).unwrap(), &sequences);

    let aa_dist = io::get_site_specific_aa_distribution(&sequences);

    let rate_model = parse_rate_model(rate_mode);

    optimization::optimize_internal(
        felsenstein,
        &distances,
        aa_dist,
        MutselParams {
            pi_reg,
            Mu_reg: R_reg,
            branch_reg: 10.0, // default value
        },
        rate_model,
        prior_R_file,
        prior_pi_file,
        substitution_model,
        Verbosity::Debug,
        "./mutsel"
    )
}
