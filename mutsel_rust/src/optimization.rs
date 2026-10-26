use std::{
    iter,
    path::Path,
    sync::{Arc, Mutex},
};

use candle_core::{Tensor, Var};
use candle_nn::{Optimizer, ops::softmax};
use phylo_grad::FelsensteinTree;

use crate::{
    RateModel, SiteSpecificRateModel, SubstitutionModel, Verbosity,
    felsenstein::{self, FelsensteinOp, FelsensteinWithEdgeOp},
    gamma::GammaOp,
    model,
    utils::{histogram, tensor_full},
};

trait Optimizable {
    fn variables(&self) -> Vec<Var>;
    fn likelihood(&self) -> Tensor;
    fn penalty(&self) -> Tensor;
    fn print_state(&self) {}
}

pub enum RateParameters {
    G(usize, Var),                      // alpha for gamma distribution
    R(Var, Var),                        // log_rate_cat and log_rate_weights for free rate model
    X(Var, f64, SiteSpecificRateModel), // site-specific rates paramter, strength of penalty, and the type of penalty
}

impl RateParameters {
    fn gamma(n: usize, alpha: f64) -> RateParameters {
        // Alpha is unconstrained, but there is a very strong barrier approaching 0. So this is not a problem.
        RateParameters::G(n, Var::from_tensor(&tensor_full(alpha, &[])).unwrap())
    }

    fn init(num_sites: usize, rate_model: RateModel, alpha: f64) -> RateParameters {
        match rate_model {
            RateModel::G(num_cat) => RateParameters::gamma(num_cat, alpha),
            RateModel::R(num_cat) => {
                let gamma_op = GammaOp::new(num_cat);
                let alpha = Var::from_tensor(&tensor_full(alpha, &[])).unwrap();
                let log_rate_cat =
                    Var::from_tensor(&alpha.apply_op1(gamma_op).unwrap().log().unwrap()).unwrap();
                let log_rate_weights = Var::from_tensor(&tensor_full(0.0, &[num_cat])).unwrap();
                RateParameters::R(log_rate_cat, log_rate_weights)
            }
            RateModel::X(strength, site_specific_rate_model) => RateParameters::X(
                Var::from_tensor(&site_specific_rate_model.init(num_sites)).unwrap(),
                strength,
                site_specific_rate_model,
            ),
        }
    }

    fn parameters_for_iqtree(&self) -> Tensor {
        match self {
            RateParameters::G(_, alpha) => alpha.as_detached_tensor().unsqueeze(0).unwrap(),
            RateParameters::R(log_rate_cat, log_rate_weights) => {
                let (categories, log_weights) = normalize_rate_cat(
                    &log_rate_cat.as_detached_tensor(),
                    &log_rate_weights.as_detached_tensor(),
                );
                Tensor::cat(&[log_weights.exp().unwrap(), categories], 0).unwrap()
            }
            RateParameters::X(_, _, _) => {
                // Dummy value, want to make sure this allocates, so there is a valid pointer to return
                tensor_full(0.0, &[1])
            }
        }
    }

    fn variables(&self) -> Vec<Var> {
        match self {
            RateParameters::G(_, alpha) => vec![alpha.clone()],
            RateParameters::R(log_rate_cat, log_rate_weights) => {
                vec![log_rate_cat.clone(), log_rate_weights.clone()]
            }
            RateParameters::X(site_rate_var, _, _) => vec![site_rate_var.clone()],
        }
    }

    fn rates(&self) -> (Tensor, Tensor) {
        match self {
            RateParameters::G(num_cat, alpha) => {
                let gamma_op = GammaOp::new(*num_cat);
                let categories = alpha.apply_op1(gamma_op.clone()).unwrap();
                let log_weights = tensor_full(f64::ln(1.0 / *num_cat as f64), &[*num_cat]);
                (categories, log_weights)
            }
            RateParameters::R(log_rate_cat, log_rate_weights) => {
                normalize_rate_cat(log_rate_cat, log_rate_weights)
            }
            RateParameters::X(_, _, _) => {
                panic!(
                    "Rates for site-specific model should be calculated separately using the site-specific model and parameters."
                )
            }
        }
    }

    fn site_specific_rates(&self) -> Tensor {
        match self {
            RateParameters::X(site_rate_var, _, site_model) => {
                site_model.rates_from_parameters(site_rate_var)
            }
            _ => {
                panic!("Site-specific rates should only be calculated for the site-specific model.")
            }
        }
    }

    fn penalty(&self) -> Tensor {
        match self {
            RateParameters::X(_, strength, site_model) => {
                (site_model.penalty(&self.site_specific_rates()) * *strength).unwrap()
            }
            _ => tensor_full(0.0, &[]),
        }
    }

    fn calc_rate_matrix(
        &self,
        R: &Tensor,
        log_pi: &Tensor,
        global_scaling: &Tensor,
        substitution_model: SubstitutionModel,
    ) -> (Tensor, Tensor) {
        let (S, sqrt_pi) = model::calc_rate_matrix(R, log_pi, global_scaling, substitution_model);
        let S = if let RateParameters::X(_, _, _) = self {
            S.broadcast_mul(
                &self
                    .site_specific_rates()
                    .unsqueeze(1)
                    .unwrap()
                    .unsqueeze(2)
                    .unwrap(),
            )
            .unwrap()
        } else {
            S
        };
        (S, sqrt_pi)
    }

    fn likelihood(
        &self,
        felsenstein_op: FelsensteinWithEdgeOp,
        R: &Tensor,
        log_pi: &Tensor,
        substitution_model: SubstitutionModel,
        calc_gradients: bool,
        log_branch_lengths: &Tensor,
    ) -> Tensor {
        self.likelihood_per_site(
            felsenstein_op,
            R,
            log_pi,
            substitution_model,
            calc_gradients,
            log_branch_lengths,
        )
        .sum_all()
        .unwrap()
    }

    fn likelihood_per_site(
        &self,
        felsenstein_op: FelsensteinWithEdgeOp,
        R: &Tensor,
        log_pi: &Tensor,
        substitution_model: SubstitutionModel,
        calc_gradients: bool,
        log_branch_lengths: &Tensor,
    ) -> Tensor {
        let (S, sqrt_pi) =
            self.calc_rate_matrix(R, log_pi, &tensor_full(1.0, &[]), substitution_model);
        let branch_lengths = log_branch_lengths.exp().unwrap();
        let likelihood = match self {
            RateParameters::G(_, _) | RateParameters::R(_, _) => {
                let mut all_likelihoods = vec![];
                let (rate_categories, log_weights) = self.rates();
                for r_cat_index in 0..rate_categories.dims()[0] {
                    let rate_tensor = rate_categories.get(r_cat_index).unwrap();
                    let S = S
                        .broadcast_mul(&rate_tensor.reshape(&[1, 1, 1]).unwrap())
                        .unwrap();
                    let likelihoods = if calc_gradients {
                        S.apply_op3(&sqrt_pi, &branch_lengths, felsenstein_op.clone())
                            .unwrap()
                    } else {
                        S.apply_op3(&sqrt_pi, &branch_lengths, felsenstein_op.into_fwd_op())
                            .unwrap()
                    };
                    let weighted_likelihoods = likelihoods
                        .broadcast_add(&log_weights.get(r_cat_index).unwrap())
                        .unwrap();
                    all_likelihoods.push(weighted_likelihoods);
                }
                let all_likelihoods = Tensor::stack(&all_likelihoods, 0).unwrap();
                let final_likelihoods = all_likelihoods.log_sum_exp(0).unwrap();
                final_likelihoods
            }
            RateParameters::X(_, _, _) => {
                if calc_gradients {
                    S.apply_op3(&sqrt_pi, &branch_lengths, felsenstein_op.clone())
                        .unwrap()
                } else {
                    S.apply_op3(&sqrt_pi, &branch_lengths, felsenstein_op.into_fwd_op())
                        .unwrap()
                }
            }
        };

        likelihood
    }
}

pub struct AlphaAndScalingParameters {
    pub felsenstein_op: FelsensteinWithEdgeOp,
    pub log_global_scaling: Var,
    pub log_branch_length_scaling: Var,
    pub init_log_branch_lengths: Tensor,
    pub R: Tensor,
    pub log_pi: Tensor,
    pub rate_model: RateParameters,
    pub branch_length_penalty: f64,
}

impl Optimizable for AlphaAndScalingParameters {
    fn variables(&self) -> Vec<Var> {
        let mut vars = vec![
            self.log_global_scaling.clone(),
            self.log_branch_length_scaling.clone(),
        ];
        vars.extend(self.rate_model.variables());
        vars
    }

    fn likelihood(&self) -> Tensor {
        let log_branch_length_scaling = self
            .log_branch_length_scaling
            .broadcast_sub(&self.log_branch_length_scaling.mean(0).unwrap())
            .unwrap();

        let log_branch_lengths = (&self.init_log_branch_lengths + log_branch_length_scaling)
            .unwrap()
            .broadcast_add(&self.log_global_scaling)
            .unwrap();

        self.rate_model.likelihood(
            self.felsenstein_op.clone(),
            &self.R,
            &self.log_pi,
            SubstitutionModel::MutSel,
            true,
            &log_branch_lengths,
        )
    }

    fn penalty(&self) -> Tensor {
        let branch_penalty = self
            .log_branch_length_scaling
            .powf(2.0)
            .unwrap()
            .sum_all()
            .unwrap()
            * self.branch_length_penalty;
        (self.rate_model.penalty() + branch_penalty.unwrap()).unwrap()
    }

    fn print_state(&self) {
        println!(
            "global_scaling: {}",
            self.log_global_scaling
                .exp()
                .unwrap()
                .to_scalar::<f64>()
                .unwrap()
        );
        match &self.rate_model {
            RateParameters::G(num_cat, alpha) if *num_cat > 1 => {
                println!("alpha: {}", alpha.to_scalar::<f64>().unwrap())
            }
            _ => {}
        }
    }
}

pub struct ModelParameters {
    pub felsenstein_op: FelsensteinWithEdgeOp,
    pub log_R: Var,
    pub log_pi: Var,
    pub log_global_scaling: Var,
    pub log_branch_length_scaling: Var,
    pub init_log_branch_lengths: Tensor,
    pub rate_parameters: RateParameters,
    pub pi_reg: f64,
    pub R_reg: f64,
    pub branch_length_penalty: f64,
    pub init_log_R: Tensor,
    pub init_log_pi: Tensor,
    pub substitution_model: SubstitutionModel,
}

impl ModelParameters {
    pub fn calc_rate_matrix(&self) -> (Tensor, Tensor) {
        self.rate_parameters.calc_rate_matrix(
            &Mu(&self.log_R.as_detached_tensor()),
            &self.log_pi.as_detached_tensor(),
            &self.log_global_scaling.as_detached_tensor().exp().unwrap(),
            self.substitution_model,
        )
    }

    pub fn save_npz(&self, path: &Path) {
        let Mu = Mu(&self.log_R.as_detached_tensor());
        let pi = softmax(&self.log_pi.as_detached_tensor(), 1).unwrap();

        Tensor::write_npz(
            &[
                ("Mu", &Mu),
                ("pi", &pi),
                (
                    "global_scaling",
                    &self.log_global_scaling.as_detached_tensor(),
                ),
                ("init_log_pi", &self.init_log_pi),
                ("init_log_R", &self.init_log_R),
                (
                    "log_branch_length_scaling",
                    &self.log_branch_length_scaling.as_detached_tensor(),
                ),
            ],
            path,
        )
        .unwrap();
    }
}

impl Optimizable for ModelParameters {
    fn variables(&self) -> Vec<Var> {
        let mut vars = vec![
            self.log_R.clone(),
            self.log_pi.clone(),
            self.log_global_scaling.clone(),
            self.log_branch_length_scaling.clone(),
        ];
        vars.extend(self.rate_parameters.variables());
        vars
    }

    fn likelihood(&self) -> Tensor {
        let Mu = Mu(&self.log_R);

        let log_branch_length_scaling = self
            .log_branch_length_scaling
            .broadcast_sub(&self.log_branch_length_scaling.mean(0).unwrap())
            .unwrap();

        let log_branch_lengths = (&self.init_log_branch_lengths + log_branch_length_scaling)
            .unwrap()
            .broadcast_add(&self.log_global_scaling)
            .unwrap();

        self.rate_parameters.likelihood(
            self.felsenstein_op.clone(),
            &Mu,
            &self.log_pi,
            self.substitution_model,
            true,
            &log_branch_lengths,
        )
    }

    fn penalty(&self) -> Tensor {
        let pi_penalty = self
            .init_log_pi
            .sub(&self.log_pi)
            .unwrap()
            .powf(2.0)
            .unwrap()
            .sum_all()
            .unwrap();
        let pi_penalty = (pi_penalty * self.pi_reg).unwrap();

        fn log_Mu(log_R: &Tensor) -> Tensor {
            let Mu = Mu(log_R);

            // One out the diagonal, since we only want to penalize the off-diagonal elements
            let Mu = (&Mu
                - (&Mu - 1.0).unwrap()
                    * Tensor::eye(20, candle_core::DType::F64, &candle_core::Device::Cpu).unwrap())
            .unwrap();

            return Mu.log().unwrap();
        }

        let Mu = log_Mu(&self.log_R)
            .sub(&log_Mu(&self.init_log_R))
            .unwrap()
            .powf(2.0)
            .unwrap()
            .sum_all()
            .unwrap();
        let R_penalty = (Mu * self.R_reg).unwrap();

        let branch_penalty = self
            .log_branch_length_scaling
            .powf(2.0)
            .unwrap()
            .sum_all()
            .unwrap()
            * self.branch_length_penalty;

        (pi_penalty + R_penalty + branch_penalty.unwrap() + self.rate_parameters.penalty()).unwrap()
    }

    fn print_state(&self) {
        println!(
            "global_scaling: {}",
            self.log_global_scaling
                .exp()
                .unwrap()
                .to_scalar::<f64>()
                .unwrap()
        );
        match &self.rate_parameters {
            RateParameters::G(num_cat, alpha) if *num_cat > 1 => {
                println!("alpha: {}", alpha.to_scalar::<f64>().unwrap())
            }
            RateParameters::G(_, _) => {}
            RateParameters::R(log_rate_cat, log_rate_weights) => {
                let (categories, log_weights) = normalize_rate_cat(
                    &log_rate_cat.as_detached_tensor(),
                    &log_rate_weights.as_detached_tensor(),
                );
                let weights = log_weights.exp().unwrap();
                println!("Rate categories: {}", categories);
                println!("Rate category weights: {}", weights);
            }
            RateParameters::X(site_rate_var, _, site_model) => {
                let rates = site_model.rates_from_parameters(site_rate_var);
                let histogram = histogram(&rates, 10);
                println!("Site rate bounds: {}", histogram.0);
                println!("Site rate counts: {}", histogram.1);
            }
        }
    }
}

fn optimize(
    model: &impl Optimizable,
    min_iterations: usize,
    max_iterations: usize,
    min_rel_improvement: f64,
    no_improve_patience: usize,
    verbosity: Verbosity,
) {
    let variables = model.variables();
    let mut opt = candle_nn::optim::AdamW::new_lr(variables, 0.05).unwrap();
    let parameter = candle_nn::optim::ParamsAdamW {
        lr: 0.05,
        weight_decay: 0.0,
        ..Default::default()
    };
    opt.set_params(parameter);

    let mut prev_opt = f64::INFINITY;
    let mut no_improve_count = 0;

    for iteration in 0.. {
        let neg_likelihood = model.likelihood().neg().unwrap();
        let penalty = model.penalty();
        let opt_fn = (&neg_likelihood + &penalty).unwrap();

        let current_opt = opt_fn.to_scalar::<f64>().unwrap();
        if verbosity.should_print(Verbosity::Med) {
            println!(
                "Iteration {}: neg Loglikelihood {:.3}, Optfn {:.3}",
                iteration,
                neg_likelihood.to_scalar::<f64>().unwrap(),
                current_opt
            );
        }

        model.print_state();
        let grads = opt_fn.backward().unwrap();
        opt.step(&grads).unwrap();

        if iteration > min_iterations {
            let rel_improvement = (prev_opt - current_opt) / prev_opt.abs().max(1e-12);
            if rel_improvement > min_rel_improvement {
                no_improve_count = 0;
            } else {
                no_improve_count += 1;
            }
            if no_improve_count >= no_improve_patience {
                println!(
                    "Stopping at iteration {} (relative loss improvement {:.2e} <= {:.2e} for {} consecutive iterations)",
                    iteration, rel_improvement, min_rel_improvement, no_improve_patience
                );
                break;
            }
        }

        if iteration > max_iterations {
            println!("Reached maximum iterations ({})", max_iterations);
            break;
        }

        prev_opt = current_opt;
    }
}

pub fn optimize_global_scaling_alpha(
    felsenstein_op: FelsensteinWithEdgeOp,
    log_pi: &Tensor,
    R: &Tensor,
    log_branch_lengths: &Tensor,
    branch_length_penalty: f64,
    verbosity: Verbosity,
) -> (Tensor, Tensor, Tensor) {
    let var_log_global_scaling = Var::from_tensor(&tensor_full(0.0, &[])).unwrap();
    let var_alpha = Var::from_tensor(&tensor_full(1.0, &[])).unwrap();
    let var_log_branch_length_scaling =
        Var::from_tensor(&tensor_full(0.0, log_branch_lengths.dims())).unwrap();

    let model = AlphaAndScalingParameters {
        felsenstein_op,
        log_global_scaling: var_log_global_scaling,
        R: R.detach(),
        log_pi: log_pi.detach(),
        init_log_branch_lengths: log_branch_lengths.detach(),
        log_branch_length_scaling: var_log_branch_length_scaling,
        rate_model: RateParameters::G(1, var_alpha),
        branch_length_penalty,
    };

    optimize(&model, 10, 100, 1e-5, 5, verbosity);

    let alpha = if let RateParameters::G(_, alpha) = model.rate_model {
        alpha
    } else {
        panic!("Expected gamma rate model.");
    };

    return (
        model.log_global_scaling.as_detached_tensor(),
        alpha.as_detached_tensor(),
        model.log_branch_length_scaling.as_detached_tensor(),
    );
}

pub fn two_step_light_pmsf(
    felsenstein_op: FelsensteinOp,
    categories: &[[f64; 20]],
    weights: &[f64],
    f_class: &[f64; 20],
    log_branch_lengths: &Tensor,
    mutsel_params: &super::MutselParams,
    verbosity: Verbosity,
) -> (Tensor, f64, f64, Tensor) {
    let step1_site_freq = light_pmsf(
        felsenstein_op.into_with_edge_op(),
        categories,
        weights,
        1.0,
        log_branch_lengths,
        f_class,
    );

    let Mu = loadMu();

    let log_pi = step1_site_freq.log().unwrap();

    let (step2_log_global_scaling, step2_alpha, step2_log_branch_length_scaling) =
        optimize_global_scaling_alpha(
            felsenstein_op.into_with_edge_op(),
            &log_pi,
            &Mu,
            log_branch_lengths,
            mutsel_params.branch_reg,
            verbosity,
        );

    let final_site_freq = light_pmsf(
        felsenstein_op.into_with_edge_op(),
        categories,
        weights,
        step2_alpha.to_scalar::<f64>().unwrap(),
        &(log_branch_lengths + &step2_log_branch_length_scaling)
            .unwrap()
            .broadcast_add(&step2_log_global_scaling)
            .unwrap(),
        f_class,
    );

    (
        final_site_freq,
        step2_log_global_scaling.to_scalar::<f64>().unwrap(),
        step2_alpha.to_scalar::<f64>().unwrap(),
        step2_log_branch_length_scaling,
    )
}

pub fn light_pmsf(
    felsenstein_op: FelsensteinWithEdgeOp,
    categories: &[[f64; 20]],
    weights: &[f64],
    alpha: f64,
    log_branch_lengths: &Tensor,
    f_class: &[f64; 20],
) -> Tensor {
    let mut likelihoods = vec![];

    let Mu = loadMu();

    let rate_model = RateParameters::gamma(1, alpha);

    for category in iter::once(f_class).chain(categories.iter()) {
        let category_tensor =
            Tensor::from_vec(category.to_vec(), &[20], &candle_core::Device::Cpu).unwrap();
        let log_pi = category_tensor.log().unwrap().unsqueeze(0).unwrap();

        likelihoods.push(rate_model.likelihood_per_site(
            felsenstein_op.clone(),
            &Mu,
            &log_pi,
            SubstitutionModel::MutSel,
            false,
            log_branch_lengths,
        ));
    }

    let likelihoods = Tensor::stack(&likelihoods, 0).unwrap();
    let weights_tensor =
        Tensor::from_slice(&weights, &[weights.len()], &candle_core::Device::Cpu).unwrap();
    let log_weights_tensor = weights_tensor.log().unwrap().unsqueeze(1).unwrap();

    let weighted_likelihoods = (likelihoods.broadcast_add(&log_weights_tensor)).unwrap();

    let posteriors = candle_nn::ops::softmax(&weighted_likelihoods, 0).unwrap();

    let category_tensor = Tensor::from_vec(
        iter::once(f_class)
            .chain(categories.iter())
            .flatten()
            .copied()
            .collect(),
        &[categories.len() + 1, 20],
        &candle_core::Device::Cpu,
    )
    .unwrap();

    let site_freq = posteriors.t().unwrap().matmul(&category_tensor).unwrap();

    site_freq
}

/// Calculates the mutation rate matrix from the log_R variable. This is a lower triangular matrix with the exchanabilities and the equilibirum in the diagonal. Both as in log space.
/// Returns Mut and the equilibrium frequencies in the diagonal. (not log space)
pub fn Mu(log_parameter: &Tensor) -> Tensor {
    let parameter = log_parameter.exp().unwrap();
    let diagonal = (&parameter
        * Tensor::eye(20, candle_core::DType::F64, &candle_core::Device::Cpu).unwrap())
    .unwrap();
    let off_diagonal = (parameter - &diagonal).unwrap();

    let pi = diagonal
        .broadcast_div(&diagonal.sum_all().unwrap())
        .unwrap();

    let R = (&off_diagonal + off_diagonal.t().unwrap()).unwrap();

    let Q = R.matmul(&pi).unwrap();

    let Q = (&Q
        - &Q * Tensor::eye(20, candle_core::DType::F64, &candle_core::Device::Cpu).unwrap())
    .unwrap();

    let row_sum = Q.sum(1).unwrap();
    let pi_vec = pi.sum(1).unwrap();
    let sum = row_sum.dot(&pi_vec).unwrap();
    let Q = Q.broadcast_div(&sum).unwrap();

    // This has one expected mutation per unit time. The guide tree will be more in the regime of 1 substitution.
    // So we mutlpiphy with 20. Exact factor is fitted later. We also divide with 8 which is emprically close to the optimum.
    // (This does depend on the protein and how many columns are under strong selection)

    let Q = (Q * (20.0 / 12.0)).unwrap();

    // diagonal is zero here, but they are not used anyway
    (Q + pi).unwrap()
}

// Used with the MutSel model.
fn loadMu() -> Tensor {
    let R_lower = crate::data::load_lower_R_with_equi(crate::data::CODON2_TXT);
    let log_R = R_lower.log().unwrap();
    Mu(&log_R)
}

/// Returns the optimal S, sqrt_pi and rate parameters.
/// S: [L, 20, 20], sqrt_pi: [L, 20]
/// rate parameters is either alpha shape: [1] or [2 * num_cat] for free rate model (first num_cat weights, second num_cat are rates)
pub fn optimize_internal(
    felsenstein: FelsensteinTree<20>,
    distances: &[f64],
    aa_dist: Tensor,
    mutsel_params: super::MutselParams,
    rate_model: RateModel,
    prior_R_file: Option<&Path>,
    prior_pi_file: Option<&Path>,
    substitution_model: SubstitutionModel,
    verbosity: Verbosity,
    out_prefix: &str,
) -> Result<(Tensor, Tensor, Tensor, Vec<f64>), candle_core::Error> {
    let op = felsenstein::FelsensteinOp::new(Arc::new(Mutex::new(felsenstein)));

    // 0.5 psuedo counts
    let aa_dist = (aa_dist + 0.5).unwrap();

    // Lower triangular matrix with zeros everywhere else
    let init_R = if let Some(prior_R_file) = prior_R_file {
        let file_content = std::fs::read_to_string(prior_R_file)?;
        crate::data::load_lower_R_with_equi(&file_content)
    } else {
        crate::data::load_lower_R_with_equi(crate::data::CODON2_TXT)
    };

    let log_branch_lengths =
        Tensor::from_slice(distances, &[distances.len()], &candle_core::Device::Cpu)?.log()?;

    let mut var_log_global_scaling = 0.0;
    let mut var_alpha = 1.0;
    let var_log_branch_length_scaling =
        Var::from_tensor(&tensor_full(0.0, log_branch_lengths.dims())).unwrap();

    let site_freq = if let Some(prior_pi_file) = prior_pi_file {
        println!(
            "Using prior site frequencies from file: {:?}",
            prior_pi_file
        );
        crate::io::read_sitefreq_file(prior_pi_file)
    } else {
        // Do our lightweight PMSF procedure for the site_freq prior:
        let f_class_weight = 0.1;

        let weights = iter::once(f_class_weight)
            .chain(
                crate::data::UDM256_WEIGHTS
                    .iter()
                    .map(|w| w * (1.0 - f_class_weight)),
            )
            .collect::<Vec<f64>>();

        let f_class = aa_dist
            .sum(0)?
            .broadcast_div(&aa_dist.sum_all()?.unsqueeze(0)?)?;
        let f_class: [f64; 20] = f_class.to_vec1()?.try_into().unwrap();

        let (site_freq, global_scaling, alpha, log_branch_length_scaling) = two_step_light_pmsf(
            op.clone(),
            crate::data::UDM256,
            &weights,
            &f_class,
            &log_branch_lengths,
            &mutsel_params,
            verbosity
        );
        var_log_global_scaling = global_scaling;
        var_alpha = alpha;
        var_log_branch_length_scaling
            .set(&log_branch_length_scaling)
            .unwrap();

        site_freq
    };

    // Variable which gets optimized
    let log_R = Var::from_tensor(&init_R.log()?)?;
    let init_log_pi = site_freq.log()?;
    let log_pi = Var::from_tensor(&init_log_pi)?;
    let num_sites = site_freq.dims()[0];

    let init_log_R = log_R.detach().copy().unwrap();
    let init_log_pi = log_pi.detach().copy().unwrap();

    let model = ModelParameters {
        felsenstein_op: op.into_with_edge_op(),
        log_R,
        log_pi,
        log_global_scaling: Var::from_tensor(&tensor_full(var_log_global_scaling, &[]))?,
        log_branch_length_scaling: var_log_branch_length_scaling,
        init_log_branch_lengths: log_branch_lengths.detach(),
        rate_parameters: RateParameters::init(num_sites, rate_model, var_alpha),
        pi_reg: mutsel_params.pi_reg,
        R_reg: mutsel_params.Mu_reg,
        branch_length_penalty: mutsel_params.branch_reg,
        init_log_R,
        init_log_pi,
        substitution_model,
    };

    optimize(&model, 100, 500, 1e-5, 5, verbosity);

    let (S, sqrt_pi) = model.calc_rate_matrix();
    let rate_para = model.rate_parameters.parameters_for_iqtree();

    let substitution_rates = model::substitution_rates(&S, &sqrt_pi);

    let average_rate = substitution_rates.iter().sum::<f64>() / substitution_rates.len() as f64;
    if verbosity.should_print(Verbosity::Min) {
        println!("Average substitution rate: {:.3}", average_rate);
        model.save_npz(Path::new(&format!("{}.mutsel.npz", out_prefix)));
    }

    let S = S.broadcast_div(&tensor_full(average_rate, &[])).unwrap();

    Ok((S, sqrt_pi, rate_para, substitution_rates))
}

// Outputs categories and log_weights
fn normalize_rate_cat(log_rate_cat: &Tensor, log_rate_weights: &Tensor) -> (Tensor, Tensor) {
    let log_weights = candle_nn::ops::log_softmax(&log_rate_weights, 0).unwrap();
    let log_sum = (log_weights.add(&log_rate_cat))
        .unwrap()
        .log_sum_exp(0)
        .unwrap();
    let categories = (log_rate_cat.broadcast_sub(&log_sum))
        .unwrap()
        .exp()
        .unwrap();
    (categories, log_weights)
}
