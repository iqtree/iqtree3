#![allow(non_snake_case)]

use candle_core::Tensor;
use mutsel_rust::gamma::GammaOp;

fn run(args : &[String]) {
    let newick = args[0].clone();
    let fasta = args[1].clone();
    let pi_reg = args[2].parse::<f64>().unwrap();
    let R_reg = args[3].parse::<f64>().unwrap();
    let rate_model = args[4].clone();
    let mutsel = args[5].clone();


    let substitution_model = match mutsel.as_str() {
        "mutselapprox" => mutsel_rust::SubstitutionModel::MutSelApprox,
        "relaxpmsf" => mutsel_rust::SubstitutionModel::RelaxPMSF,
        "mutsel" => mutsel_rust::SubstitutionModel::MutSel,
        _ => panic!("Unknown substitution model"),
    };

    let (S, sqrt_pi, rate_para, _substitution_rates) = mutsel_rust::optimize_rust_binary(
        &std::path::Path::new(&newick),
        &std::path::Path::new(&fasta),
        pi_reg,
        R_reg,
        &rate_model,
        None,
        None,
        substitution_model
    )
    .unwrap();

    if substitution_model == mutsel_rust::SubstitutionModel::RelaxPMSF {
        let (R, pi) = mutsel_rust::model::phylograd2iqtree_parametrization(&S, &sqrt_pi).unwrap();
        mutsel_rust::io::write_sitefreq_file(&pi, &std::path::Path::new(args[6].as_str()));
        // R should be the same for all sites
        mutsel_rust::io::write_pam_file(&R.get(0).unwrap(), &std::path::Path::new(args[7].as_str()));
    } else {
        mutsel_rust::io::write_nexus_file(&S, &sqrt_pi, &rate_para, &std::path::Path::new(args[6].as_str()));
    }
}

fn phylo_grad_param(args : &[String]) {
    let model_file = args[0].clone();
    let parsed = mutsel_rust::io::read_binary_site_model_file(std::path::Path::new(&model_file))
        .unwrap_or_else(|e| panic!("Error reading site model file '{}': {e}", model_file));

    let rate_model =  parse_rate_model(&parsed.rate_model_string).unwrap();

    println!("Parsed rate model: {:?}\n", rate_model);

    let (weights, rates) = if let ParsedRateModel::Gamma { categories, alpha } = rate_model {
        (Tensor::full(1.0 / categories as f64, &[categories], &candle_core::Device::Cpu).unwrap(), Tensor::full(alpha.unwrap(), &[categories], &candle_core::Device::Cpu).unwrap().apply_op1(GammaOp::new(categories)).unwrap())
    } else if let ParsedRateModel::FreeRate { categories, weights, rates } = rate_model {
        (Tensor::from_vec(weights.unwrap(), &[categories], &candle_core::Device::Cpu).unwrap(), Tensor::from_vec(rates.unwrap(), &[categories], &candle_core::Device::Cpu).unwrap())
    } else if let ParsedRateModel::X = rate_model {
        (Tensor::full(1.0, &[1], &candle_core::Device::Cpu).unwrap(), Tensor::full(1.0, &[1], &candle_core::Device::Cpu).unwrap())
    } else {
        panic!("Unexpected rate model variant");
    };

    println!("Weights: {:?}", weights);
    println!("Rates: {:?}", rates);


    let (S, sqrt_pi) = mutsel_rust::model::iqtree2phylograd_parametrization(&parsed.rate_matrices, &parsed.site_freq).unwrap();

    Tensor::write_npz(&[("weights", weights), ("rates", rates), ("S", S), ("sqrt_pi", sqrt_pi)], args[1].as_str()).unwrap();
}

#[derive(Debug, Clone, PartialEq)]
enum ParsedRateModel {
    Gamma {
        categories: usize,
        alpha: Option<f64>,
    },
    FreeRate {
        categories: usize,
        weights: Option<Vec<f64>>,
        rates: Option<Vec<f64>>,
    },
    X
}

fn parse_rate_model(rate_model: &str) -> Result<ParsedRateModel, String> {
    let model = rate_model.trim();
    if model.is_empty() {
        return Err("Rate model string is empty".to_string());
    }

    if model.starts_with("R1") {
        return Ok(ParsedRateModel::X);
    }

    let first_brace = model.find('{');
    let (head, body) = match first_brace {
        Some(pos) => {
            if !model.ends_with('}') {
                return Err(format!("Missing closing '}}' in rate model '{model}'"));
            }
            (&model[..pos], Some(&model[pos + 1..model.len() - 1]))
        }
        None => (model, None),
    };

    let mut chars = head.chars();
    let model_type = chars
        .next()
        .ok_or_else(|| "Missing rate model type".to_string())?
        .to_ascii_uppercase();
    let categories_str = chars.as_str().trim();
    let categories = categories_str
        .parse::<usize>()
        .map_err(|_| format!("Invalid category count '{categories_str}' in '{model}'"))?;
    if categories == 0 {
        return Err("Category count must be greater than 0".to_string());
    }

    let params = if let Some(inner) = body {
        let values = inner
            .split(',')
            .map(|v| {
                let token = v.trim();
                token
                    .parse::<f64>()
                    .map_err(|_| format!("Invalid numeric value '{token}' in '{model}'"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Some(values)
    } else {
        None
    };

    match model_type {
        'G' => {
            if let Some(values) = params {
                if values.len() != 1 {
                    return Err(format!(
                        "Gamma model expects exactly 1 parameter (alpha), got {}",
                        values.len()
                    ));
                }
                Ok(ParsedRateModel::Gamma {
                    categories,
                    alpha: Some(values[0]),
                })
            } else {
                Ok(ParsedRateModel::Gamma {
                    categories,
                    alpha: None,
                })
            }
        }
        'R' => {
            if let Some(values) = params {
                if values.len() != 2 * categories {
                    return Err(format!(
                        "FreeRate model expects {} parameters ({} weights + {} rates), got {}",
                        2 * categories,
                        categories,
                        categories,
                        values.len()
                    ));
                }
                let weights = values[..categories].to_vec();
                let rates = values[categories..].to_vec();
                Ok(ParsedRateModel::FreeRate {
                    categories,
                    weights: Some(weights),
                    rates: Some(rates),
                })
            } else {
                Ok(ParsedRateModel::FreeRate {
                    categories,
                    weights: None,
                    rates: None,
                })
            }
        },
        'X' => Ok(ParsedRateModel::X),
        _ => Err(format!(
            "Unknown rate model type '{}', expected 'G' or 'R'",
            model_type
        )),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args[1] == "run" {
        run(&args[2..].to_vec());
    } else if args[1] == "phylograd-param" {
        phylo_grad_param(&args[2..].to_vec());
    } 
    else {
        panic!("Unknown command");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gamma_with_alpha() {
        let parsed = parse_rate_model("G4{0.456}").unwrap();
        assert_eq!(
            parsed,
            ParsedRateModel::Gamma {
                categories: 4,
                alpha: Some(0.456),
            }
        );
    }

    #[test]
    fn parses_freerate_with_weights_and_rates() {
        let parsed = parse_rate_model("R2{0.5,1.0, 0.5, 3.5}").unwrap();
        assert_eq!(
            parsed,
            ParsedRateModel::FreeRate {
                categories: 2,
                weights: Some(vec![0.5, 1.0]),
                rates: Some(vec![0.5, 3.5]),
            }
        );
    }
}
