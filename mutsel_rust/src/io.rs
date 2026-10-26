use candle_core::Tensor;
use lazy_static::lazy_static;
use seq_io::fasta::Record;
use std::{collections::HashMap, io::Read, path::Path, vec};

use anyhow::{Context, Result};

lazy_static! {
    static ref AMINO_MAPPING: HashMap<u8, u8> = {
        let mut m = HashMap::new();
        m.insert(b'A', 0);
        m.insert(b'R', 1);
        m.insert(b'N', 2);
        m.insert(b'D', 3);
        m.insert(b'C', 4);
        m.insert(b'Q', 5);
        m.insert(b'E', 6);
        m.insert(b'G', 7);
        m.insert(b'H', 8);
        m.insert(b'I', 9);
        m.insert(b'L', 10);
        m.insert(b'K', 11);
        m.insert(b'M', 12);
        m.insert(b'F', 13);
        m.insert(b'P', 14);
        m.insert(b'S', 15);
        m.insert(b'T', 16);
        m.insert(b'W', 17);
        m.insert(b'Y', 18);
        m.insert(b'V', 19);
        m.insert(b'-', 20);
        m.insert(b'X', 20);
        m.insert(b'?', 20);
        m.insert(b'.', 20);
        m
    };
}

fn normalize_residue(residue: u8) -> u8 {
    residue.to_ascii_uppercase()
}

fn seq2pll(
    seq: impl Iterator<Item = u8>,
) -> Vec<phylo_grad::nalgebra::SVector<f64, 20>> {
    seq.map(|c| *AMINO_MAPPING.get(&normalize_residue(c)).unwrap_or(&20))
        .map(|idx| {
            let mut v = phylo_grad::nalgebra::SVector::<f64, 20>::zeros();
                
            if idx < 20 {
                v[idx as usize] = 1.0;
            } else {
                for i in 0..20 {
                    v[i] = 1.0;
                }
            }
            v
        })
        .collect()
}

pub fn read_alignment(fasta_file: &Path) -> HashMap<String, Vec<u8>> {
    let mut alignment = seq_io::fasta::Reader::new(
        std::fs::File::open(fasta_file).expect("Could not open fasta file"),
    );
    let mut sequences = HashMap::new();
    while let Some(result) = alignment.next() {
        let record = result.expect("Error reading fasta record");
        let seq: Vec<u8> = record.seq().iter().copied().collect();
        sequences.insert(record.id().unwrap().to_string(), seq);
    }
    sequences
}

pub fn get_site_specific_aa_distribution(sequences: &HashMap<String, Vec<u8>>) -> Tensor {
    let num_sites = sequences.values().next().unwrap().len();

    let mut data = vec![0f64; num_sites * 20];

    for seq in sequences.values() {
        for (site_idx, &residue) in seq.iter().enumerate() {
            let aa_idx = *AMINO_MAPPING.get(&normalize_residue(residue)).unwrap_or(&20);
            if aa_idx == 20 {
                continue;
            }
            data[site_idx * 20 + aa_idx as usize] += 1.0;
        }
    }

    Tensor::from_vec(data, &[num_sites, 20], &candle_core::Device::Cpu).unwrap()
}

pub fn get_site_specific_aa_distribution_iqtree(alignment: &[u8], L: usize, N: usize) -> Tensor {
    let mut data = vec![0f64; L * 20];

    for site_idx in 0..L {
        for seq_idx in 0..N {
            let residue = alignment[site_idx * N + seq_idx];
            if residue == 20 {
                continue;
            }
            data[site_idx * 20 + residue as usize] += 1.0;
        }
    }

    Tensor::from_vec(data, &[L, 20], &candle_core::Device::Cpu).unwrap()
}

pub fn create_felsenstein_tree(
    parents: &[i32],
    distances: &[f64],
    alignment: &[u8],
    L : usize,
    N : usize,
) -> phylo_grad::FelsensteinTree<20> {
    let mut felsenstein = phylo_grad::FelsensteinTree::<20>::new(parents, distances);

    let mut sites = vec![];
    sites.resize(L, vec![]);
    for site_idx in 0..L {
        sites[site_idx].resize(N, phylo_grad::nalgebra::SVector::<f64, 20>::zeros());
        for seq_idx in 0..N {
            let residue = alignment[site_idx * N + seq_idx];
            if residue < 20 {
                sites[site_idx][seq_idx][residue as usize] = 1.0;
            } else {
                for i in 0..20 {
                    sites[site_idx][seq_idx][i] = 1.0;
                }
            }
        }
    }
    felsenstein.bind_leaf_pl(sites);
    felsenstein
}

pub fn process_newick_alignment(
    newick: &str,
    sequences: &HashMap<String, Vec<u8>>,
) -> (phylo_grad::FelsensteinTree<20>, Vec<f64>) {
    let tree = phylotree::tree::Tree::from_newick(newick).unwrap();

    let num_sides = sequences.values().next().unwrap().len();
    let level_order = tree.levelorder(&tree.get_root().unwrap()).unwrap();

    let mut parents = vec![i32::MAX; level_order.len()];
    let mut distances = vec![f64::NAN; level_order.len()];

    let num_leaves = tree.n_leaves();

    // idx_mapping[tree_idx] = new_idx
    let mut idx_mapping = vec![i32::MAX; level_order.len()];

    let mut leaf_idx = 0usize;
    let mut internal_idx = level_order.len() - 1;

    for node_idx in &level_order {
        let node = tree.get(node_idx).unwrap();
        if node.is_tip() {
            idx_mapping[*node_idx] = leaf_idx as i32;
            parents[leaf_idx] = idx_mapping[node.parent.unwrap()];
            distances[leaf_idx] = node.parent_edge.unwrap();
            leaf_idx += 1;
        } else {
            idx_mapping[*node_idx] = internal_idx as i32;
            if let Some(parent_idx) = node.parent {
                parents[internal_idx] = idx_mapping[parent_idx];
                distances[internal_idx] = node.parent_edge.unwrap();
            } else {
                parents[internal_idx] = -1;
            }
            internal_idx -= 1;
        }
    }

    let mut leaf_pll = vec![];

    for i in 0..num_sides {
        let mut column_seq = vec![b'x'; num_leaves];

        for node_idx in level_order.iter() {
            let node = tree.get(&node_idx).unwrap();
            if node.is_tip() {
                let new_idx = idx_mapping[*node_idx];
                assert!(new_idx < num_leaves as i32);
                let seq = sequences.get(node.name.as_ref().unwrap()).unwrap();
                column_seq[new_idx as usize] = seq[i];
            }
        }
        leaf_pll.push(seq2pll(column_seq.into_iter()));
    }

    let mut felsenstein = phylo_grad::FelsensteinTree::<20>::new(&parents, &distances);
    felsenstein.bind_leaf_pl(leaf_pll);
    (felsenstein, distances)
}

pub fn write_sitefreq_file(pi: &Tensor, path: &Path) {
    use std::io::Write;
    let pi_data: Vec<Vec<f64>> = pi.to_vec2().unwrap();
    let mut file = std::fs::File::create(path).expect("Could not create sitefreq file");
    for (idx, chunk) in pi_data.iter().enumerate() {
        let line = chunk
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<String>>()
            .join("\t");
        writeln!(file, "{}\t{}", idx + 1, line).expect("Could not write to sitefreq file");
    }
}

pub fn write_pam_file(R: &Tensor, path: &Path) {
    use std::io::Write;
    let R_data: Vec<Vec<f64>> = R.to_vec2().unwrap();
    let mut file = std::fs::File::create(path).expect("Could not create pam file");
    for i in 1..20 {
        for j in 0..i {
            write!(file, "{}\t", R_data[i][j]).expect("Could not write to pam file");
        }
        write!(file, "\n").expect("Could not write to pam file");
    }
    let equi: Vec<&str> = vec!["0.05"; 20];
    writeln!(file, "{}", equi.join("\t")).expect("Could not write to pam file");
}

/// Ignores the first column (site index)
pub fn read_sitefreq_file(path: &Path) -> Tensor {
    use std::io::BufRead;
    let file = std::fs::File::open(path).expect("Could not open sitefreq file");
    let reader = std::io::BufReader::new(file);
    let mut data = vec![];
    for line in reader.lines() {
        let line = line.expect("Could not read line from sitefreq file");
        line.split_whitespace().skip(1).for_each(|v| {
            data.push(
                v.parse::<f64>()
                    .expect("Could not parse float from sitefreq file"),
            )
        });
    }
    let L = data.len() / 20;
    return Tensor::from_vec(data, &[L, 20], &candle_core::Device::Cpu).unwrap();
}

pub fn read_pam_file(path: &Path) -> Tensor {
    let file_content = std::fs::read_to_string(path).expect("Could not read pam file");
    return crate::data::load_lower_R(&file_content);
}

pub fn write_nexus_file(S: &Tensor, sqrt_pi: &Tensor, alpha : &Tensor, path: &Path) {
    use std::io::Write;
    let mut file = std::fs::File::create(path).expect("Could not create nexus file");

    file.write(b"#nexus\n")
        .expect("Could not write to nexus file");
    file.write(b"begin sets;\n")
        .expect("Could not write to nexus file");

    let (R, pi) = crate::model::phylograd2iqtree_parametrization(S, sqrt_pi).unwrap();

    let L = R.dims()[0];

    for i in 0..L {
        file.write(format!("    charset p{} = AA, {}-{};\n", i + 1, i + 1, i + 1).as_bytes())
            .expect("Could not write to nexus file");
    }

    file.write(b"charpartition mine = ").expect("Could not write to nexus file");

    let mut charpart_elems = vec![];

    let alpha : f64 = alpha.to_scalar().unwrap();

    for i in 0..L {
        let freq_str = pi
            .get(i)
            .unwrap()
            .to_vec1::<f64>()
            .unwrap()
            .iter()
            .map(|v| format!("{:.6}", v))
            .collect::<Vec<String>>()
            .join("/");

        let R_data: Vec<Vec<f64>> = R.get(i).unwrap().to_vec2().unwrap();

        let mut rate_elems = vec![];
        for a in 1..20 {
            for b in 0..a {
                rate_elems.push(format!("{:.6}", R_data[a][b]));
            }
        }

        let rate_str = rate_elems.join("/");

        charpart_elems.push(format!(
            "GTR{{{}}}+F{{{}}}+G4{{{}}} : p{}",
            rate_str,
            freq_str,
            format!("{:.6}", alpha),
            i + 1
        ));
    }

    file.write(charpart_elems.join(", ").as_bytes())
        .expect("Could not write to nexus file");
    file.write(b";\nend;\n").expect("Could not write to nexus file");
}

pub struct BinarySiteModelData {
    pub rate_model_string: String,
    pub site_freq: Tensor,
    pub rate_matrices: Tensor,
}

fn read_u64_le(reader: &mut impl Read, field_name: &str) -> Result<u64> {
    let mut buf = [0u8; 8];
    reader
        .read_exact(&mut buf)
        .with_context(|| format!("Failed to read {} from binary site model file", field_name))?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f64_vec_le(reader: &mut impl Read, len: usize, field_name: &str) -> Result<Vec<f64>> {
    let byte_len = len
        .checked_mul(8)
        .context("Binary site model file field is too large")?;
    let mut bytes = vec![0u8; byte_len];
    reader
        .read_exact(&mut bytes)
        .with_context(|| format!("Failed to read {} from binary site model file", field_name))?;

    let mut out = Vec::with_capacity(len);
    for chunk in bytes.chunks_exact(8) {
        out.push(f64::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

/// Read IQ-TREE binary site model format.
///
/// Format:
/// - uint64 little-endian: byte length of rate-model string
/// - [len] bytes: rate-model string
/// - uint64 little-endian: number of sites
/// - num_sites * 20 little-endian f64: per-site frequencies
/// - num_sites * 190 little-endian f64: per-site upper-triangular rates
pub fn read_binary_site_model_file(path: &Path) -> Result<BinarySiteModelData> {
    if cfg!(target_endian = "big") {
        anyhow::bail!("Only little-endian machine is supported for reading binary site model file");
    }

    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open binary site model file: {}", path.display()))?;
    let mut reader = std::io::BufReader::new(file);

    let rate_model_string_len = usize::try_from(read_u64_le(&mut reader, "rate model string length")?)
        .context("Rate model string length does not fit in usize")?;
    let mut rate_model_string_buf = vec![0u8; rate_model_string_len];
    if rate_model_string_len > 0 {
        reader
            .read_exact(&mut rate_model_string_buf)
            .context("Failed to read rate model string from binary site model file")?;
    }
    let rate_model_string = String::from_utf8(rate_model_string_buf)
        .context("Rate model string is not valid UTF-8")?;

    let num_sites = usize::try_from(read_u64_le(&mut reader, "number of sites")?)
        .context("Number of sites does not fit in usize")?;

    let freq_len = num_sites
        .checked_mul(20)
        .context("Number of frequency values overflows usize")?;
    let site_freq_data = read_f64_vec_le(&mut reader, freq_len, "site frequencies")?;

    let rate_len = num_sites
        .checked_mul(190)
        .context("Number of rate values overflows usize")?;
    let rate_matrices_data = read_f64_vec_le(&mut reader, rate_len, "rate matrices")?;

    let site_freq = Tensor::from_vec(site_freq_data, &[num_sites, 20], &candle_core::Device::Cpu)
        .context("Failed to create site frequency tensor")?;

    let mut rate_matrices = Vec::with_capacity(num_sites * 20 * 20);

    for i in 0..num_sites {
        let mut full_matrix = vec![0f64; 20 * 20];
        let mut idx = 0;
        for a in 0..20 {
            for b in (a+1)..20 {
                let rate = rate_matrices_data[i * 190 + idx];
                full_matrix[a * 20 + b] = rate;
                full_matrix[b * 20 + a] = rate;
                idx += 1;
            }
        }
        rate_matrices.extend(full_matrix);
    }

    let rate_matrices = Tensor::from_vec(rate_matrices, &[num_sites, 20, 20], &candle_core::Device::Cpu)
        .context("Failed to create rate matrices tensor")?;

    // Convert the upper diagonal rate matrices to full R matrices
    

    Ok(BinarySiteModelData {
        rate_model_string,
        site_freq,
        rate_matrices,
    })
}