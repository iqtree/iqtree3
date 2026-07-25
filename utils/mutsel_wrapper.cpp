// C++ wrapper for the mutsel rust library

#include "mutsel_wrapper.h"

#if !defined(USE_MUTSEL)
void rust_mutsel(int32_t *parents,
                 double *branch_lengths,
                 uint8_t *alignment,
                 uint32_t site_num,
                 uint32_t leave_num,
                 uint32_t node_num,
                 double pi_reg,
                 double r_reg,
                 const char *rate_model_string,
                 const char *priorRfile,
                 const char *priorPiFile,
                 const char *substituion_model,
                 uint8_t verbose,
                 double *out_site_freq,
                 double *out_rate_matrix,
                 double *out_rate_para,
                 const char *logfile) {
                    std::cout << "Mutsel support not compiled in!" << std::endl;
                    exit(1);
                 }

void rust_set_rayon_threads(int32_t num_threads) {
    return;
}

#endif

std::pair<std::vector<double>, std::vector<int32_t>> prepare_mutsel_tree(MTree *orig_tree)
{
    // Copy tree

    MTree *tree = new MTree(*orig_tree);


    if (tree->rooted) {
        tree->convertToUnrooted();
    }

    std::vector<double> branch_lengths;
    std::vector<int32_t> parent_indices;

    auto num_nodes = tree->nodeNum;

    branch_lengths.resize(num_nodes);
    parent_indices.resize(num_nodes);

    auto orig_root = tree->root;
    ASSERT(orig_root->isLeaf());

    auto root = orig_root->neighbors[0]->node;

    std::vector<std::pair<Node *, Node *>> stack;
    stack.push_back({root, nullptr});

    branch_lengths.at(root->id) = 0.0; // root branch length
    parent_indices.at(root->id) = -1;  // root has no parent

    while (!stack.empty())
    {
        Node *node = stack.back().first;
        Node *dad = stack.back().second;
        stack.pop_back();

        for (auto nei : node->neighbors)
        {
            if (dad != nullptr && nei->node->id == dad->id)
            {
                continue;
            }
            Node *child = nei->node;

            if(nei->length < 0.0) {
                throw std::runtime_error("The guide Tree for MUTSEL needs to have branchlengths. Please provide a tree with branch lengths.");
            }

            branch_lengths.at(child->id) = nei->length;
            parent_indices.at(child->id) = node->id;

            stack.push_back({child, node});
        }
    }

    delete tree;

    return {std::move(branch_lengths), std::move(parent_indices)};
}

// Returns the alignment in the format required by mutsel library as dense [L, N] matrix and returns L the sequence length
std::tuple<std::vector<uint8_t>, int32_t, int32_t> prepare_mutsel_alignment(Alignment *alignment)
{
    // [L, N] matrix where L is sequence length and N is number of sequences
    std::vector<uint8_t> sequences;
    size_t L = alignment->getNSite();
    size_t N = alignment->getNSeq();

    std::cout << "Preparing alignment for mutsel inference: " << N << " sequences, " << L << " sites" << std::endl;

    sequences.resize(L * N);

    for (size_t site = 0; site < L; ++site)
    {
        Pattern &pat = alignment->at(alignment->getPatternID(site));
        for (size_t seq_idx = 0; seq_idx < N; ++seq_idx)
        {
            auto state = static_cast<uint8_t>(pat.at(seq_idx));
            if (!(state < 20))
            {
                // map unknown states to gaps (state 20)
                state = 20;
            }
            sequences[site * N + seq_idx] = state;
        }
    }

    return std::make_tuple(std::move(sequences), static_cast<int32_t>(L), static_cast<int32_t>(N));
}

std::string read_binary_site_model_file_internal(std::string &filename, std::vector<double> &site_freq, std::vector<double> &rate_matrices, std::vector<int> &site_model) {
    cout << endl << "Reading site-specific model file " << filename << " ..." << endl;
    
    site_freq.clear();
    site_model.clear();
    rate_matrices.clear();

    std::ifstream in;
    std::string rate_model_string;

    // assert little-endian
    uint16_t num = 1;
    if (*((uint8_t *)&num) != 1) {
        throw std::runtime_error("Only little-endian machine is supported for reading binary site model file");
    }

    try {
        in.open(filename, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("Failed to open binary site model file");
        }

        auto read_exact = [&](char *buffer, std::streamsize size, const char *field_name) {
            in.read(buffer, size);
            if (!in || in.gcount() != size) {
                throw std::runtime_error(std::string("Failed to read ") + field_name + " from binary site model file");
            }
        };
        
        // Read in the model string. First 8 bytes little endian length of the string, followed by the string itself.
        uint64_t rate_model_string_length;
        read_exact(reinterpret_cast<char*>(&rate_model_string_length), sizeof(uint64_t), "rate model string length");

        std::vector<char> rate_model_string_buf(rate_model_string_length);
        if (rate_model_string_length > 0) {
            read_exact(rate_model_string_buf.data(), static_cast<std::streamsize>(rate_model_string_length), "rate model string");
        }
        rate_model_string = std::string(rate_model_string_buf.data(), rate_model_string_length);

        // Read number of sites. 8 bytes

        uint64_t num_sites;
        read_exact(reinterpret_cast<char*>(&num_sites), sizeof(uint64_t), "number of sites");

        // num_sites * 20 doubles for site frequencies
        site_freq.resize(num_sites * 20);
        read_exact(reinterpret_cast<char*>(site_freq.data()),
                   static_cast<std::streamsize>(num_sites * 20 * sizeof(double)),
                   "site frequencies");

        // num_sites * 190 doubles for rate matrices
        rate_matrices.resize(num_sites * 190);
        read_exact(reinterpret_cast<char*>(rate_matrices.data()),
                   static_cast<std::streamsize>(num_sites * 190 * sizeof(double)),
                   "rate matrices");

        for (size_t i = 0; i < num_sites; ++i) {
            site_model.push_back(i);
            for (int j = 0; j < 20; ++j) {
                if (site_freq[i*20 + j] <= 1e-10)
                    throw std::runtime_error("Frequencies must be strictly bigger than 1e-10");
            }
            double sum = 0;
            for (int j = 0; j < 20; ++j) {
                sum += site_freq[i*20 + j];
            }
            if (std::abs(sum - 1.0) > 1e-4) {
                std::cout << "Warning: frequencies for site " << i+1 << " do not sum to 1, normalizing..." << std::endl;
                for (int j = 0; j < 20; ++j) {
                    site_freq[i*20 + j] /= sum;
                }
            }
            for (int j = 0; j < 190; ++j) {
                if (rate_matrices[i*190 + j] <= 0.0) throw "Rate parameters must be positive";
            }
        }
    }
    catch (const std::exception &e) {
        throw std::runtime_error("Error reading site model file: " + std::string(e.what()));
    }

    return rate_model_string;
}

void process_site_model_file(Alignment &alignment, const double *site_freq, const double *rate_matrices, int len) {
    alignment.ptn_state_freq.clear();
    alignment.site_rate_matrices.clear();

    size_t nsite = alignment.getNSite();
    if (len != static_cast<int>(nsite)) {
        throw std::runtime_error("Site model file site count does not match alignment length");
    }

    IntVector site_model(nsite, -1); // map each site to a model
    IntVector pattern_first_site(alignment.getNPattern(), -1);
    for (size_t site = 0; site < nsite; ++site) {
        if (pattern_first_site[alignment.getPatternID(site)] == -1) {
            pattern_first_site[alignment.getPatternID(site)] = static_cast<int>(site);
        }
    }

    bool aln_changed = false;
    ASSERT(alignment.num_states == 20); // currently we only support 20 states for mutsel model, so this function should only be called for protein alignments

    vector<double*> models_freq;
    vector<DoubleVector> models_rate;
    for (size_t site = 0; site < nsite; ++site) {
        site_model[site] = models_freq.size();

        const double *freq = site_freq + site * 20;
        const double *rate_para = rate_matrices + site * 190;

        bool add = true;
        int first_site = pattern_first_site[alignment.getPatternID(site)];
        if (first_site < static_cast<int>(site) && site_model[first_site] != -1) {
            int first_model = site_model[first_site];
            bool matched_freq_and_rate = true;
            for (int i = 0; i < 20; ++i) {
                if (freq[i] != models_freq[first_model][i]) {
                    matched_freq_and_rate = false;
                    break;
                }
            }
            if (matched_freq_and_rate) {
                for (int i = 0; i < 190; ++i) {
                    if (rate_para[i] != models_rate[first_model][i]) {
                        matched_freq_and_rate = false;
                        break;
                    }
                }
            }

            if (matched_freq_and_rate) {
                site_model[site] = first_model;
                add = false;
            } else {
                aln_changed = true;
            }
        }

        if (add) {
            double *site_freq_entry = new double[20];
            memcpy(site_freq_entry, freq, sizeof(double) * 20);
            models_freq.push_back(site_freq_entry);
            models_rate.emplace_back(rate_para, rate_para + 190);
        }
    }

    if (aln_changed) {
        cout << "Regrouping alignment sites..." << endl;
        alignment.regroupSitePattern(site_model);
        pattern_first_site = IntVector(alignment.getNPattern(), -1);
        for (size_t site = 0; site < nsite; ++site) {
            if (pattern_first_site[alignment.getPatternID(site)] == -1) {
                pattern_first_site[alignment.getPatternID(site)] = static_cast<int>(site);
            }
        }
    }

    vector<bool> used_model(models_freq.size(), false);
    for (size_t ptn = 0; ptn < alignment.getNPattern(); ++ptn) {
        int first_site = pattern_first_site[ptn];
        int model_id = site_model[first_site];
        used_model[model_id] = true;
        alignment.ptn_state_freq.push_back(models_freq[model_id]);
        alignment.site_rate_matrices.insert(
            alignment.site_rate_matrices.end(),
            models_rate[model_id].begin(),
            models_rate[model_id].end());
    }

    for (size_t model_id = 0; model_id < models_freq.size(); ++model_id) {
        if (!used_model[model_id]) {
            delete[] models_freq[model_id];
        }
    }

    cout << models_freq.size() << " distinct per-site state frequency vectors detected" << endl;
}

std::string read_site_model_file(const std::string &filename, Alignment &alignment) {
    auto site_freq = std::vector<double>();
    auto rate_matrices = std::vector<double>();
    auto site_model = std::vector<int>();
    auto rate_str = read_binary_site_model_file_internal(const_cast<std::string&>(filename), site_freq, rate_matrices, site_model);
    process_site_model_file(alignment, site_freq.data(), rate_matrices.data(), site_model.size());
    return rate_str;
}

void print_binary_site_model_file(const std::string &filename, Alignment &alignment, const std::string &rate_model_string)
{
    size_t nsites = alignment.getNSite();
    size_t nstates = alignment.num_states;
    ASSERT(nstates == 20);
    
    try {
        ofstream out;
        out.exceptions(ios::failbit | ios::badbit);
        out.open(filename, std::ios::binary);
        IntVector pattern_index;
        alignment.getSitePatternIndex(pattern_index);

        uint64_t rate_model_string_length = rate_model_string.size();
        out.write(reinterpret_cast<const char*>(&rate_model_string_length), sizeof(uint64_t));
        out.write(rate_model_string.data(), rate_model_string_length);

        uint64_t num_sites = nsites;
        out.write(reinterpret_cast<const char*>(&num_sites), sizeof(uint64_t));

        for (size_t i = 0; i < nsites; ++i) {
            double *state_freq = alignment.ptn_state_freq[pattern_index[i]];
            out.write(reinterpret_cast<const char*>(state_freq), 20 * sizeof(double));
        }

        for (size_t i = 0; i < nsites; ++i) {
            double *rate_para_ptr = alignment.site_rate_matrices.data() + pattern_index[i]*190;
            out.write(reinterpret_cast<const char*>(rate_para_ptr), 190 * sizeof(double));
        }

        cout << "Site mutsel model printed to " << filename << endl;
    } catch (ios::failure) {
        outError(ERR_WRITE_OUTPUT, filename);
    }
}

std::string rate_model_string(char rate_model_type, int num_cat, const double *para) {
    std::string result = rate_model_type + std::to_string(num_cat);
    if (rate_model_type == 'R') {
        result += "{";
        for (int i = 0; i < num_cat; ++i) {
            result += std::to_string(para[i]) + "/";
            result += std::to_string(para[num_cat + i]) + "/";
        }
        result.back() = '}'; // replace last slash with }
    } else if (rate_model_type == 'G') {
        result += "{" + std::to_string(para[0]) + "}";
    } else {
        outError("Invalid rate model type in rate_model_string function");
    }
    return result;
}

DoubleVector computeMutselSiteRates(Alignment &alignment) {
    ASSERT(alignment.num_states == 20);
    ASSERT(alignment.ptn_state_freq.size() == alignment.getNPattern());
    ASSERT(alignment.site_rate_matrices.size() == alignment.getNPattern() * 190);

    size_t npattern = alignment.getNPattern();
    DoubleVector pattern_rates(npattern);
    for (size_t ptn = 0; ptn < npattern; ++ptn) {
        double *pi = alignment.ptn_state_freq[ptn];
        double *R = alignment.site_rate_matrices.data() + ptn * 190;
        double rate = 0.0;
        int idx = 0;
        for (int i = 0; i < 20; ++i) {
            for (int j = i + 1; j < 20; ++j) {
                rate += R[idx] * pi[i] * pi[j];
                idx++;
            }
        }
        pattern_rates[ptn] = 2.0 * rate;
    }

    size_t nsite = alignment.getNSite();
    DoubleVector site_rates(nsite);
    for (size_t site = 0; site < nsite; ++site) {
        site_rates[site] = pattern_rates[alignment.getPatternID(site)];
    }
    return site_rates;
}

std::string computeMutselSiteFrequencyModel(Params &params, Alignment *alignment)
{
    ASSERT(params.tree_freq_file);
    cout << endl
         << "===> COMPUTING MUTSEL MODEL BASED ON TREE FILE " << params.tree_freq_file << endl;
    PhyloTree *tree = new PhyloTree(alignment);
    tree->setParams(&params);
    bool myrooted = params.is_rooted;
    tree->readTree(params.tree_freq_file, myrooted);
    tree->setAlignment(alignment);
    tree->setRootNode(params.root);

    tree->setNumThreads(params.num_threads);

    tree->ensureNumberOfThreadsIsSet(nullptr);

    auto [branch_lengths, parent_indices] = prepare_mutsel_tree(tree);
    auto [sequences, L, N] = prepare_mutsel_alignment(alignment);

    double *site_freq = new double[L * 20];

    double *site_rate = new double[L * 190];

    // We only allow MUTSEL{*}+{R|G|X}number model strings.

    auto plus_pos = params.model_name.find("+");

    string rate_model = "G1"; // default
    if (plus_pos != string::npos) {
        rate_model = params.model_name.substr(plus_pos + 1);
    }

    auto first_char = rate_model[0];
    if (first_char != 'G' && first_char != 'R' && first_char != 'X') {
        outError("MutSel model must be used with +G or +R or +X rate model");
    }

    auto rate_para_num = 1;
    auto num_cat = std::string();

    if (first_char != 'X') {

        num_cat = rate_model.substr(1);

        if (num_cat.empty()) {
            outError("MutSel model must be used with +G or +R rate model followed by number of categories, e.g., +G4 or +R5");
        }

        if (num_cat.find_first_not_of("0123456789") != string::npos) {
            outError("MutSel model must be used with +G or +R rate model followed by number of categories, e.g., +G4 or +R5, no additional options allowed.");
        }
        if (first_char == 'R') {
            rate_para_num = 2 * std::stoi(num_cat);
        }
    }

    double *rate_para = new double[rate_para_num];

    // Close log file, so we can append in the mutsel library without messing up the order of log messages from IQ-TREE and mutsel library
    std::cout << std::flush;
    auto outstream = dynamic_cast<outstreambuf*>(std::cout.rdbuf());
    if (outstream) {
        outstream->close();
    } else {
        throw std::runtime_error("IQTREE-Logging seems to be uninitialized");
    }

    rust_set_rayon_threads(params.num_threads);

    rust_mutsel(parent_indices.data(),
                branch_lengths.data(),
                sequences.data(),
                L,
                N,
                parent_indices.size(),
                -1.0, // pi regularization, not used
                -1.0, // r regularization, not used
                params.model_name.c_str(),
                params.mutsel_prior_rate_file.empty() ? nullptr : params.mutsel_prior_rate_file.c_str(),
                params.mutsel_prior_freq_file.empty() ? nullptr : params.mutsel_prior_freq_file.c_str(),
                "MutSel",
                verbose_mode,
                site_freq,
                site_rate,
                rate_para,
                ((string)params.out_prefix).c_str());

    outstream->open(((string)params.out_prefix+".log").c_str(), std::ios::app); // reopen log file

    std::string rate_model_str;
    if (first_char == 'X') {
        // We do not want rate categories for +X model, so we just return "R1".
        rate_model_str = "R1";
    } else {
        rate_model_str = rate_model_string(first_char, std::stoi(num_cat), rate_para);
    }

    process_site_model_file(*alignment, site_freq, site_rate, L);

    print_binary_site_model_file(((string)params.out_prefix+".sitemodel").c_str(), *alignment, rate_model_str);

    params.print_site_state_freq = WSF_NONE;
    
    delete [] site_freq;
    delete [] site_rate;
    delete [] rate_para;
    delete tree;

    cout << endl
         << "===> CONTINUE ANALYSIS USING THE INFERRED MUTSEL MODEL" << endl;

    return rate_model_str;
}