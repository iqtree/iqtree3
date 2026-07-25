// C++ wrapper for the mutsel rust library

#ifndef IQTREE_UTILS_MUTSEL_WRAPPER_H
#define IQTREE_UTILS_MUTSEL_WRAPPER_H

#include <vector>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include "tree/mtree.h"
#include "alignment/alignment.h"
#include "tree/phylotree.h"
#include "main/outstreambuf.h"

/**
 * Convert a IQ-TREE tree into the format required by the mutsel library.
 */
std::pair<std::vector<double>, std::vector<int32_t>> prepare_mutsel_tree(MTree *tree);

/**
 * Returns the alignment in the format required by mutsel library as dense [L, N] matrix and returns L the sequence length and N the number of sequences.
 */
std::tuple<std::vector<uint8_t>, int32_t, int32_t> prepare_mutsel_alignment(Alignment *alignment);


// Mutsel internals, if not compiled in we provide dummy implementations here, otherwise these functions need to be linked.
#if defined(USE_MUTSEL)
extern "C" void rust_mutsel(int32_t *parents,
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
                            const char *substitution_model,
                            uint8_t verbose,
                            double *out_site_freq,
                            double *out_rate_matrix,
                            double *out_rate_para,
                            const char *logfile);

extern "C" void rust_set_rayon_threads(int32_t num_threads);
#else
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
                 const char *logfile);

void rust_set_rayon_threads(int32_t num_threads);
#endif

/**
 * Simple read function which just puts the binary file into vectors. frequences get normalized, rates are checked to be positive. It returns the rate model string of the file.
 * Format is documented in the body of the function.
 */
std::string read_binary_site_model_file_internal(std::string &filename, std::vector<double> &site_freq, std::vector<double> &rate_matrices, std::vector<int> &site_model);

/**
 * Takes the data from the file (or the rust_mutsel function) and updates the alignment object.
 * This deals with duplicated pattern and stuff.
 */
void process_site_model_file(Alignment &alignment, const double *site_freq, const double *rate_matrices, int len);

/**
 * Read a site model file and update the alignment object.
 * This is a simple wrapper function around read_binary_site_model_file_internal and process_site_model_file.
 */
std::string read_site_model_file(const std::string &filename, Alignment &alignment);

/**
 * Print a site model from the alignment to binary file.
 */
void print_binary_site_model_file(const std::string &filename, Alignment &alignment, const std::string &rate_model_string);

/**
 * Rate model to string. Currently FreeRate and Gamma.
 */
std::string rate_model_string(char rate_model_type, int num_cat, const double *para);

/**
 * Compute the site frequency model using the mutsel library. It writes a site model file with the inferred parameters and also updates the alignment object with the inferred site frequency and rate parameters. It returns a string representing the rate model, e.g., "G4{0.5}" or "R5{0.1/0.2/0.3/0.4/0.5/0.6/0.7/0.8/0.9/1.0}".
 */
std::string computeMutselSiteFrequencyModel(Params &params, Alignment *alignment);

/**
 * Per-site substitution rate implied by the fitted MUTSEL model
 */
DoubleVector computeMutselSiteRates(Alignment &alignment);

#endif // IQTREE_UTILS_MUTSEL_WRAPPER_H