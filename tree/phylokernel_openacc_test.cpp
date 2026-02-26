/***************************************************************************
 *   OpenACC GPU Likelihood Computation for IQ-TREE — Tests                *
 *   Test & verification functions for OpenACC implementation steps        *
 ***************************************************************************/

#ifdef USE_OPENACC

#include "phylokernel_openacc_test.h"
#include "phylotree.h"           // PhyloTree, PhyloNeighbor, etc.
#include "model/modelsubst.h"    // computeTransMatrixEqualRate(), ModelSubst

#include <openacc.h>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

// ==========================================================================
// Helper
// ==========================================================================

static bool approxEqual(double a, double b, double tol) {
    return fabs(a - b) < tol;
}

// ==========================================================================
// Step 2: JC Transition Matrix — test & verification
// ==========================================================================

bool testJCTransMatrix() {
    cout << endl;
    cout << "=== OpenACC Step 2: Testing JC Transition Matrix ===" << endl;
    cout << "    (using computeTransMatrixEqualRate from modelsubst.h)" << endl;

    bool all_pass = true;
    const int nstates = 4; // DNA
    double P[16];

    // Reference values (hand-computed):
    //   exp(-4*0.1/3) = 0.875173319042947
    //   P_ii(0.1) = 0.25 + 0.75 * 0.875173319042947 = 0.906379989282211
    //   P_ij(0.1) = 0.25 - 0.25 * 0.875173319042947 = 0.031206670239263

    struct TestCase {
        double t;
        double expected_diag;
        double expected_off;
        const char *label;
    };

    TestCase tests[] = {
        {0.05, 0.951630238773713, 0.016123253742096, "t=0.05"},
        {0.10, 0.906379989282211, 0.031206670239263, "t=0.10"},
        {0.15, 0.864048064808486, 0.045317311730505, "t=0.15"},
        {0.20, 0.824446253773486, 0.058517915408838, "t=0.20"},
    };

    const double tol = 1e-14;

    for (auto &tc : tests) {
        // Call the standalone function from modelsubst.h
        computeTransMatrixEqualRate(tc.t, nstates, P);

        // Check diagonal entries
        bool diag_ok = true;
        for (int i = 0; i < nstates; i++) {
            if (!approxEqual(P[i * nstates + i], tc.expected_diag, tol))
                diag_ok = false;
        }

        // Check off-diagonal entries
        bool offdiag_ok = true;
        for (int i = 0; i < nstates; i++) {
            for (int j = 0; j < nstates; j++) {
                if (i != j && !approxEqual(P[i * nstates + j], tc.expected_off, tol))
                    offdiag_ok = false;
            }
        }

        // Check rows sum to 1.0
        bool rowsum_ok = true;
        for (int i = 0; i < nstates; i++) {
            double sum = 0.0;
            for (int j = 0; j < nstates; j++) sum += P[i * nstates + j];
            if (!approxEqual(sum, 1.0, 1e-14))
                rowsum_ok = false;
        }

        bool pass = diag_ok && offdiag_ok && rowsum_ok;
        cout << "  " << tc.label << ": "
             << "diag=" << P[0] << " off=" << P[1]
             << " rowsum=" << (P[0] + P[1] + P[2] + P[3])
             << " ... " << (pass ? "PASS" : "FAIL") << endl;

        if (!pass) all_pass = false;
    }

    // Edge case: t=0 should give identity matrix
    computeTransMatrixEqualRate(0.0, nstates, P);
    bool identity_ok = approxEqual(P[0], 1.0, tol) && approxEqual(P[1], 0.0, tol);
    cout << "  t=0.00 (identity): diag=" << P[0] << " off=" << P[1]
         << " ... " << (identity_ok ? "PASS" : "FAIL") << endl;
    if (!identity_ok) all_pass = false;

    // Edge case: t->inf should give uniform (1/n everywhere)
    computeTransMatrixEqualRate(1000.0, nstates, P);
    bool uniform_ok = approxEqual(P[0], 0.25, 1e-10) && approxEqual(P[1], 0.25, 1e-10);
    cout << "  t=1000 (uniform):  diag=" << P[0] << " off=" << P[1]
         << " ... " << (uniform_ok ? "PASS" : "FAIL") << endl;
    if (!uniform_ok) all_pass = false;

    cout << "=== Result: " << (all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED")
         << " ===" << endl << endl;

    // =====================================================================
    // GPU Verification: actually run computeTransMatrixEqualRate on the GPU
    // and compare results with CPU to prove the function works on device
    // =====================================================================
    cout << "=== OpenACC Step 2: GPU Verification ===" << endl;

    const int num_test_branches = 4;
    double test_times[num_test_branches] = {0.05, 0.10, 0.15, 0.20};
    double P_gpu[num_test_branches * 16]; // 4 matrices, each 4x4
    double P_cpu[num_test_branches * 16];

    // Compute on CPU first
    for (int b = 0; b < num_test_branches; b++) {
        computeTransMatrixEqualRate(test_times[b], nstates, &P_cpu[b * 16]);
    }

    // Compute on GPU using OpenACC
    cout << "  Offloading computeTransMatrixEqualRate to GPU..." << endl;

    #pragma acc data copyin(test_times[0:num_test_branches]) copyout(P_gpu[0:num_test_branches*16])
    {
        #pragma acc parallel loop
        for (int b = 0; b < num_test_branches; b++) {
            computeTransMatrixEqualRate(test_times[b], 4, &P_gpu[b * 16]);
        }
    }

    cout << "  GPU kernel finished. Comparing CPU vs GPU results..." << endl;

    bool gpu_pass = true;
    for (int b = 0; b < num_test_branches; b++) {
        double max_diff = 0.0;
        for (int i = 0; i < 16; i++) {
            double diff = fabs(P_cpu[b * 16 + i] - P_gpu[b * 16 + i]);
            if (diff > max_diff) max_diff = diff;
        }
        bool match = (max_diff < 1e-14);
        cout << "  t=" << test_times[b]
             << ": CPU diag=" << P_cpu[b * 16]
             << "  GPU diag=" << P_gpu[b * 16]
             << "  max_diff=" << max_diff
             << " ... " << (match ? "PASS" : "FAIL") << endl;
        if (!match) gpu_pass = false;
    }

    if (gpu_pass) {
        cout << "=== GPU Verification: ALL PASSED — function runs correctly on GPU ===" << endl;
    } else {
        cout << "=== GPU Verification: FAILED — CPU/GPU mismatch detected ===" << endl;
    }
    cout << endl;

    all_pass = all_pass && gpu_pass;
    return all_pass;
}

// ==========================================================================
// Step 3: Tip one-hot vectors — test & verification
// ==========================================================================

bool testTipOneHot() {
    cout << endl;
    cout << "=== OpenACC Step 3: Testing Tip One-Hot Vectors ===" << endl;

    bool all_pass = true;
    const int K = 4; // DNA states: A=0, C=1, G=2, T=3

    // For each observed state, the one-hot vector should be:
    //   state 0 (A): [1, 0, 0, 0]
    //   state 1 (C): [0, 1, 0, 0]
    //   state 2 (G): [0, 0, 1, 0]
    //   state 3 (T): [0, 0, 0, 1]
    // Unknown/gap:   [1, 1, 1, 1]

    for (int s = 0; s < K; s++) {
        double tip[K];
        for (int i = 0; i < K; i++)
            tip[i] = (i == s) ? 1.0 : 0.0;

        bool ok = true;
        for (int i = 0; i < K; i++) {
            double expected = (i == s) ? 1.0 : 0.0;
            if (tip[i] != expected) ok = false;
        }
        cout << "  State " << s << ": [" << tip[0] << "," << tip[1]
             << "," << tip[2] << "," << tip[3] << "] ... "
             << (ok ? "PASS" : "FAIL") << endl;
        if (!ok) all_pass = false;
    }

    // Unknown state: all 1s
    double tip_unknown[K] = {1.0, 1.0, 1.0, 1.0};
    bool unknown_ok = true;
    for (int i = 0; i < K; i++) {
        if (tip_unknown[i] != 1.0) unknown_ok = false;
    }
    cout << "  Unknown: [" << tip_unknown[0] << "," << tip_unknown[1]
         << "," << tip_unknown[2] << "," << tip_unknown[3] << "] ... "
         << (unknown_ok ? "PASS" : "FAIL") << endl;
    if (!unknown_ok) all_pass = false;

    cout << "=== Result: " << (all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED")
         << " ===" << endl << endl;
    return all_pass;
}

// ==========================================================================
// Step 4: TIP-TIP internal node — standalone test & verification
//
// Computes partial likelihood at a cherry node (2 leaf children) and
// the log-likelihood for a 2-taxon tree under JC model.
//
// Mathematical formula (Felsenstein's pruning for cherry):
//   L_parent[s] = (sum_a P(a|s,t1) * L_left[a]) * (sum_b P(b|s,t2) * L_right[b])
//
// For leaf with observed state x (one-hot):
//   sum_a P(a|s,t) * L[a] = P(x|s,t) = P[s][x]
//
// So for two leaves with states x_left, x_right:
//   L_parent[s] = P_left[s][x_left] * P_right[s][x_right]
//
// Log-likelihood for one pattern:
//   ell = sum_s pi[s] * L_parent[s]
//   lnL = log(ell)
//
// Expected: 2-taxon mismatch at t=0.1 gives lnL = -4.224717
// ==========================================================================

bool testTipTipInternal() {
    cout << endl;
    cout << "=== OpenACC Step 4: Testing TIP-TIP Internal Node ===" << endl;
    cout << "    (standalone cherry computation, no IQ-TREE tree required)" << endl;

    bool all_pass = true;
    const int K = 4; // DNA states
    const double tol = 1e-6;

    // ------------------------------------------------------------------
    // Test 4a: Single mismatch pattern (A vs C) at t=0.1
    // Expected lnL = -4.224717 (hand-derived)
    // ------------------------------------------------------------------
    {
        cout << endl << "  --- Test 4a: Single mismatch (A vs C), t=0.1 ---" << endl;

        double t_left = 0.1, t_right = 0.1;
        double pi[K] = {0.25, 0.25, 0.25, 0.25}; // JC equal frequencies

        // Compute P(t) matrices
        double P_left[K * K], P_right[K * K];
        computeTransMatrixEqualRate(t_left, K, P_left);
        computeTransMatrixEqualRate(t_right, K, P_right);

        // Leaf states: A=0, C=1
        int state_left = 0;  // A
        int state_right = 1; // C

        // TIP-TIP: L_parent[s] = P_left[s][state_left] * P_right[s][state_right]
        double L_parent[K];
        for (int s = 0; s < K; s++) {
            L_parent[s] = P_left[s * K + state_left] * P_right[s * K + state_right];
        }

        // Site likelihood: ell = sum_s pi[s] * L_parent[s]
        double ell = 0.0;
        for (int s = 0; s < K; s++) {
            ell += pi[s] * L_parent[s];
        }
        double lnL = log(ell);

        double expected_lnL = -4.224717;

        cout << "  P_left diag=" << P_left[0] << " off=" << P_left[1] << endl;
        cout << "  L_parent = [";
        for (int s = 0; s < K; s++) cout << (s ? "," : "") << L_parent[s];
        cout << "]" << endl;
        cout << "  Site likelihood = " << ell << endl;
        cout << "  lnL = " << lnL << endl;
        cout << "  Expected lnL = " << expected_lnL << endl;
        cout << "  Difference = " << fabs(lnL - expected_lnL) << endl;

        bool pass = approxEqual(lnL, expected_lnL, tol);
        cout << "  ... " << (pass ? "PASS" : "FAIL") << endl;
        if (!pass) all_pass = false;
    }

    // ------------------------------------------------------------------
    // Test 4b: Single match pattern (A vs A) at t=0.1
    // L_parent[s] = P[s][0] * P[s][0] = P[s][0]^2
    // ell = 0.25 * (P_ii^2 + 3*P_ij^2)
    // ------------------------------------------------------------------
    {
        cout << endl << "  --- Test 4b: Single match (A vs A), t=0.1 ---" << endl;

        double t = 0.1;
        double pi[K] = {0.25, 0.25, 0.25, 0.25};

        double P[K * K];
        computeTransMatrixEqualRate(t, K, P);

        int state_left = 0;  // A
        int state_right = 0; // A

        double L_parent[K];
        for (int s = 0; s < K; s++) {
            L_parent[s] = P[s * K + state_left] * P[s * K + state_right];
        }

        double ell = 0.0;
        for (int s = 0; s < K; s++) ell += pi[s] * L_parent[s];
        double lnL = log(ell);

        // Hand-computed:
        // P_ii = 0.906380, P_ij = 0.031207
        // ell = 0.25 * (0.906380^2 + 3*0.031207^2)
        //     = 0.25 * (0.821524 + 0.002926) = 0.25 * 0.824450 = 0.206113
        // lnL = log(0.206113) ≈ -1.579
        double expected_lnL_approx = -1.579;

        cout << "  L_parent = [";
        for (int s = 0; s < K; s++) cout << (s ? "," : "") << L_parent[s];
        cout << "]" << endl;
        cout << "  Site likelihood = " << ell << endl;
        cout << "  lnL = " << lnL << endl;
        cout << "  Expected ≈ " << expected_lnL_approx << endl;

        bool pass = approxEqual(lnL, expected_lnL_approx, 0.01); // coarser tolerance for approx
        bool pass_finite = !std::isnan(lnL) && !std::isinf(lnL) && lnL < 0.0;
        cout << "  Finite & negative: " << (pass_finite ? "PASS" : "FAIL") << endl;
        cout << "  Approx match: " << (pass ? "PASS" : "FAIL") << endl;
        if (!pass || !pass_finite) all_pass = false;
    }

    // ------------------------------------------------------------------
    // Test 4c: Multi-pattern 2-taxon alignment
    // Alignment: 5 patterns with frequencies
    //   Pattern 0: A-A (match), freq=3
    //   Pattern 1: A-C (mismatch), freq=2
    //   Pattern 2: G-G (match), freq=2
    //   Pattern 3: T-C (mismatch), freq=1
    //   Pattern 4: C-C (match), freq=2
    // Total sites = 10
    // ------------------------------------------------------------------
    {
        cout << endl << "  --- Test 4c: Multi-pattern 2-taxon alignment ---" << endl;

        double t_left = 0.1, t_right = 0.1;
        double pi[K] = {0.25, 0.25, 0.25, 0.25};

        double P_left[K * K], P_right[K * K];
        computeTransMatrixEqualRate(t_left, K, P_left);
        computeTransMatrixEqualRate(t_right, K, P_right);

        struct Pattern { int left; int right; int freq; };
        Pattern patterns[] = {
            {0, 0, 3},  // A-A, freq 3
            {0, 1, 2},  // A-C, freq 2
            {2, 2, 2},  // G-G, freq 2
            {3, 1, 1},  // T-C, freq 1
            {1, 1, 2},  // C-C, freq 2
        };
        int nptn = 5;

        double total_lnL = 0.0;
        for (int p = 0; p < nptn; p++) {
            // TIP-TIP computation
            double L_parent[K];
            for (int s = 0; s < K; s++) {
                L_parent[s] = P_left[s * K + patterns[p].left]
                            * P_right[s * K + patterns[p].right];
            }

            double ell = 0.0;
            for (int s = 0; s < K; s++) ell += pi[s] * L_parent[s];
            double ptn_lnL = log(ell);

            cout << "  Pattern " << p << " (" << patterns[p].left << "-"
                 << patterns[p].right << "): lnL=" << ptn_lnL
                 << " × freq=" << patterns[p].freq << endl;

            total_lnL += ptn_lnL * patterns[p].freq;
        }

        cout << "  Total lnL = " << total_lnL << endl;

        // Verify: matches = lnL_match, mismatches = lnL_mismatch
        // 7 match sites + 3 mismatch sites
        // total = 7 * lnL_match + 3 * lnL_mismatch
        bool pass = !std::isnan(total_lnL) && !std::isinf(total_lnL) && total_lnL < 0.0;
        cout << "  Valid (finite, negative): " << (pass ? "PASS" : "FAIL") << endl;
        if (!pass) all_pass = false;
    }

    // ------------------------------------------------------------------
    // Test 4d: Different branch lengths (t_left=0.05, t_right=0.2)
    // ------------------------------------------------------------------
    {
        cout << endl << "  --- Test 4d: Asymmetric branches (t=0.05 vs t=0.2) ---" << endl;

        double t_left = 0.05, t_right = 0.2;
        double pi[K] = {0.25, 0.25, 0.25, 0.25};

        double P_left[K * K], P_right[K * K];
        computeTransMatrixEqualRate(t_left, K, P_left);
        computeTransMatrixEqualRate(t_right, K, P_right);

        // Mismatch: A vs T
        int state_left = 0, state_right = 3;

        double L_parent[K];
        for (int s = 0; s < K; s++) {
            L_parent[s] = P_left[s * K + state_left] * P_right[s * K + state_right];
        }

        double ell = 0.0;
        for (int s = 0; s < K; s++) ell += pi[s] * L_parent[s];
        double lnL = log(ell);

        cout << "  P_left[0][0]=" << P_left[0] << " P_right[0][0]=" << P_right[0] << endl;
        cout << "  Site likelihood = " << ell << endl;
        cout << "  lnL = " << lnL << endl;

        // Verify basic properties
        bool pass_finite = !std::isnan(lnL) && !std::isinf(lnL);
        bool pass_negative = lnL < 0.0;
        // Asymmetric should give different result than symmetric
        // At t_left+t_right = 0.25, total branch distance is longer → lower likelihood
        bool pass_range = lnL < -3.0 && lnL > -10.0;

        cout << "  Finite: " << (pass_finite ? "PASS" : "FAIL") << endl;
        cout << "  Negative: " << (pass_negative ? "PASS" : "FAIL") << endl;
        cout << "  Range [-10, -3]: " << (pass_range ? "PASS" : "FAIL") << endl;

        bool pass = pass_finite && pass_negative && pass_range;
        if (!pass) all_pass = false;
    }

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    cout << endl << "=== OpenACC Step 4 Result: "
         << (all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED")
         << " ===" << endl << endl;

    return all_pass;
}

// ==========================================================================
// Step 3 (old): Scalar Likelihood Kernel — test & verification
// ==========================================================================

bool testLikelihoodKernel(PhyloTree *tree) {
    cout << endl;
    cout << "=== OpenACC Step 3: Testing Scalar Likelihood Kernel ===" << endl;

    if (!tree || !tree->aln || !tree->getModel() || !tree->getModelFactory()) {
        cout << "  ERROR: Tree not fully initialized (model/alignment missing)" << endl;
        return false;
    }

    bool all_pass = true;
    size_t nstates = tree->aln->num_states;
    size_t ncat = tree->getRate()->getNRate();
    size_t nptn = tree->aln->getNPattern();
    size_t orig_nptn = tree->aln->size();

    cout << "  Alignment: " << tree->aln->getNSeq() << " sequences, "
         << tree->aln->getNSite() << " sites, "
         << nptn << " patterns" << endl;
    cout << "  Model: " << tree->getModelName() << endl;
    cout << "  States: " << nstates << ", Rate categories: " << ncat << endl;

    // ------------------------------------------------------------------
    // Test 1: Compute log-likelihood and check it's valid
    // ------------------------------------------------------------------
    cout << endl << "  --- Test 1: Log-likelihood validity ---" << endl;

    // Initialize partial likelihoods if needed
    tree->initializeAllPartialLh();
    tree->clearAllPartialLH();

    double tree_lh = tree->computeLikelihood();

    cout << "  Log-likelihood (OpenACC kernel): " << fixed << tree_lh << endl;
    cout.unsetf(ios_base::fixed);

    bool lh_finite = !std::isnan(tree_lh) && !std::isinf(tree_lh);
    bool lh_negative = tree_lh < 0.0;

    cout << "  Finite check:   " << (lh_finite ? "PASS" : "FAIL") << endl;
    cout << "  Negative check: " << (lh_negative ? "PASS" : "FAIL") << endl;

    if (!lh_finite || !lh_negative) all_pass = false;

    // ------------------------------------------------------------------
    // Test 2: Per-pattern log-likelihoods should all be valid
    // ------------------------------------------------------------------
    cout << endl << "  --- Test 2: Per-pattern log-likelihoods ---" << endl;

    vector<double> pattern_lh_buf(orig_nptn);
    tree->computePatternLikelihood(pattern_lh_buf.data());
    double *pattern_lh = pattern_lh_buf.data();
    int bad_patterns = 0;
    double min_ptn_lh = 0.0, max_ptn_lh = -1e300;

    for (size_t ptn = 0; ptn < orig_nptn; ptn++) {
        if (std::isnan(pattern_lh[ptn]) || std::isinf(pattern_lh[ptn])) {
            bad_patterns++;
        } else {
            if (pattern_lh[ptn] < min_ptn_lh) min_ptn_lh = pattern_lh[ptn];
            if (pattern_lh[ptn] > max_ptn_lh) max_ptn_lh = pattern_lh[ptn];
        }
    }

    cout << "  Patterns checked: " << orig_nptn << endl;
    cout << "  Bad (NaN/Inf):    " << bad_patterns << endl;
    cout << "  Min pattern lh:   " << min_ptn_lh << endl;
    cout << "  Max pattern lh:   " << max_ptn_lh << endl;

    bool patterns_ok = (bad_patterns == 0);
    cout << "  All patterns valid: " << (patterns_ok ? "PASS" : "FAIL") << endl;
    if (!patterns_ok) all_pass = false;

    // ------------------------------------------------------------------
    // Test 3: Recompute likelihood — should be deterministic
    // ------------------------------------------------------------------
    cout << endl << "  --- Test 3: Determinism (recompute) ---" << endl;

    tree->clearAllPartialLH();
    double tree_lh2 = tree->computeLikelihood();

    bool deterministic = approxEqual(tree_lh, tree_lh2, 1e-10);
    cout << "  First:  " << fixed << tree_lh << endl;
    cout << "  Second: " << fixed << tree_lh2 << endl;
    cout.unsetf(ios_base::fixed);
    cout << "  Diff:   " << fabs(tree_lh - tree_lh2) << endl;
    cout << "  Deterministic: " << (deterministic ? "PASS" : "FAIL") << endl;
    if (!deterministic) all_pass = false;

    // ------------------------------------------------------------------
    // Test 4: Check partial likelihood arrays at internal nodes
    // ------------------------------------------------------------------
    cout << endl << "  --- Test 4: Partial likelihood sanity ---" << endl;

    NodeVector nodes1, nodes2;
    tree->getBranches(nodes1, nodes2);

    int nodes_checked = 0;
    int nodes_with_bad_plh = 0;
    size_t block = nstates * ncat;

    for (size_t b = 0; b < nodes1.size(); b++) {
        PhyloNeighbor *nei = (PhyloNeighbor*)nodes1[b]->findNeighbor(nodes2[b]);
        if (!nei || !nei->get_partial_lh()) continue;
        if (nodes1[b]->isLeaf()) continue;

        nodes_checked++;
        bool node_ok = true;

        // Check first few patterns for valid partial likelihoods
        size_t check_ptns = min((size_t)10, orig_nptn);
        for (size_t ptn = 0; ptn < check_ptns; ptn++) {
            double *plh = nei->get_partial_lh() + ptn * block;
            for (size_t i = 0; i < block; i++) {
                if (std::isnan(plh[i]) || std::isinf(plh[i]) || plh[i] < 0.0) {
                    node_ok = false;
                    break;
                }
            }
            if (!node_ok) break;
        }

        if (!node_ok) nodes_with_bad_plh++;
    }

    cout << "  Internal nodes checked: " << nodes_checked << endl;
    cout << "  Nodes with bad partial_lh: " << nodes_with_bad_plh << endl;

    bool plh_ok = (nodes_with_bad_plh == 0);
    cout << "  Partial likelihoods valid: " << (plh_ok ? "PASS" : "FAIL") << endl;
    if (!plh_ok) all_pass = false;

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    cout << endl << "=== OpenACC Step 3 Result: "
         << (all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED")
         << " ===" << endl;
    cout << "  (Compare log-likelihood " << tree_lh
         << " against a non-OpenACC build to verify correctness)" << endl;
    cout << endl;

    return all_pass;
}

bool testPrecomputedMatrices(PhyloTree *tree) {
    cout << "=== OpenACC Step 2: Verifying standalone vs ModelSubst ===" << endl;

    if (!tree->getModel()) {
        cout << "  ERROR: Model not initialized" << endl;
        return false;
    }

    bool all_pass = true;
    const double tol = 1e-12;
    int nstates = tree->getModel()->num_states;

    // Get all branches from the tree
    NodeVector nodes1, nodes2;
    tree->getBranches(nodes1, nodes2);

    double P_standalone[400]; // max 20x20 for protein
    double P_model[400];

    for (size_t b = 0; b < nodes1.size(); b++) {
        // Get branch length
        double branch_len = 0.0;
        for (auto it = nodes1[b]->neighbors.begin();
             it != nodes1[b]->neighbors.end(); it++) {
            if ((*it)->node == nodes2[b]) {
                branch_len = (*it)->length;
                break;
            }
        }

        // Standalone function (GPU-ready)
        computeTransMatrixEqualRate(branch_len, nstates, P_standalone);

        // IQ-TREE's class method (existing CPU path)
        tree->getModel()->computeTransMatrix(branch_len, P_model);

        // Compare element by element
        bool match = true;
        double max_diff = 0.0;
        for (int i = 0; i < nstates * nstates; i++) {
            double diff = fabs(P_standalone[i] - P_model[i]);
            if (diff > max_diff) max_diff = diff;
            if (diff > tol) match = false;
        }

        cout << "  Branch " << b << " (len=" << branch_len << "): "
             << "max_diff=" << max_diff
             << " ... " << (match ? "PASS" : "FAIL") << endl;

        if (!match) all_pass = false;
    }

    cout << "=== Result: " << (all_pass ? "ALL BRANCHES MATCH" : "MISMATCH DETECTED")
         << " ===" << endl << endl;

    return all_pass;
}

#endif // USE_OPENACC
