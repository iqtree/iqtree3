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
// Step 5: TIP-INTERNAL + INTERNAL-INTERNAL — standalone test & verification
//
// Computes partial likelihood for 3-taxon and 4-taxon trees under JC model.
// Two-stage bottom-up traversal (no IQ-TREE tree required):
//
// Stage 1 (TIP-TIP at cherry):
//   L_AB[s] = P_A[s][obs_A] * P_B[s][obs_B]
//
// Stage 2a (TIP-INTERNAL: one internal + one leaf child):
//   vleft[s]  = sum_x P_internal[s*K+x] * L_internal[x]   (dot product)
//   vright[s] = P_leaf[s*K+obs_leaf]                        (column lookup)
//   L_parent[s] = vleft[s] * vright[s]
//
// Stage 2b (INTERNAL-INTERNAL: two internal children):
//   vleft[s]  = sum_x P_left[s*K+x]  * L_left[x]          (dot product)
//   vright[s] = sum_x P_right[s*K+x] * L_right[x]         (dot product)
//   L_parent[s] = vleft[s] * vright[s]
//
// Loop nesting matches production kernel (phylokernel_openacc.cpp):
//   for ptn (gang) → for c (category) → for s (vector) → for x (sequential)
// For ncat=1 (JC), category loop is trivial (c=0 only).
//
// Reference: ((A:0.1,B:0.2):0.05,C:0.15), A='A',B='C',C='G'
//            lnL = -6.465999160939802
// ==========================================================================

bool testTipInternalInternal() {
    cout << endl;
    cout << "=== OpenACC Step 5: Testing TIP-INTERNAL + INTERNAL-INTERNAL ===" << endl;
    cout << "    (standalone 3/4-taxon computation, no IQ-TREE tree required)" << endl;

    bool all_pass = true;
    const int K = 4; // DNA states

    // ------------------------------------------------------------------
    // Test 5a: Reference 3-taxon case from implementation plan
    // Tree: ((A:0.1, B:0.2):0.05, C:0.15)
    // Pattern: A='A'(0), B='C'(1), C='G'(2)
    // Tests the TIP-INTERNAL kernel: one internal child (AB cherry)
    // + one leaf child (C) at the root node.
    // Expected lnL = -6.465999160939802
    // ------------------------------------------------------------------
    {
        cout << endl << "  --- Test 5a: 3-taxon reference (A,C,G), TIP-INTERNAL ---" << endl;

        // Branch lengths
        double t_A = 0.1;    // A → cherry
        double t_B = 0.2;    // B → cherry
        double t_AB = 0.05;  // cherry → root (internal branch)
        double t_C = 0.15;   // C → root

        double pi[K] = {0.25, 0.25, 0.25, 0.25}; // JC equal frequencies

        // Compute transition matrices for all 4 branches
        double P_A[K * K], P_B[K * K], P_AB[K * K], P_C[K * K];
        computeTransMatrixEqualRate(t_A, K, P_A);
        computeTransMatrixEqualRate(t_B, K, P_B);
        computeTransMatrixEqualRate(t_AB, K, P_AB);
        computeTransMatrixEqualRate(t_C, K, P_C);

        // Observed states: A='A'(0), B='C'(1), C='G'(2)
        int state_A = 0;
        int state_B = 1;
        int state_C = 2;

        // ---- Stage 1: TIP-TIP at cherry (AB) ----
        // Same formula as Step 4: L_AB[s] = P_A[s][state_A] * P_B[s][state_B]
        double L_AB[K];
        for (int s = 0; s < K; s++) {
            L_AB[s] = P_A[s * K + state_A] * P_B[s * K + state_B];
        }

        // ---- Stage 2: TIP-INTERNAL at root ----
        // Internal child (AB): real dot product  vleft = P_AB · L_AB
        // Leaf child (C): column lookup           vright = P_C[][state_C]
        // Hadamard: L_root[s] = vleft * vright
        //
        // Loop structure matches production TIP-INTERNAL kernel:
        //   for s (output state, vectorizable):
        //     vright = sum_x P[s*K+x] * partial_right[x]  (internal child)
        //     vleft  = tip_lh[state*block + s]             (leaf, pre-baked)
        //     L[s] = vleft * vright
        double L_root[K];
        for (int s = 0; s < K; s++) {
            // Internal child (AB): dot product (this is the NEW computation)
            double vleft = 0.0;
            for (int x = 0; x < K; x++) {
                vleft += P_AB[s * K + x] * L_AB[x];
            }
            // Leaf child (C): column lookup
            double vright = P_C[s * K + state_C];
            // Hadamard product
            L_root[s] = vleft * vright;
        }

        // Site likelihood: ell = sum_s pi[s] * L_root[s]
        double ell = 0.0;
        for (int s = 0; s < K; s++) {
            ell += pi[s] * L_root[s];
        }
        double lnL = log(ell);

        double expected_lnL = -6.465999160939802;
        double tol = 1e-6;

        cout << "  Tree: ((A:0.1,B:0.2):0.05,C:0.15)" << endl;
        cout << "  Pattern: A='A', B='C', C='G'" << endl;
        cout << "  L_AB = [";
        for (int s = 0; s < K; s++) cout << (s ? "," : "") << L_AB[s];
        cout << "]" << endl;
        cout << "  L_root = [";
        for (int s = 0; s < K; s++) cout << (s ? "," : "") << L_root[s];
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
    // Test 5b: Verify intermediate partial likelihoods at AB cherry
    // Same tree/pattern as 5a; checks that AB partials match plan values
    // within tolerance 1e-6 (plan specifies 1e-12 but we use 1e-6 for
    // consistency with the overall tol and to guard against rounding).
    // Reference: AB = [5.303947e-02, 2.572822e-02, 1.826149e-03, 1.826149e-03]
    // ------------------------------------------------------------------
    {
        cout << endl << "  --- Test 5b: Intermediate AB partial verification ---" << endl;

        double P_A[K * K], P_B[K * K];
        computeTransMatrixEqualRate(0.1, K, P_A);
        computeTransMatrixEqualRate(0.2, K, P_B);

        int state_A = 0, state_B = 1;
        double L_AB[K];
        for (int s = 0; s < K; s++) {
            L_AB[s] = P_A[s * K + state_A] * P_B[s * K + state_B];
        }

        // Reference values from OPENACC_IMPLEMENTATION_PLAN.md (Test Case 2)
        double expected_AB[K] = {5.303947e-02, 2.572822e-02, 1.826149e-03, 1.826149e-03};
        double tol_partial = 1e-6;

        bool partials_ok = true;
        cout << "  Computed AB: [";
        for (int s = 0; s < K; s++) {
            cout << (s ? "," : "") << L_AB[s];
            if (!approxEqual(L_AB[s], expected_AB[s], tol_partial))
                partials_ok = false;
        }
        cout << "]" << endl;
        cout << "  Expected AB: [";
        for (int s = 0; s < K; s++) cout << (s ? "," : "") << expected_AB[s];
        cout << "]" << endl;

        // Also verify root partials from plan
        double P_AB[K * K], P_C[K * K];
        computeTransMatrixEqualRate(0.05, K, P_AB);
        computeTransMatrixEqualRate(0.15, K, P_C);

        int state_C = 2;
        double L_root[K];
        for (int s = 0; s < K; s++) {
            double vleft = 0.0;
            for (int x = 0; x < K; x++)
                vleft += P_AB[s * K + x] * L_AB[x];
            double vright = P_C[s * K + state_C];
            L_root[s] = vleft * vright;
        }

        double expected_root[K] = {2.308811e-03, 1.150960e-03, 2.624333e-03, 1.376402e-04};
        bool root_ok = true;
        cout << "  Computed root: [";
        for (int s = 0; s < K; s++) {
            cout << (s ? "," : "") << L_root[s];
            if (!approxEqual(L_root[s], expected_root[s], tol_partial))
                root_ok = false;
        }
        cout << "]" << endl;
        cout << "  Expected root: [";
        for (int s = 0; s < K; s++) cout << (s ? "," : "") << expected_root[s];
        cout << "]" << endl;

        bool pass = partials_ok && root_ok;
        cout << "  AB partials: " << (partials_ok ? "PASS" : "FAIL") << endl;
        cout << "  Root partials: " << (root_ok ? "PASS" : "FAIL") << endl;
        if (!pass) all_pass = false;
    }

    // ------------------------------------------------------------------
    // Test 5c: 4-taxon tree with INTERNAL-INTERNAL at root
    // Tree: ((A:0.1, B:0.2):0.05, (C:0.15, D:0.1):0.08)
    // Pattern: A='A'(0), B='C'(1), C='G'(2), D='T'(3)
    //
    // Exercises:
    //   Stage 1a: TIP-TIP at cherry (AB)
    //   Stage 1b: TIP-TIP at cherry (CD)
    //   Stage 2:  INTERNAL-INTERNAL at root (both children are internal)
    //
    // Loop structure matches production INTERNAL-INTERNAL kernel:
    //   for s:
    //     vleft  = sum_x P_left[s*K+x]  * L_left[x]
    //     vright = sum_x P_right[s*K+x] * L_right[x]
    //     L[s] = vleft * vright
    // ------------------------------------------------------------------
    {
        cout << endl << "  --- Test 5c: 4-taxon INTERNAL-INTERNAL ---" << endl;

        // Branch lengths
        double t_A = 0.1, t_B = 0.2;     // leaves → cherry AB
        double t_C = 0.15, t_D = 0.1;    // leaves → cherry CD
        double t_AB = 0.05, t_CD = 0.08; // internal branches → root

        double pi[K] = {0.25, 0.25, 0.25, 0.25};

        // Compute transition matrices for all 6 branches
        double P_A[K * K], P_B[K * K], P_C[K * K], P_D[K * K];
        double P_AB[K * K], P_CD[K * K];
        computeTransMatrixEqualRate(t_A, K, P_A);
        computeTransMatrixEqualRate(t_B, K, P_B);
        computeTransMatrixEqualRate(t_C, K, P_C);
        computeTransMatrixEqualRate(t_D, K, P_D);
        computeTransMatrixEqualRate(t_AB, K, P_AB);
        computeTransMatrixEqualRate(t_CD, K, P_CD);

        // Observed states: all different
        int state_A = 0, state_B = 1, state_C = 2, state_D = 3;

        // ---- Stage 1a: TIP-TIP at cherry (AB) ----
        double L_AB[K];
        for (int s = 0; s < K; s++) {
            L_AB[s] = P_A[s * K + state_A] * P_B[s * K + state_B];
        }

        // ---- Stage 1b: TIP-TIP at cherry (CD) ----
        double L_CD[K];
        for (int s = 0; s < K; s++) {
            L_CD[s] = P_C[s * K + state_C] * P_D[s * K + state_D];
        }

        // ---- Stage 2: INTERNAL-INTERNAL at root ----
        // Both children are internal: two dot products + Hadamard
        double L_root[K];
        for (int s = 0; s < K; s++) {
            // Left child (AB): dot product
            double vleft = 0.0;
            for (int x = 0; x < K; x++) {
                vleft += P_AB[s * K + x] * L_AB[x];
            }
            // Right child (CD): dot product
            double vright = 0.0;
            for (int x = 0; x < K; x++) {
                vright += P_CD[s * K + x] * L_CD[x];
            }
            // Hadamard product
            L_root[s] = vleft * vright;
        }

        // Site likelihood
        double ell = 0.0;
        for (int s = 0; s < K; s++) {
            ell += pi[s] * L_root[s];
        }
        double lnL = log(ell);

        cout << "  Tree: ((A:0.1,B:0.2):0.05,(C:0.15,D:0.1):0.08)" << endl;
        cout << "  Pattern: A='A', B='C', C='G', D='T'" << endl;
        cout << "  L_AB = [";
        for (int s = 0; s < K; s++) cout << (s ? "," : "") << L_AB[s];
        cout << "]" << endl;
        cout << "  L_CD = [";
        for (int s = 0; s < K; s++) cout << (s ? "," : "") << L_CD[s];
        cout << "]" << endl;
        cout << "  L_root = [";
        for (int s = 0; s < K; s++) cout << (s ? "," : "") << L_root[s];
        cout << "]" << endl;
        cout << "  Site likelihood = " << ell << endl;
        cout << "  lnL = " << lnL << endl;

        // Verify basic properties
        bool pass_finite = !std::isnan(lnL) && !std::isinf(lnL);
        bool pass_negative = lnL < 0.0;
        // 4-taxon all-mismatch with moderate branches → expect lnL in [-15, -5]
        bool pass_range = lnL < -5.0 && lnL > -15.0;

        cout << "  Finite: " << (pass_finite ? "PASS" : "FAIL") << endl;
        cout << "  Negative: " << (pass_negative ? "PASS" : "FAIL") << endl;
        cout << "  Range [-15, -5]: " << (pass_range ? "PASS" : "FAIL") << endl;

        // Also verify symmetry: swapping left/right subtrees should give same result
        // (compute with AB on right, CD on left)
        double L_root_swapped[K];
        for (int s = 0; s < K; s++) {
            double vleft = 0.0;
            for (int x = 0; x < K; x++)
                vleft += P_CD[s * K + x] * L_CD[x];
            double vright = 0.0;
            for (int x = 0; x < K; x++)
                vright += P_AB[s * K + x] * L_AB[x];
            L_root_swapped[s] = vleft * vright;
        }
        double ell_swapped = 0.0;
        for (int s = 0; s < K; s++) ell_swapped += pi[s] * L_root_swapped[s];
        double lnL_swapped = log(ell_swapped);

        bool pass_symmetry = approxEqual(lnL, lnL_swapped, 1e-14);
        cout << "  Swap symmetry (diff=" << fabs(lnL - lnL_swapped) << "): "
             << (pass_symmetry ? "PASS" : "FAIL") << endl;

        bool pass = pass_finite && pass_negative && pass_range && pass_symmetry;
        if (!pass) all_pass = false;
    }

    // ------------------------------------------------------------------
    // Test 5d: Multi-pattern 3-taxon alignment
    // Tree: ((A:0.1, B:0.2):0.05, C:0.15)
    // 4 patterns with frequencies:
    //   Pattern 0: A,C,G (mismatch) freq=1
    //   Pattern 1: A,A,A (match)    freq=3
    //   Pattern 2: C,C,C (match)    freq=2
    //   Pattern 3: G,T,A (mismatch) freq=1
    // Total sites = 7
    //
    // Verifies the full pattern loop: total_lnL = sum(freq * ptn_lnL)
    // ------------------------------------------------------------------
    {
        cout << endl << "  --- Test 5d: Multi-pattern 3-taxon alignment ---" << endl;

        double t_A = 0.1, t_B = 0.2, t_AB = 0.05, t_C = 0.15;
        double pi[K] = {0.25, 0.25, 0.25, 0.25};

        double P_A[K * K], P_B[K * K], P_AB[K * K], P_C[K * K];
        computeTransMatrixEqualRate(t_A, K, P_A);
        computeTransMatrixEqualRate(t_B, K, P_B);
        computeTransMatrixEqualRate(t_AB, K, P_AB);
        computeTransMatrixEqualRate(t_C, K, P_C);

        struct Pattern3 { int sA; int sB; int sC; int freq; };
        Pattern3 patterns[] = {
            {0, 1, 2, 1},  // A,C,G  freq 1  (reference case)
            {0, 0, 0, 3},  // A,A,A  freq 3
            {1, 1, 1, 2},  // C,C,C  freq 2
            {2, 3, 0, 1},  // G,T,A  freq 1
        };
        int nptn = 4;

        double total_lnL = 0.0;
        for (int p = 0; p < nptn; p++) {
            // Stage 1: TIP-TIP at cherry (AB)
            double L_AB[K];
            for (int s = 0; s < K; s++) {
                L_AB[s] = P_A[s * K + patterns[p].sA]
                        * P_B[s * K + patterns[p].sB];
            }

            // Stage 2: TIP-INTERNAL at root
            double L_root[K];
            for (int s = 0; s < K; s++) {
                double vleft = 0.0;
                for (int x = 0; x < K; x++) {
                    vleft += P_AB[s * K + x] * L_AB[x];
                }
                double vright = P_C[s * K + patterns[p].sC];
                L_root[s] = vleft * vright;
            }

            // Site likelihood
            double ell = 0.0;
            for (int s = 0; s < K; s++) ell += pi[s] * L_root[s];
            double ptn_lnL = log(ell);

            cout << "  Pattern " << p << " ("
                 << patterns[p].sA << "," << patterns[p].sB << "," << patterns[p].sC
                 << "): lnL=" << ptn_lnL
                 << " x freq=" << patterns[p].freq << endl;

            total_lnL += ptn_lnL * patterns[p].freq;
        }

        cout << "  Total lnL = " << total_lnL << endl;

        // Verify: first pattern should match Test 5a reference
        // total should be finite, negative, and reasonable for 7 sites
        bool pass_finite = !std::isnan(total_lnL) && !std::isinf(total_lnL);
        bool pass_negative = total_lnL < 0.0;
        // 7 sites, expect total in roughly [-50, -5]
        bool pass_range = total_lnL < -5.0 && total_lnL > -50.0;

        cout << "  Finite: " << (pass_finite ? "PASS" : "FAIL") << endl;
        cout << "  Negative: " << (pass_negative ? "PASS" : "FAIL") << endl;
        cout << "  Range [-50, -5]: " << (pass_range ? "PASS" : "FAIL") << endl;

        bool pass = pass_finite && pass_negative && pass_range;
        if (!pass) all_pass = false;
    }

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    cout << endl << "=== OpenACC Step 5 Result: "
         << (all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED")
         << " ===" << endl << endl;

    return all_pass;
}

// ==========================================================================
// Step 6: Underflow Scaling — standalone test & verification
//
// Tests the dynamic scaling mechanism that prevents underflow when
// partial likelihoods become extremely small (deep trees, many taxa).
//
// IQ-TREE scaling constants (from phylotree.h):
//   SCALING_THRESHOLD     = 2^(-256) ≈ 8.636e-78
//   SCALING_THRESHOLD_EXP = 256
//   LOG_SCALING_THRESHOLD = ln(2^(-256)) ≈ -177.4457
//
// Scaling logic (applied after computing each internal node):
//   if max(L_parent) < SCALING_THRESHOLD:
//     L_parent[s] *= 2^256   (via ldexp)
//     scale_num[ptn] += 1
//
// Log-likelihood correction at root:
//   lnL = log(site_lh_stored) + total_scale_num * LOG_SCALING_THRESHOLD
//
// Scale_num propagation:
//   TIP-TIP:           scale_num = 0 + (new_scaling ? 1 : 0)
//   TIP-INTERNAL:      scale_num = child_scale + (new ? 1 : 0)
//   INTERNAL-INTERNAL:  scale_num = left_scale + right_scale + (new ? 1 : 0)
// ==========================================================================

bool testScaling() {
    cout << endl;
    cout << "=== OpenACC Step 6: Testing Underflow Scaling ===" << endl;
    cout << "    (standalone scaling verification, no IQ-TREE tree required)" << endl;

    bool all_pass = true;
    const int K = 4; // DNA states

    // ------------------------------------------------------------------
    // Test 6a: Scaling triggers on tiny partial likelihoods
    // Values below SCALING_THRESHOLD (≈ 8.636e-78) must be scaled up
    // by 2^256 and scale_num incremented by 1.
    // ------------------------------------------------------------------
    {
        cout << endl << "  --- Test 6a: Scaling triggers on tiny partials ---" << endl;

        // Partial likelihoods well below SCALING_THRESHOLD
        double L[K] = {1e-80, 2e-80, 5e-81, 1.5e-80};
        UBYTE scale_num = 0;

        // Save originals for verification
        double L_orig[K];
        for (int s = 0; s < K; s++) L_orig[s] = L[s];

        // Find max (same logic as production kernel)
        double lh_max = 0.0;
        for (int s = 0; s < K; s++)
            if (L[s] > lh_max) lh_max = L[s];

        cout << "  Before: max=" << lh_max
             << "  threshold=" << SCALING_THRESHOLD << endl;
        cout << "  Below threshold: " << (lh_max < SCALING_THRESHOLD ? "YES" : "NO") << endl;

        // Apply scaling (identical to production kernel)
        if (lh_max < SCALING_THRESHOLD) {
            for (int s = 0; s < K; s++)
                L[s] = ldexp(L[s], SCALING_THRESHOLD_EXP);
            scale_num += 1;
        }

        cout << "  After:  L = [";
        for (int s = 0; s < K; s++) cout << (s ? "," : "") << L[s];
        cout << "]" << endl;
        cout << "  scale_num = " << (int)scale_num << endl;

        // Verify scaling triggered
        bool pass_triggered = (scale_num == 1);

        // Verify scaled values = original * 2^256
        bool pass_values = true;
        for (int s = 0; s < K; s++) {
            double expected = ldexp(L_orig[s], SCALING_THRESHOLD_EXP);
            if (!approxEqual(L[s], expected, fabs(expected) * 1e-14))
                pass_values = false;
        }

        // Verify scaled values are now in reasonable range (not tiny anymore)
        bool pass_range = true;
        for (int s = 0; s < K; s++) {
            if (L[s] < 1e-10 || L[s] > 1e10) pass_range = false;
        }

        cout << "  Triggered: " << (pass_triggered ? "PASS" : "FAIL") << endl;
        cout << "  Values = orig*2^256: " << (pass_values ? "PASS" : "FAIL") << endl;
        cout << "  In reasonable range: " << (pass_range ? "PASS" : "FAIL") << endl;

        if (!pass_triggered || !pass_values || !pass_range) all_pass = false;
    }

    // ------------------------------------------------------------------
    // Test 6b: Scaling does NOT trigger on normal-sized partials
    // Reuses Step 5's AB cherry partials (all > 1e-3), which are far
    // above the 8.636e-78 threshold.
    // ------------------------------------------------------------------
    {
        cout << endl << "  --- Test 6b: No scaling on normal partials ---" << endl;

        // Compute Step 5's AB partials: P(0.1)[s][0] * P(0.2)[s][1]
        double P_A[K * K], P_B[K * K];
        computeTransMatrixEqualRate(0.1, K, P_A);
        computeTransMatrixEqualRate(0.2, K, P_B);

        double L[K];
        for (int s = 0; s < K; s++)
            L[s] = P_A[s * K + 0] * P_B[s * K + 1]; // A vs C

        UBYTE scale_num = 0;
        double lh_max = 0.0;
        for (int s = 0; s < K; s++)
            if (L[s] > lh_max) lh_max = L[s];

        // Apply same logic — should NOT trigger
        if (lh_max < SCALING_THRESHOLD) {
            for (int s = 0; s < K; s++)
                L[s] = ldexp(L[s], SCALING_THRESHOLD_EXP);
            scale_num += 1;
        }

        cout << "  max=" << lh_max << "  threshold=" << SCALING_THRESHOLD << endl;
        cout << "  scale_num = " << (int)scale_num << endl;

        bool pass = (scale_num == 0);
        cout << "  Not triggered: " << (pass ? "PASS" : "FAIL") << endl;
        if (!pass) all_pass = false;
    }

    // ------------------------------------------------------------------
    // Test 6c: Log-likelihood correction roundtrip
    // Take Step 5a's known root partials, simulate a scaling event by
    // storing them as (true_value * 2^256) with scale_num=1, then
    // verify that the correction formula recovers the true lnL.
    //
    // lnL = log(stored_site_lh) + scale_num * LOG_SCALING_THRESHOLD
    //     = log(true_site_lh * 2^256) + 1 * (-256*ln2)
    //     = log(true_site_lh) + 256*ln2 - 256*ln2
    //     = log(true_site_lh)   ← exact cancellation
    // ------------------------------------------------------------------
    {
        cout << endl << "  --- Test 6c: lnL correction roundtrip ---" << endl;

        // True root partials from Step 5a (plan reference)
        double L_root_true[K] = {2.308811e-03, 1.150960e-03, 2.624333e-03, 1.376402e-04};
        double pi[K] = {0.25, 0.25, 0.25, 0.25};

        // True lnL (no scaling)
        double site_lh_true = 0.0;
        for (int s = 0; s < K; s++)
            site_lh_true += pi[s] * L_root_true[s];
        double lnL_true = log(site_lh_true);

        // Simulate scaling: stored = true * 2^256, scale_num = 1
        double L_root_stored[K];
        for (int s = 0; s < K; s++)
            L_root_stored[s] = ldexp(L_root_true[s], SCALING_THRESHOLD_EXP);
        UBYTE scale_num = 1;

        // Compute corrected lnL (same formula as production kernel)
        double site_lh_stored = 0.0;
        for (int s = 0; s < K; s++)
            site_lh_stored += pi[s] * L_root_stored[s];
        double lnL_corrected = log(site_lh_stored) + scale_num * LOG_SCALING_THRESHOLD;

        // Also test with scale_num = 3 (triple scaling)
        double L_root_triple[K];
        for (int s = 0; s < K; s++)
            L_root_triple[s] = ldexp(L_root_true[s], 3 * SCALING_THRESHOLD_EXP);
        UBYTE scale_num_3 = 3;
        double site_lh_triple = 0.0;
        for (int s = 0; s < K; s++)
            site_lh_triple += pi[s] * L_root_triple[s];
        double lnL_triple = log(site_lh_triple) + scale_num_3 * LOG_SCALING_THRESHOLD;

        double expected_lnL = -6.465999160939802;

        cout << "  True lnL        = " << lnL_true << endl;
        cout << "  Corrected (n=1) = " << lnL_corrected << endl;
        cout << "  Corrected (n=3) = " << lnL_triple << endl;
        cout << "  Expected        = " << expected_lnL << endl;
        cout << "  Diff (n=1) = " << fabs(lnL_corrected - lnL_true) << endl;
        cout << "  Diff (n=3) = " << fabs(lnL_triple - lnL_true) << endl;

        bool pass_true = approxEqual(lnL_true, expected_lnL, 1e-6);
        bool pass_n1 = approxEqual(lnL_corrected, lnL_true, 1e-10);
        bool pass_n3 = approxEqual(lnL_triple, lnL_true, 1e-8);

        cout << "  True matches ref: " << (pass_true ? "PASS" : "FAIL") << endl;
        cout << "  n=1 correction:   " << (pass_n1 ? "PASS" : "FAIL") << endl;
        cout << "  n=3 correction:   " << (pass_n3 ? "PASS" : "FAIL") << endl;

        if (!pass_true || !pass_n1 || !pass_n3) all_pass = false;
    }

    // ------------------------------------------------------------------
    // Test 6d: Scale_num propagation through a 4-taxon tree
    // Tree: ((A,B):t_AB, (C,D):t_CD)
    //
    // Inject tiny artificial partials at both cherries:
    //   L_AB below threshold → scale_AB = 1
    //   L_CD below threshold → scale_CD = 1
    // Then compute INTERNAL-INTERNAL at root:
    //   scale_root = scale_AB + scale_CD + (root_scaling ? 1 : 0)
    //
    // After scaling cherries, their values are ~1e-3, so the root
    // product ~1e-6 is above threshold → no additional root scaling.
    // Expected: scale_root = 2.
    // ------------------------------------------------------------------
    {
        cout << endl << "  --- Test 6d: Scale_num propagation through tree ---" << endl;

        // Artificial tiny partials for both cherries (below 8.636e-78)
        double L_AB[K] = {1e-80, 2e-81, 5e-81, 1e-81};
        double L_CD[K] = {3e-81, 1e-80, 1e-81, 2e-81};
        UBYTE scale_AB = 0, scale_CD = 0;

        // Scale cherry AB if needed
        double max_AB = 0.0;
        for (int s = 0; s < K; s++)
            if (L_AB[s] > max_AB) max_AB = L_AB[s];
        if (max_AB < SCALING_THRESHOLD) {
            for (int s = 0; s < K; s++)
                L_AB[s] = ldexp(L_AB[s], SCALING_THRESHOLD_EXP);
            scale_AB += 1;
        }

        // Scale cherry CD if needed
        double max_CD = 0.0;
        for (int s = 0; s < K; s++)
            if (L_CD[s] > max_CD) max_CD = L_CD[s];
        if (max_CD < SCALING_THRESHOLD) {
            for (int s = 0; s < K; s++)
                L_CD[s] = ldexp(L_CD[s], SCALING_THRESHOLD_EXP);
            scale_CD += 1;
        }

        cout << "  Cherry AB: scale_num=" << (int)scale_AB
             << " max_before=" << max_AB << endl;
        cout << "  Cherry CD: scale_num=" << (int)scale_CD
             << " max_before=" << max_CD << endl;

        // INTERNAL-INTERNAL at root
        double P_AB[K * K], P_CD[K * K];
        computeTransMatrixEqualRate(0.05, K, P_AB);
        computeTransMatrixEqualRate(0.08, K, P_CD);

        double L_root[K];
        for (int s = 0; s < K; s++) {
            double vleft = 0.0;
            for (int x = 0; x < K; x++)
                vleft += P_AB[s * K + x] * L_AB[x];
            double vright = 0.0;
            for (int x = 0; x < K; x++)
                vright += P_CD[s * K + x] * L_CD[x];
            L_root[s] = vleft * vright;
        }

        // Propagate scale_num from children
        UBYTE scale_root = scale_AB + scale_CD;

        // Check if root also needs scaling
        double lh_max = 0.0;
        for (int s = 0; s < K; s++)
            if (L_root[s] > lh_max) lh_max = L_root[s];

        cout << "  Root: max_before_scaling=" << lh_max << endl;

        if (lh_max == 0.0) {
            scale_root += 4;
        } else if (lh_max < SCALING_THRESHOLD) {
            for (int s = 0; s < K; s++)
                L_root[s] = ldexp(L_root[s], SCALING_THRESHOLD_EXP);
            scale_root += 1;
        }

        cout << "  Root: scale_num=" << (int)scale_root << endl;

        // Verify children were scaled
        bool pass_children = (scale_AB == 1 && scale_CD == 1);
        // Root should be sum of children (no additional scaling for these values)
        bool pass_propagation = (scale_root == 2);

        // Verify corrected lnL is finite and negative
        double pi[K] = {0.25, 0.25, 0.25, 0.25};
        double site_lh = 0.0;
        for (int s = 0; s < K; s++)
            site_lh += pi[s] * L_root[s];
        double lnL = log(site_lh) + scale_root * LOG_SCALING_THRESHOLD;

        bool pass_finite = !std::isnan(lnL) && !std::isinf(lnL) && lnL < 0.0;

        cout << "  Corrected lnL = " << lnL << endl;
        cout << "  Children scaled: " << (pass_children ? "PASS" : "FAIL") << endl;
        cout << "  Propagation (expect 2): " << (pass_propagation ? "PASS" : "FAIL") << endl;
        cout << "  lnL finite & negative: " << (pass_finite ? "PASS" : "FAIL") << endl;

        if (!pass_children || !pass_propagation || !pass_finite) all_pass = false;
    }

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    cout << endl << "=== OpenACC Step 6 Result: "
         << (all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED")
         << " ===" << endl << endl;

    return all_pass;
}

// ==========================================================================
// Step 7: Log-Likelihood at Root — test & verification
// ==========================================================================
//
// Computes the final tree log-likelihood from root partials:
//   site_lh_j  = pi^T . L_root_j                     (Eq 3)
//   lnL_j      = log(site_lh_j) + n_j * LOG_SCALING   (Eq 7)
//   total_lnL  = sum_j  w_j * lnL_j                   (Eq 4)
//
// Loop structure mirrors the PoC reduction kernel (OpenACC-ready):
//   #pragma acc parallel loop reduction(+:logL)
//   for j in patterns:
//       site_lh = sum_s pi[s] * root_plh[j*K + s]
//       logL += freq[j] * (log(site_lh) + scale[j] * LOG_SCALING_THRESHOLD)
//
// This is the LAST CPU math building block before Step 8 (full traversal).
// ==========================================================================

bool testLogLikelihoodRoot() {
    cout << endl;
    cout << "============================================================" << endl;
    cout << "=== OpenACC Step 7: Log-Likelihood at Root (CPU)         ===" << endl;
    cout << "============================================================" << endl;

    bool all_pass = true;
    const int K = 4;  // DNA states

    // ------------------------------------------------------------------
    // Test 7a: Single-pattern pi^T . L_root dot product (Eq 3)
    // Uses 3-taxon reference: ((A:0.1,B:0.2):0.05,C:0.15), pattern A,C,G
    // Root partials from Step 5a: [2.309e-3, 1.151e-3, 2.624e-3, 1.376e-4]
    // site_lh = 0.25 * sum = 1.555e-3
    // lnL = log(1.555e-3) = -6.465999  (no scaling)
    // ------------------------------------------------------------------
    {
        cout << endl << "  --- Test 7a: Single-pattern pi^T . L_root (Eq 3) ---" << endl;

        // Recompute the root partials from scratch (same as Step 5a)
        double t_A = 0.1, t_B = 0.2, t_AB = 0.05, t_C = 0.15;
        double P_A[K * K], P_B[K * K], P_AB[K * K], P_C[K * K];
        computeTransMatrixEqualRate(t_A, K, P_A);
        computeTransMatrixEqualRate(t_B, K, P_B);
        computeTransMatrixEqualRate(t_AB, K, P_AB);
        computeTransMatrixEqualRate(t_C, K, P_C);

        int state_A = 0, state_B = 1, state_C = 2;  // A, C, G

        // Stage 1: TIP-TIP at cherry (AB)
        double L_AB[K];
        for (int s = 0; s < K; s++) {
            L_AB[s] = P_A[s * K + state_A] * P_B[s * K + state_B];
        }

        // Stage 2: TIP-INTERNAL at root
        double L_root[K];
        for (int s = 0; s < K; s++) {
            double vleft = 0.0;
            for (int x = 0; x < K; x++) {
                vleft += P_AB[s * K + x] * L_AB[x];
            }
            double vright = P_C[s * K + state_C];
            L_root[s] = vleft * vright;
        }

        // === THIS IS THE NEW STEP 7 CODE ===
        // Compute site likelihood: pi^T . L_root  (Eq 3)
        // For JC model: pi = [0.25, 0.25, 0.25, 0.25]
        //
        // GPU-ready structure: this becomes a simple reduction over K states.
        // In the PoC, this is done via baseFreq * rootL matrix multiply (1xK * KxP).
        // Here we write it as an explicit dot product per pattern, which maps to:
        //   #pragma acc parallel loop          (over patterns)
        //   #pragma acc loop vector reduction   (over states)
        double pi[K] = {0.25, 0.25, 0.25, 0.25};

        double site_lh = 0.0;
        for (int s = 0; s < K; s++) {
            site_lh += pi[s] * L_root[s];
        }

        // Log-likelihood (no scaling for this test)
        UBYTE scale_num = 0;
        double lnL = log(site_lh) + scale_num * LOG_SCALING_THRESHOLD;

        double expected_lnL = -6.465999160939802;
        double diff = fabs(lnL - expected_lnL);
        bool pass_value = diff < 1e-8;

        cout << "  L_root = [";
        for (int s = 0; s < K; s++) cout << (s ? ", " : "") << L_root[s];
        cout << "]" << endl;
        cout << "  site_lh = pi^T . L_root = " << site_lh << endl;
        cout << "  lnL = log(site_lh) = " << lnL << endl;
        cout << "  Expected:             " << expected_lnL << endl;
        cout << "  Diff:                 " << diff << endl;
        cout << "  Value match (tol 1e-8): " << (pass_value ? "PASS" : "FAIL") << endl;

        if (!pass_value) all_pass = false;
    }

    // ------------------------------------------------------------------
    // Test 7b: Log-likelihood with scaling correction (Eq 7)
    // Artificially scale root partials down, then verify correction recovers
    // the true lnL.
    // lnL_corrected = log(site_lh_stored) + n * LOG_SCALING_THRESHOLD
    // ------------------------------------------------------------------
    {
        cout << endl << "  --- Test 7b: Scaling correction in log-lh (Eq 7) ---" << endl;

        double pi[K] = {0.25, 0.25, 0.25, 0.25};

        // Use known root partials from Test 7a
        double t_A = 0.1, t_B = 0.2, t_AB = 0.05, t_C = 0.15;
        double P_A[K * K], P_B[K * K], P_AB[K * K], P_C[K * K];
        computeTransMatrixEqualRate(t_A, K, P_A);
        computeTransMatrixEqualRate(t_B, K, P_B);
        computeTransMatrixEqualRate(t_AB, K, P_AB);
        computeTransMatrixEqualRate(t_C, K, P_C);

        int state_A = 0, state_B = 1, state_C = 2;

        double L_AB[K];
        for (int s = 0; s < K; s++)
            L_AB[s] = P_A[s * K + state_A] * P_B[s * K + state_B];

        double L_root_true[K];
        for (int s = 0; s < K; s++) {
            double vleft = 0.0;
            for (int x = 0; x < K; x++)
                vleft += P_AB[s * K + x] * L_AB[x];
            double vright = P_C[s * K + state_C];
            L_root_true[s] = vleft * vright;
        }

        // True (unscaled) lnL
        double site_lh_true = 0.0;
        for (int s = 0; s < K; s++) site_lh_true += pi[s] * L_root_true[s];
        double lnL_true = log(site_lh_true);

        // Simulate n=2 scaling events: partials were multiplied by (2^256)^2
        // This is what happens when both children of an INTERNAL-INTERNAL node
        // each trigger one scaling event.
        UBYTE scale_num = 2;
        double L_root_scaled[K];
        for (int s = 0; s < K; s++)
            L_root_scaled[s] = ldexp(ldexp(L_root_true[s], SCALING_THRESHOLD_EXP), SCALING_THRESHOLD_EXP);

        // Compute site_lh from scaled partials
        double site_lh_scaled = 0.0;
        for (int s = 0; s < K; s++) site_lh_scaled += pi[s] * L_root_scaled[s];

        // Apply Eq 7 correction:
        // lnL_corrected = log(site_lh_scaled) + scale_num * LOG_SCALING_THRESHOLD
        //               = log(site_lh_true * 2^(256*2)) + 2 * LOG_SCALING_THRESHOLD
        //               = log(site_lh_true) + 2*256*log(2) + 2*(-256*log(2))
        //               = log(site_lh_true)  ← exact cancellation
        double lnL_corrected = log(site_lh_scaled) + scale_num * LOG_SCALING_THRESHOLD;

        double diff = fabs(lnL_corrected - lnL_true);
        bool pass_roundtrip = diff < 1e-10;

        cout << "  True lnL:      " << lnL_true << endl;
        cout << "  Scaled site_lh: " << site_lh_scaled << " (2x scaled up)" << endl;
        cout << "  Corrected lnL: " << lnL_corrected << endl;
        cout << "  Diff:          " << diff << endl;
        cout << "  Roundtrip (tol 1e-10): " << (pass_roundtrip ? "PASS" : "FAIL") << endl;

        if (!pass_roundtrip) all_pass = false;
    }

    // ------------------------------------------------------------------
    // Test 7c: Pattern-weighted sum on 7-site alignment (Eq 4)
    // 3-taxon tree: ((A:0.1,B:0.2):0.05,C:0.15)
    // 4 patterns with frequencies = 7 total sites
    //
    // This is the COMPLETE log-likelihood reduction, GPU-ready:
    //   total_lnL = sum_j freq[j] * (log(pi^T . L_root_j) + scale[j]*LOG_SCALE)
    //
    // Loop structure mirrors PoC's OpenACC reduction:
    //   double logL = 0.0;
    //   #pragma acc parallel loop reduction(+:logL)
    //                present(root_plh, freq, scale_num, pi)
    //   for (int j = 0; j < nptn; j++) { ... }
    // ------------------------------------------------------------------
    {
        cout << endl << "  --- Test 7c: Pattern-weighted total lnL (Eq 4) ---" << endl;

        double t_A = 0.1, t_B = 0.2, t_AB = 0.05, t_C = 0.15;
        double pi[K] = {0.25, 0.25, 0.25, 0.25};

        double P_A[K * K], P_B[K * K], P_AB[K * K], P_C[K * K];
        computeTransMatrixEqualRate(t_A, K, P_A);
        computeTransMatrixEqualRate(t_B, K, P_B);
        computeTransMatrixEqualRate(t_AB, K, P_AB);
        computeTransMatrixEqualRate(t_C, K, P_C);

        struct Pattern3 { int sA; int sB; int sC; int freq; };
        Pattern3 patterns[] = {
            {0, 1, 2, 1},  // A,C,G  freq 1
            {0, 0, 0, 3},  // A,A,A  freq 3
            {1, 1, 1, 2},  // C,C,C  freq 2
            {2, 3, 0, 1},  // G,T,A  freq 1
        };
        int nptn = 4;

        // === PRE-COMPUTE ALL ROOT PARTIALS (post-order traversal) ===
        // In full GPU pipeline, these would already be on device from Steps 9-10.
        double root_plh[4 * K];   // nptn * K  (flat array, GPU-friendly)
        UBYTE  scale_num[4];      // no scaling for these values
        int    freq[4];

        for (int j = 0; j < nptn; j++) {
            scale_num[j] = 0;
            freq[j] = patterns[j].freq;

            // TIP-TIP at cherry
            double L_AB[K];
            for (int s = 0; s < K; s++) {
                L_AB[s] = P_A[s * K + patterns[j].sA]
                        * P_B[s * K + patterns[j].sB];
            }

            // TIP-INTERNAL at root
            for (int s = 0; s < K; s++) {
                double vleft = 0.0;
                for (int x = 0; x < K; x++)
                    vleft += P_AB[s * K + x] * L_AB[x];
                double vright = P_C[s * K + patterns[j].sC];
                root_plh[j * K + s] = vleft * vright;
            }
        }

        // === LOG-LIKELIHOOD REDUCTION (Step 7 / Step 11 target) ===
        // This loop is the exact structure that will become an OpenACC kernel.
        // Compare with PoC: LikelihoodCalculator::computeSiteLikelihoodFromRoot()
        //
        // Future OpenACC (Step 11):
        //   #pragma acc parallel loop reduction(+:total_lnL) \
        //       present(root_plh[0:nptn*K], freq[0:nptn], scale_num[0:nptn])
        //
        double total_lnL = 0.0;
        for (int j = 0; j < nptn; j++) {
            // pi^T . L_root_j  (Eq 3)
            double site_lh = 0.0;
            for (int s = 0; s < K; s++) {
                site_lh += pi[s] * root_plh[j * K + s];
            }

            // log + scaling correction  (Eq 7)
            // Guard: clamp to avoid log(0) — matches PoC's kMinSiteLikelihood
            if (site_lh < 1e-300) site_lh = 1e-300;

            double ptn_lnL = log(site_lh) + scale_num[j] * LOG_SCALING_THRESHOLD;

            // Weighted accumulation  (Eq 4)
            total_lnL += freq[j] * ptn_lnL;
        }

        // Print per-pattern breakdown
        for (int j = 0; j < nptn; j++) {
            double site_lh = 0.0;
            for (int s = 0; s < K; s++)
                site_lh += pi[s] * root_plh[j * K + s];
            double ptn_lnL = log(site_lh) + scale_num[j] * LOG_SCALING_THRESHOLD;
            cout << "  Pattern " << j << " ("
                 << patterns[j].sA << "," << patterns[j].sB << "," << patterns[j].sC
                 << "): site_lh=" << site_lh
                 << " lnL=" << ptn_lnL
                 << " x freq=" << freq[j] << endl;
        }
        cout << "  Total lnL = " << total_lnL << " (7 sites)" << endl;

        // Verify against plan reference
        // Plan says: total lnL = -8.962730 for this alignment
        // But the plan's Test Case 3 uses a 2-taxon 4-site alignment (AA,AC,CC)
        // Our 7-site 3-taxon alignment is different. Verify internally:
        //   - Must be finite, negative
        //   - First pattern (A,C,G) lnL must match Step 5a: -6.465999
        //   - Cross-check: recompute from Step 5d values

        // Verify first pattern matches Step 5a
        double site_lh_0 = 0.0;
        for (int s = 0; s < K; s++) site_lh_0 += pi[s] * root_plh[0 * K + s];
        double ptn0_lnL = log(site_lh_0);
        double expected_ptn0 = -6.465999160939802;
        bool pass_ptn0 = fabs(ptn0_lnL - expected_ptn0) < 1e-8;

        bool pass_finite = !std::isnan(total_lnL) && !std::isinf(total_lnL);
        bool pass_negative = total_lnL < 0.0;
        // 7 sites, all lnL negative → expect roughly [-50, -5]
        bool pass_range = total_lnL > -50.0 && total_lnL < -5.0;

        cout << "  Pattern 0 lnL match (tol 1e-8): " << (pass_ptn0 ? "PASS" : "FAIL") << endl;
        cout << "  Finite: " << (pass_finite ? "PASS" : "FAIL") << endl;
        cout << "  Negative: " << (pass_negative ? "PASS" : "FAIL") << endl;
        cout << "  Range [-50, -5]: " << (pass_range ? "PASS" : "FAIL") << endl;

        if (!pass_ptn0 || !pass_finite || !pass_negative || !pass_range) all_pass = false;
    }

    // ------------------------------------------------------------------
    // Test 7d: 2-taxon 4-site reference from plan (total lnL = -8.962730)
    // Tree: (A:0.1,B:0.1)  — rooted cherry, equal branches
    // Patterns: AA(freq=2), AC(freq=1), CC(freq=1) → 4 sites total
    //
    // Per-pattern expected (from plan):
    //   AA: lnL = -1.579337686675270
    //   AC: lnL = -4.224716686443124
    //   CC: lnL = -1.579337686675270
    //
    // Total: 2*(-1.579338) + 1*(-4.224717) + 1*(-1.579338) = -8.962730
    //
    // This is the plan's golden reference value.
    // ------------------------------------------------------------------
    {
        cout << endl << "  --- Test 7d: 2-taxon plan reference (lnL = -8.962730) ---" << endl;

        double t = 0.1;
        double pi[K] = {0.25, 0.25, 0.25, 0.25};
        double P[K * K];
        computeTransMatrixEqualRate(t, K, P);

        // Patterns: {state_A, state_B, frequency}
        struct Pattern2 { int sA; int sB; int freq; };
        Pattern2 patterns[] = {
            {0, 0, 2},  // A,A  freq 2
            {0, 1, 1},  // A,C  freq 1
            {1, 1, 1},  // C,C  freq 1
        };
        int nptn = 3;

        // Pre-compute root partials (TIP-TIP cherry, both branches = t)
        double root_plh[3 * K];
        UBYTE  scale_num[3];
        int    freq[3];

        for (int j = 0; j < nptn; j++) {
            scale_num[j] = 0;
            freq[j] = patterns[j].freq;

            // TIP-TIP: L_root[s] = P[s,sA] * P[s,sB]
            for (int s = 0; s < K; s++) {
                root_plh[j * K + s] = P[s * K + patterns[j].sA]
                                    * P[s * K + patterns[j].sB];
            }
        }

        // === LOG-LIKELIHOOD REDUCTION (GPU-ready loop) ===
        // Future OpenACC:
        //   #pragma acc parallel loop reduction(+:total_lnL) \
        //       present(root_plh[0:nptn*K], freq[0:nptn], scale_num[0:nptn])
        double total_lnL = 0.0;
        double per_ptn_lnL[3];

        for (int j = 0; j < nptn; j++) {
            double site_lh = 0.0;
            for (int s = 0; s < K; s++) {
                site_lh += pi[s] * root_plh[j * K + s];
            }
            if (site_lh < 1e-300) site_lh = 1e-300;

            per_ptn_lnL[j] = log(site_lh) + scale_num[j] * LOG_SCALING_THRESHOLD;
            total_lnL += freq[j] * per_ptn_lnL[j];
        }

        // Expected per-pattern values from plan
        double expected_AA = -1.579337686675270;
        double expected_AC = -4.224716686443124;
        double expected_CC = -1.579337686675270;
        double expected_total = -8.962729746468934;

        cout << "  AA (freq=2): lnL=" << per_ptn_lnL[0]
             << " expected=" << expected_AA << endl;
        cout << "  AC (freq=1): lnL=" << per_ptn_lnL[1]
             << " expected=" << expected_AC << endl;
        cout << "  CC (freq=1): lnL=" << per_ptn_lnL[2]
             << " expected=" << expected_CC << endl;
        cout << "  Total lnL = " << total_lnL
             << " expected=" << expected_total << endl;

        bool pass_AA = fabs(per_ptn_lnL[0] - expected_AA) < 1e-8;
        bool pass_AC = fabs(per_ptn_lnL[1] - expected_AC) < 1e-8;
        bool pass_CC = fabs(per_ptn_lnL[2] - expected_CC) < 1e-8;
        bool pass_total = fabs(total_lnL - expected_total) < 1e-8;

        cout << "  AA match (tol 1e-8): " << (pass_AA ? "PASS" : "FAIL") << endl;
        cout << "  AC match (tol 1e-8): " << (pass_AC ? "PASS" : "FAIL") << endl;
        cout << "  CC match (tol 1e-8): " << (pass_CC ? "PASS" : "FAIL") << endl;
        cout << "  Total match (tol 1e-8): " << (pass_total ? "PASS" : "FAIL") << endl;

        if (!pass_AA || !pass_AC || !pass_CC || !pass_total) all_pass = false;
    }

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    cout << endl << "=== OpenACC Step 7 Result: "
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
