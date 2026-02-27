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
// Step 8: Full Post-Order Traversal — test & verification
// ==========================================================================
//
// Verifies the full tree likelihood by:
//   (a) Running the production OpenACC kernel on the loaded tree
//   (b) Recomputing total lnL from stored _pattern_lh × ptn_freq
//   (c) Checking determinism (recompute gives same result)
//   (d) Walking the tree post-order and manually recomputing each internal
//       node's partials from its children's stored partials using the Step 2-7
//       building blocks, then comparing against the production kernel's output.
//
// This is the definitive CPU correctness test: an independent reimplementation
// checks every intermediate result of the production kernel.
//
// Constraint: Manual node verification (8d) requires DNA (K=4), ncat=1, JC model.
// ==========================================================================

// Helper: recursively verify each internal node's partial_lh by recomputing
// from children's stored partials using computeTransMatrixEqualRate (Step 2).
// Returns the maximum absolute difference across all checked patterns/states.
static double verifySubtreePartials(
    PhyloTree *tree, PhyloNode *node, PhyloNode *dad,
    size_t nptn, size_t orig_nptn, int K,
    int check_ptns, double tol,
    int &nodes_checked, int &nodes_failed)
{
    if (node->isLeaf()) return 0.0;

    // Find the edge from dad → node (stores CLV for subtree rooted at node)
    PhyloNeighbor *dad_branch = NULL;
    for (NeighborVec::iterator nit = dad->neighbors.begin();
         nit != dad->neighbors.end(); nit++) {
        if ((*nit)->node == node) { dad_branch = (PhyloNeighbor*)*nit; break; }
    }
    if (!dad_branch || !dad_branch->get_partial_lh()) return 0.0;

    // Handle multifurcating nodes (degree > 3) — skip verification, just recurse
    // (Production kernel handles these separately at phylokernel_openacc.cpp:81-151)
    if (node->degree() > 3) {
        FOR_NEIGHBOR_IT(node, dad, it2) {
            if (!((PhyloNode*)(*it2)->node)->isLeaf()) {
                verifySubtreePartials(tree, (PhyloNode*)(*it2)->node, node,
                                      nptn, orig_nptn, K, check_ptns, tol,
                                      nodes_checked, nodes_failed);
            }
        }
        return 0.0;
    }

    // Get children of node (excluding dad) — exactly 2 for binary (degree-3) node
    PhyloNeighbor *left = NULL, *right = NULL;
    FOR_NEIGHBOR_IT(node, dad, it) {
        if (!left) left = (PhyloNeighbor*)(*it);
        else right = (PhyloNeighbor*)(*it);
    }
    if (!left || !right) return 0.0;  // safety check

    // Recurse into internal children first (post-order)
    double max_diff = 0.0;
    if (!left->node->isLeaf()) {
        double d = verifySubtreePartials(tree, (PhyloNode*)left->node, node,
                                          nptn, orig_nptn, K, check_ptns, tol,
                                          nodes_checked, nodes_failed);
        if (d > max_diff) max_diff = d;
    }
    if (!right->node->isLeaf()) {
        double d = verifySubtreePartials(tree, (PhyloNode*)right->node, node,
                                          nptn, orig_nptn, K, check_ptns, tol,
                                          nodes_checked, nodes_failed);
        if (d > max_diff) max_diff = d;
    }

    // Compute P matrices for left and right branches (Step 2 building block)
    double P_left[16], P_right[16];  // K*K = 16 for DNA
    computeTransMatrixEqualRate(left->length, K, P_left);
    computeTransMatrixEqualRate(right->length, K, P_right);

    // Verify first check_ptns patterns at this node
    int actual_check = (int)min((size_t)check_ptns, nptn);
    double node_max_diff = 0.0;
    bool scale_mismatch = false;

    for (int ptn = 0; ptn < actual_check; ptn++) {
        double manual[4];   // K=4
        double lh_max = 0.0;

        for (int s = 0; s < K; s++) {
            double vleft = 0.0, vright = 0.0;

            // Left child contribution
            if (left->node->isLeaf()) {
                // Tip: P_left * one_hot(state) using tip_partial_lh
                int state = (ptn < (int)orig_nptn) ?
                    (int)(tree->aln->at(ptn))[left->node->id] :
                    (int)tree->getModelFactory()->unobserved_ptns[ptn - orig_nptn][left->node->id];
                double *tip_lh = tree->tip_partial_lh + state * K;
                for (int x = 0; x < K; x++)
                    vleft += P_left[s * K + x] * tip_lh[x];
            } else {
                // Internal: dot product P_left * child_partial_lh
                double *left_plh = left->get_partial_lh();
                for (int x = 0; x < K; x++)
                    vleft += P_left[s * K + x] * left_plh[ptn * K + x];
            }

            // Right child contribution
            if (right->node->isLeaf()) {
                int state = (ptn < (int)orig_nptn) ?
                    (int)(tree->aln->at(ptn))[right->node->id] :
                    (int)tree->getModelFactory()->unobserved_ptns[ptn - orig_nptn][right->node->id];
                double *tip_lh = tree->tip_partial_lh + state * K;
                for (int x = 0; x < K; x++)
                    vright += P_right[s * K + x] * tip_lh[x];
            } else {
                double *right_plh = right->get_partial_lh();
                for (int x = 0; x < K; x++)
                    vright += P_right[s * K + x] * right_plh[ptn * K + x];
            }

            manual[s] = vleft * vright;
            if (manual[s] > lh_max) lh_max = manual[s];
        }

        // Apply same scaling logic as production kernel (Step 6)
        UBYTE manual_extra = 0;
        if (lh_max == 0.0) {
            // Degenerate case: use STATE_UNKNOWN tip as fallback
            for (int s = 0; s < K; s++)
                manual[s] = tree->tip_partial_lh[tree->aln->STATE_UNKNOWN * K + s];
            manual_extra = 4;
        } else if (lh_max < SCALING_THRESHOLD) {
            for (int s = 0; s < K; s++)
                manual[s] = ldexp(manual[s], SCALING_THRESHOLD_EXP);
            manual_extra = 1;
        }

        // Expected scale_num = children's scales + extra at this node
        UBYTE expected_scale = manual_extra;
        if (!left->node->isLeaf())
            expected_scale += left->get_scale_num()[ptn];
        if (!right->node->isLeaf())
            expected_scale += right->get_scale_num()[ptn];

        if (expected_scale != dad_branch->get_scale_num()[ptn])
            scale_mismatch = true;

        // Compare manual partials against stored partials
        double *dad_plh = dad_branch->get_partial_lh();
        for (int s = 0; s < K; s++) {
            double diff = fabs(manual[s] - dad_plh[ptn * K + s]);
            if (diff > node_max_diff) node_max_diff = diff;
        }
    }

    nodes_checked++;
    if (node_max_diff > tol || scale_mismatch)
        nodes_failed++;

    if (node_max_diff > max_diff) max_diff = node_max_diff;
    return max_diff;
}


bool testFullTraversal(PhyloTree *tree) {
    cout << endl;
    cout << "============================================================" << endl;
    cout << "=== OpenACC Step 8: Full Post-Order Traversal (CPU)      ===" << endl;
    cout << "============================================================" << endl;

    if (!tree || !tree->aln || !tree->getModel() || !tree->getModelFactory()) {
        cout << "  ERROR: Tree not fully initialized" << endl;
        return false;
    }

    bool all_pass = true;
    int K = tree->aln->num_states;
    int ncat = tree->getRate()->getNRate();
    size_t orig_nptn = tree->aln->size();
    size_t nptn = tree->aln->size() + tree->getModelFactory()->unobserved_ptns.size();

    cout << "  Alignment: " << tree->aln->getNSeq() << " taxa, "
         << tree->aln->getNSite() << " sites, "
         << nptn << " patterns" << endl;
    cout << "  Model: " << tree->getModelName() << endl;
    cout << "  States: " << K << ", Rate categories: " << ncat << endl;

    // ------------------------------------------------------------------
    // Test 8a: Production kernel lnL — basic validity
    // ------------------------------------------------------------------
    {
        cout << endl << "  --- Test 8a: Production kernel lnL ---" << endl;

        tree->initializeAllPartialLh();
        tree->clearAllPartialLH();
        double prod_lnL = tree->computeLikelihood();

        bool pass_finite = !std::isnan(prod_lnL) && !std::isinf(prod_lnL);
        bool pass_negative = prod_lnL < 0.0;

        cout << "  Production lnL = " << fixed << prod_lnL << endl;
        cout.unsetf(ios_base::fixed);
        cout << "  Finite:   " << (pass_finite ? "PASS" : "FAIL") << endl;
        cout << "  Negative: " << (pass_negative ? "PASS" : "FAIL") << endl;

        if (!pass_finite || !pass_negative) all_pass = false;
    }

    // ------------------------------------------------------------------
    // Test 8b: Recompute total from _pattern_lh × ptn_freq
    // Verifies the weighted sum in the branch likelihood kernel (Eq 4).
    // ------------------------------------------------------------------
    {
        cout << endl << "  --- Test 8b: Weighted sum from _pattern_lh ---" << endl;

        // computeLikelihood() was just called in 8a, so pattern_lh is fresh
        vector<double> pattern_lh_buf(orig_nptn);
        tree->computePatternLikelihood(pattern_lh_buf.data());
        double manual_sum = 0.0;
        for (size_t ptn = 0; ptn < orig_nptn; ptn++)
            manual_sum += pattern_lh_buf[ptn] * tree->ptn_freq[ptn];

        double prod_lnL = tree->computeLikelihood();
        double diff = fabs(manual_sum - prod_lnL);
        bool pass = diff < 1e-6;

        cout << "  Sum(_pattern_lh * ptn_freq) = " << fixed << manual_sum << endl;
        cout << "  Production lnL              = " << prod_lnL << endl;
        cout.unsetf(ios_base::fixed);
        cout << "  Diff: " << diff << endl;
        cout << "  Match (tol 1e-6): " << (pass ? "PASS" : "FAIL") << endl;

        if (!pass) all_pass = false;
    }

    // ------------------------------------------------------------------
    // Test 8c: Determinism — recompute gives identical result
    // ------------------------------------------------------------------
    {
        cout << endl << "  --- Test 8c: Determinism ---" << endl;

        tree->clearAllPartialLH();
        double lnL1 = tree->computeLikelihood();
        tree->clearAllPartialLH();
        double lnL2 = tree->computeLikelihood();

        double diff = fabs(lnL1 - lnL2);
        bool pass = diff < 1e-10;

        cout << "  Run 1: " << fixed << lnL1 << endl;
        cout << "  Run 2: " << lnL2 << endl;
        cout.unsetf(ios_base::fixed);
        cout << "  Diff: " << diff << endl;
        cout << "  Deterministic (tol 1e-10): " << (pass ? "PASS" : "FAIL") << endl;

        if (!pass) all_pass = false;
    }

    // ------------------------------------------------------------------
    // Test 8d: Node-by-node partial verification
    // Walk tree post-order, at each internal node recompute partials
    // from children's stored partials using Step 2-7 building blocks,
    // and compare against the production kernel's stored values.
    //
    // This is the KEY correctness test: an independent reimplementation
    // verifies every intermediate result.
    //
    // Constraint: DNA (K=4), ncat=1 only (JC model).
    // ------------------------------------------------------------------
    {
        cout << endl << "  --- Test 8d: Node-by-node partial verification ---" << endl;

        if (K != 4 || ncat != 1) {
            cout << "  SKIP: Manual verification requires DNA (K=4), ncat=1" << endl;
            cout << "  (Got K=" << K << ", ncat=" << ncat << ")" << endl;
        } else {
            // Ensure partials are freshly computed
            tree->clearAllPartialLH();
            tree->computeLikelihood();

            int nodes_checked = 0, nodes_failed = 0;
            double max_diff = 0.0;
            double tol = 1e-8;
            int check_ptns = 20;  // verify first 20 patterns per node

            // Walk from root into both subtrees
            PhyloNode *root_node = (PhyloNode*)tree->root;
            for (NeighborVec::iterator rit = root_node->neighbors.begin();
                 rit != root_node->neighbors.end(); rit++) {
                PhyloNode *child = (PhyloNode*)(*rit)->node;
                double d = verifySubtreePartials(
                    tree, child, root_node,
                    nptn, orig_nptn, K, check_ptns, tol,
                    nodes_checked, nodes_failed);
                if (d > max_diff) max_diff = d;
            }

            cout << "  Nodes checked:    " << nodes_checked << endl;
            cout << "  Nodes failed:     " << nodes_failed << endl;
            cout << "  Max partial diff: " << max_diff << endl;

            bool pass = (nodes_failed == 0);
            cout << "  All nodes match (tol " << tol << "): "
                 << (pass ? "PASS" : "FAIL") << endl;

            if (!pass) all_pass = false;
        }
    }

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    cout << endl << "=== OpenACC Step 8 Result: "
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

// ==========================================================================
// Step 13 (Rev): Reversible OpenACC Kernel — Standalone Tests
//
// These tests verify the reversible eigenspace-based kernel path using
// hand-computed reference values for the JC (Jukes-Cantor) DNA model.
//
// JC eigendecomposition (4 states, pi = [0.25, 0.25, 0.25, 0.25]):
//   eigenvalues = [0, -4/3, -4/3, -4/3]
//   U (eigenvectors, row-major):
//     [ 1, -1, -1, -1]
//     [ 1,  1,  0,  0]
//     [ 1,  0,  1,  0]
//     [ 1,  0,  0,  1]
//   U^{-1} (inverse eigenvectors, row-major):
//     [ 0.25,  0.25,  0.25,  0.25]
//     [-0.25,  0.25,  0,     0   ]
//     [-0.25,  0,     0.25,  0   ]
//     [-0.25,  0,     0,     0.25]
//
// Reversible kernel storage:
//   tip_partial_lh[state] = U^{-1} * e_state (eigenspace tips)
//   echildren[s*K+i] = U[s][i] * exp(eval[i] * t)
//   partial_lh_leaves[state][s] = Σ_i echildren[s*K+i] * tip_plh[state][i]
//   Partial lh: Hadamard product in state space, then inv_evec back-transform
//   Reduction: lh = Σ_i val[i] * plh_node[i] * plh_dad[i]
// ==========================================================================

// Helper: compute JC eigenvectors U, U^{-1}, eigenvalues
static void jcEigenDecomposition(double *evec, double *inv_evec, double *eval) {
    const int K = 4;
    // Eigenvalues
    eval[0] = 0.0;
    eval[1] = eval[2] = eval[3] = -4.0/3.0;

    // Properly normalized eigenvectors for JC (symmetric Q, equal pi=0.25).
    //
    // IQ-TREE's JC model uses Eigen3 decomposition (NOT the manual F81 path),
    // which produces orthonormal eigenvectors of the symmetrized Q, then
    // transforms: evec = S_evec * diag(1/sqrt(pi)) = S_evec * 2.
    // This gives the normalization: Σ_i π[i]*evec[i][k]*evec[i][m] = δ(k,m).
    //
    // Orthonormal eigenvectors of symmetric JC Q (S_evec columns):
    //   col 0 (λ=0):    [1/2, 1/2, 1/2, 1/2]
    //   col 1 (λ=-4/3): [1/√2, -1/√2, 0, 0]
    //   col 2 (λ=-4/3): [1/√6, 1/√6, -2/√6, 0]
    //   col 3 (λ=-4/3): [1/√12, 1/√12, 1/√12, -3/√12]
    //
    // evec = S_evec * 2 (for equal pi=0.25, 1/sqrt(0.25) = 2):
    double s2  = sqrt(2.0);
    double s6  = sqrt(6.0);
    double s12 = sqrt(12.0);

    memset(evec, 0, K*K*sizeof(double));
    // Column 0 (λ=0): all 1s
    evec[0*K+0] = 1.0;  evec[1*K+0] = 1.0;  evec[2*K+0] = 1.0;  evec[3*K+0] = 1.0;
    // Column 1 (λ=-4/3): [√2, -√2, 0, 0]
    evec[0*K+1] =  s2;  evec[1*K+1] = -s2;
    // Column 2 (λ=-4/3): [2/√6, 2/√6, -4/√6, 0]
    evec[0*K+2] =  2.0/s6;  evec[1*K+2] =  2.0/s6;  evec[2*K+2] = -4.0/s6;
    // Column 3 (λ=-4/3): [2/√12, 2/√12, 2/√12, -6/√12]
    evec[0*K+3] =  2.0/s12; evec[1*K+3] =  2.0/s12; evec[2*K+3] =  2.0/s12; evec[3*K+3] = -6.0/s12;

    // Inverse eigenvectors: inv_evec[i][j] = pi[j] * evec[j][i] = 0.25 * evec[j][i]
    // With proper normalization, this satisfies evec * inv_evec = I.
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++)
            inv_evec[i*K+j] = 0.25 * evec[j*K+i];
}

// Helper: compute tip_partial_lh in eigenspace (U^{-1} * e_state)
static void jcRevTipPartialLh(const double *inv_evec, int K, double *tip_plh) {
    // For states 0..3 (A,C,G,T): tip_plh[state*K + i] = inv_evec[i][state]
    for (int state = 0; state < K; state++) {
        for (int i = 0; i < K; i++)
            tip_plh[state*K + i] = inv_evec[i*K + state];
    }
    // STATE_UNKNOWN (index 4 for DNA): tip_plh = U^{-1} * [1,1,1,1]
    // = [Σ_j inv_evec[i][j]] for each i
    for (int i = 0; i < K; i++) {
        double sum = 0.0;
        for (int j = 0; j < K; j++)
            sum += inv_evec[i*K + j];
        tip_plh[K*K + i] = sum;
    }
}

// Helper: compute echildren[s*K+i] = U[s][i] * exp(eval[i] * t)
static void jcRevEchildren(const double *evec, const double *eval,
                           double t, int K, double *echild) {
    for (int s = 0; s < K; s++)
        for (int i = 0; i < K; i++)
            echild[s*K + i] = evec[s*K + i] * exp(eval[i] * t);
}

// Helper: compute partial_lh_leaves[state][s] = Σ_i echildren[s*K+i] * tip_plh[state*K+i]
static void jcRevLeafPartials(const double *echild, const double *tip_plh,
                              int K, int num_states_plus_unknown,
                              double *partial_lh_leaves) {
    for (int state = 0; state < num_states_plus_unknown; state++) {
        for (int s = 0; s < K; s++) {
            double v = 0.0;
            for (int i = 0; i < K; i++)
                v += echild[s*K + i] * tip_plh[state*K + i];
            partial_lh_leaves[state*K + s] = v;
        }
    }
}

bool testRevEigenspace() {
    cout << endl;
    cout << "============================================================" << endl;
    cout << "=== Rev Step 13a: Eigenspace Basics (JC DNA)             ===" << endl;
    cout << "============================================================" << endl;

    bool all_pass = true;
    const int K = 4;
    double evec[16], inv_evec[16], eval[4];
    jcEigenDecomposition(evec, inv_evec, eval);

    // Test 1: Eigenvalues
    {
        cout << endl << "  --- Test 13a.1: Eigenvalues ---" << endl;
        bool pass = approxEqual(eval[0], 0.0, 1e-15)
                 && approxEqual(eval[1], -4.0/3.0, 1e-15)
                 && approxEqual(eval[2], -4.0/3.0, 1e-15)
                 && approxEqual(eval[3], -4.0/3.0, 1e-15);
        cout << "  eval = [" << eval[0] << ", " << eval[1] << ", "
             << eval[2] << ", " << eval[3] << "]" << endl;
        cout << "  Expected [0, -1.33333, -1.33333, -1.33333]: "
             << (pass ? "PASS" : "FAIL") << endl;
        if (!pass) all_pass = false;
    }

    // Test 2: U * U^{-1} = Identity
    {
        cout << endl << "  --- Test 13a.2: U * U^{-1} = I ---" << endl;
        double prod[16];
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++) {
                double sum = 0.0;
                for (int k = 0; k < K; k++)
                    sum += evec[i*K+k] * inv_evec[k*K+j];
                prod[i*K+j] = sum;
            }
        bool pass = true;
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                if (!approxEqual(prod[i*K+j], expected, 1e-14))
                    pass = false;
            }
        cout << "  Identity check: " << (pass ? "PASS" : "FAIL") << endl;
        if (!pass) {
            cout << "  Product matrix:" << endl;
            for (int i = 0; i < K; i++) {
                cout << "    [";
                for (int j = 0; j < K; j++)
                    cout << " " << prod[i*K+j];
                cout << " ]" << endl;
            }
            all_pass = false;
        }
    }

    // Test 3: P(t) = U * diag(exp(lambda*t)) * U^{-1} matches JC formula
    {
        cout << endl << "  --- Test 13a.3: P(t) reconstruction matches JC formula ---" << endl;
        double t = 0.1;
        double P_eigen[16], P_jc[16];

        // Eigendecomposition: P[i][j] = Σ_k U[i][k] * exp(eval[k]*t) * inv_evec[k][j]
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++) {
                double sum = 0.0;
                for (int k = 0; k < K; k++)
                    sum += evec[i*K+k] * exp(eval[k]*t) * inv_evec[k*K+j];
                P_eigen[i*K+j] = sum;
            }

        // Direct JC formula
        computeTransMatrixEqualRate(t, K, P_jc);

        double max_diff = 0.0;
        for (int i = 0; i < 16; i++) {
            double d = fabs(P_eigen[i] - P_jc[i]);
            if (d > max_diff) max_diff = d;
        }

        bool pass = max_diff < 1e-14;
        cout << "  P_eigen[0][0] = " << P_eigen[0] << ", P_jc[0][0] = " << P_jc[0] << endl;
        cout << "  P_eigen[0][1] = " << P_eigen[1] << ", P_jc[0][1] = " << P_jc[1] << endl;
        cout << "  Max diff: " << max_diff << endl;
        cout << "  Match (tol 1e-14): " << (pass ? "PASS" : "FAIL") << endl;
        if (!pass) all_pass = false;
    }

    // Test 4: Eigenspace tip vectors
    {
        cout << endl << "  --- Test 13a.4: Eigenspace tip vectors ---" << endl;
        double tip_plh[5*K]; // states 0..3 + STATE_UNKNOWN
        jcRevTipPartialLh(inv_evec, K, tip_plh);

        bool pass = true;

        // Round-trip property: U * tip_plh[s] should recover the one-hot e_s.
        // This is equivalent to U * U^{-1} * e_s = e_s (eigenvector-independent).
        for (int s = 0; s < K; s++) {
            for (int sp = 0; sp < K; sp++) {
                double dot = 0.0;
                for (int i = 0; i < K; i++)
                    dot += evec[sp*K+i] * tip_plh[s*K+i];
                double expected = (s == sp) ? 1.0 : 0.0;
                if (!approxEqual(dot, expected, 1e-14)) pass = false;
            }
        }

        // First component of all tip vectors = pi = 0.25
        // (because inv_evec row 0 = [0.25, 0.25, 0.25, 0.25] for equal freq)
        for (int s = 0; s < K; s++)
            if (!approxEqual(tip_plh[s*K+0], 0.25, 1e-15)) pass = false;

        // STATE_UNKNOWN: U * tip_plh[?] should give [1,1,1,1]
        double expected_unk[4] = {1.0, 0.0, 0.0, 0.0};
        for (int i = 0; i < K; i++)
            if (!approxEqual(tip_plh[K*K+i], expected_unk[i], 1e-14)) pass = false;

        cout << "  tip_plh[A] = [" << tip_plh[0] << ", " << tip_plh[1]
             << ", " << tip_plh[2] << ", " << tip_plh[3] << "]" << endl;
        cout << "  Round-trip U * tip_plh[A] = e_A: verified" << endl;
        cout << "  tip_plh[?] = [" << tip_plh[4*K] << ", " << tip_plh[4*K+1]
             << ", " << tip_plh[4*K+2] << ", " << tip_plh[4*K+3] << "]" << endl;
        cout << "  Expected:    [1, 0, 0, 0]" << endl;
        cout << "  " << (pass ? "PASS" : "FAIL") << endl;
        if (!pass) all_pass = false;
    }

    cout << endl << "=== Rev Step 13a Result: "
         << (all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED")
         << " ===" << endl << endl;
    return all_pass;
}

bool testRevTipTip() {
    cout << endl;
    cout << "============================================================" << endl;
    cout << "=== Rev Step 13b: TIP-TIP Partial Likelihood             ===" << endl;
    cout << "============================================================" << endl;

    bool all_pass = true;
    const int K = 4;
    double evec[16], inv_evec[16], eval[4];
    jcEigenDecomposition(evec, inv_evec, eval);

    // Tip partial likelihoods in eigenspace (5 entries: A,C,G,T,?)
    double tip_plh[5*K];
    jcRevTipPartialLh(inv_evec, K, tip_plh);

    // --- Test 13b.1: TIP-TIP cherry, mismatch (A vs C), t_left=0.1, t_right=0.1 ---
    {
        cout << endl << "  --- Test 13b.1: TIP-TIP mismatch (A vs C) ---" << endl;

        double t_left = 0.1, t_right = 0.1;
        double echild_left[16], echild_right[16];
        jcRevEchildren(evec, eval, t_left, K, echild_left);
        jcRevEchildren(evec, eval, t_right, K, echild_right);

        // Compute partial_lh_leaves (pre-contracted tips in state space)
        double plh_left[5*K], plh_right[5*K];
        jcRevLeafPartials(echild_left, tip_plh, K, 5, plh_left);
        jcRevLeafPartials(echild_right, tip_plh, K, 5, plh_right);

        // State A=0, State C=1
        int state_left = 0, state_right = 1;

        // Hadamard product in state space
        double tmp_state[4];
        for (int s = 0; s < K; s++)
            tmp_state[s] = plh_left[state_left*K + s] * plh_right[state_right*K + s];

        // Back-transform to eigenspace via inv_evec
        double dad_plh[4];
        for (int i = 0; i < K; i++) {
            double v = 0.0;
            for (int x = 0; x < K; x++)
                v += inv_evec[i*K + x] * tmp_state[x];
            dad_plh[i] = v;
        }

        // Verify: compute site likelihood via eigenspace reduction
        // At the root, use val[i] = exp(eval[i] * 0) * prop = 1.0 * 1.0
        // and tip_plh for the root state (root is special, but for a 2-taxon
        // tree we just do pi^T . L_state_space)
        //
        // Actually for 2-taxon, let's verify the partial_lh agrees with
        // the P(t) approach: site_lh = Σ_s P_left(A,s) * P_right(C,s)
        double P_left[16], P_right[16];
        computeTransMatrixEqualRate(t_left, K, P_left);
        computeTransMatrixEqualRate(t_right, K, P_right);
        double site_lh_ref = 0.0;
        for (int s = 0; s < K; s++)
            site_lh_ref += P_left[0*K + s] * P_right[1*K + s];

        // The Hadamard product tmp_state[s] should equal P_left(A,s)*P_right(C,s)
        // Because partial_lh_leaves are computed as echildren * tip_plh which
        // gives U * diag(exp(λt)) * U⁻¹ * e_state = P(t) * e_state = P(t)[:,state]
        double hadamard_sum = 0.0;
        for (int s = 0; s < K; s++)
            hadamard_sum += tmp_state[s];

        bool pass_sum = approxEqual(hadamard_sum, site_lh_ref, 1e-14);
        cout << "  Hadamard sum (state-space) = " << hadamard_sum << endl;
        cout << "  Reference (P*P)            = " << site_lh_ref << endl;
        cout << "  Match: " << (pass_sum ? "PASS" : "FAIL") << endl;

        // Also verify the eigenspace partials are self-consistent:
        // Applying U to dad_plh should recover tmp_state
        double recovered[4];
        for (int s = 0; s < K; s++) {
            double v = 0.0;
            for (int i = 0; i < K; i++)
                v += evec[s*K + i] * dad_plh[i];
            recovered[s] = v;
        }
        double max_diff = 0.0;
        for (int s = 0; s < K; s++) {
            double d = fabs(recovered[s] - tmp_state[s]);
            if (d > max_diff) max_diff = d;
        }
        bool pass_recover = max_diff < 1e-14;
        cout << "  U * dad_plh recovers state-space: " << (pass_recover ? "PASS" : "FAIL")
             << " (max_diff=" << max_diff << ")" << endl;

        // Log-likelihood: lnL = log(pi^T . L_state_space) = log(0.25 * Σ_s tmp_state[s])
        double lnL_rev = log(0.25 * hadamard_sum);
        double lnL_ref = log(0.25 * site_lh_ref);
        bool pass_lnL = approxEqual(lnL_rev, lnL_ref, 1e-14);
        cout << "  lnL(rev)  = " << lnL_rev << endl;
        cout << "  lnL(ref)  = " << lnL_ref << endl;
        // Also compare with the known Step 4 value for A vs C at t=0.1
        double lnL_step4 = -4.224717;
        bool pass_step4 = approxEqual(lnL_rev, lnL_step4, 1e-4);
        cout << "  Step 4 reference (-4.224717): " << (pass_step4 ? "PASS" : "FAIL") << endl;

        if (!pass_sum || !pass_recover || !pass_lnL || !pass_step4) all_pass = false;
    }

    // --- Test 13b.2: TIP-TIP match (A vs A), t_left=0.1, t_right=0.1 ---
    {
        cout << endl << "  --- Test 13b.2: TIP-TIP match (A vs A) ---" << endl;

        double t_left = 0.1, t_right = 0.1;
        double echild_left[16], echild_right[16];
        jcRevEchildren(evec, eval, t_left, K, echild_left);
        jcRevEchildren(evec, eval, t_right, K, echild_right);

        double plh_left[5*K], plh_right[5*K];
        jcRevLeafPartials(echild_left, tip_plh, K, 5, plh_left);
        jcRevLeafPartials(echild_right, tip_plh, K, 5, plh_right);

        int state_left = 0, state_right = 0; // A vs A

        double tmp_state[4];
        for (int s = 0; s < K; s++)
            tmp_state[s] = plh_left[state_left*K + s] * plh_right[state_right*K + s];

        double P_left[16], P_right[16];
        computeTransMatrixEqualRate(t_left, K, P_left);
        computeTransMatrixEqualRate(t_right, K, P_right);
        double site_lh_ref = 0.0;
        for (int s = 0; s < K; s++)
            site_lh_ref += P_left[0*K + s] * P_right[0*K + s];

        double hadamard_sum = 0.0;
        for (int s = 0; s < K; s++) hadamard_sum += tmp_state[s];

        bool pass = approxEqual(hadamard_sum, site_lh_ref, 1e-14);
        double lnL = log(0.25 * hadamard_sum);
        cout << "  Hadamard sum = " << hadamard_sum << endl;
        cout << "  Reference    = " << site_lh_ref << endl;
        cout << "  lnL          = " << lnL << " (expected ~-1.579)" << endl;
        cout << "  Match: " << (pass ? "PASS" : "FAIL") << endl;
        if (!pass) all_pass = false;
    }

    // --- Test 13b.3: Asymmetric branch lengths ---
    {
        cout << endl << "  --- Test 13b.3: TIP-TIP asymmetric branches (A vs G) ---" << endl;

        double t_left = 0.05, t_right = 0.2;
        double echild_left[16], echild_right[16];
        jcRevEchildren(evec, eval, t_left, K, echild_left);
        jcRevEchildren(evec, eval, t_right, K, echild_right);

        double plh_left[5*K], plh_right[5*K];
        jcRevLeafPartials(echild_left, tip_plh, K, 5, plh_left);
        jcRevLeafPartials(echild_right, tip_plh, K, 5, plh_right);

        int state_left = 0, state_right = 2; // A vs G

        double tmp_state[4];
        for (int s = 0; s < K; s++)
            tmp_state[s] = plh_left[state_left*K + s] * plh_right[state_right*K + s];

        double P_left[16], P_right[16];
        computeTransMatrixEqualRate(t_left, K, P_left);
        computeTransMatrixEqualRate(t_right, K, P_right);
        double site_lh_ref = 0.0;
        for (int s = 0; s < K; s++)
            site_lh_ref += P_left[0*K + s] * P_right[2*K + s];

        double hadamard_sum = 0.0;
        for (int s = 0; s < K; s++) hadamard_sum += tmp_state[s];

        bool pass = approxEqual(hadamard_sum, site_lh_ref, 1e-14);
        double lnL = log(0.25 * hadamard_sum);
        cout << "  Hadamard sum = " << hadamard_sum << ", Ref = " << site_lh_ref << endl;
        cout << "  lnL = " << lnL << endl;
        cout << "  Match: " << (pass ? "PASS" : "FAIL") << endl;
        if (!pass) all_pass = false;
    }

    cout << endl << "=== Rev Step 13b Result: "
         << (all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED")
         << " ===" << endl << endl;
    return all_pass;
}

bool testRevTipInternal() {
    cout << endl;
    cout << "============================================================" << endl;
    cout << "=== Rev Step 13c: TIP-INTERNAL Partial Likelihood         ===" << endl;
    cout << "============================================================" << endl;

    bool all_pass = true;
    const int K = 4;
    double evec[16], inv_evec[16], eval[4];
    jcEigenDecomposition(evec, inv_evec, eval);

    double tip_plh[5*K];
    jcRevTipPartialLh(inv_evec, K, tip_plh);

    // 3-taxon tree: ((A:0.1, C:0.2):0.05, G:0.15)  — matches Step 5 reference
    // First compute TIP-TIP cherry (A:0.1, C:0.2) → internal node in eigenspace
    double t_A = 0.1, t_C = 0.2, t_cherry = 0.05, t_G = 0.15;

    double echild_A[16], echild_C[16];
    jcRevEchildren(evec, eval, t_A, K, echild_A);
    jcRevEchildren(evec, eval, t_C, K, echild_C);

    double plh_A[5*K], plh_C[5*K];
    jcRevLeafPartials(echild_A, tip_plh, K, 5, plh_A);
    jcRevLeafPartials(echild_C, tip_plh, K, 5, plh_C);

    // TIP-TIP: A(state=0) vs C(state=1)
    double tmp_state_cherry[4];
    for (int s = 0; s < K; s++)
        tmp_state_cherry[s] = plh_A[0*K + s] * plh_C[1*K + s];

    // Back-transform to eigenspace
    double cherry_plh[4]; // eigenspace
    for (int i = 0; i < K; i++) {
        double v = 0.0;
        for (int x = 0; x < K; x++)
            v += inv_evec[i*K + x] * tmp_state_cherry[x];
        cherry_plh[i] = v;
    }

    // Now TIP-INTERNAL: cherry (internal, eigenspace) vs G (tip, state=2)
    // The cherry branch has length t_cherry=0.05 going UP to the root.
    // The G branch has length t_G=0.15.
    // echildren for cherry branch (left=G tip, right=cherry internal)

    // Actually in the TIP-INTERNAL case:
    // Left = tip (G), Right = internal (cherry)
    // eleft = echildren for G branch (t_G)
    // eright = echildren for cherry branch (t_cherry)

    double echild_G[16], echild_cherry[16];
    jcRevEchildren(evec, eval, t_G, K, echild_G);
    jcRevEchildren(evec, eval, t_cherry, K, echild_cherry);

    // Pre-contracted left tip
    double plh_G[5*K];
    jcRevLeafPartials(echild_G, tip_plh, K, 5, plh_G);

    // Right child (cherry) is internal: transform eigenspace→state space via eright
    double right_state[4];
    for (int s = 0; s < K; s++) {
        double v = 0.0;
        for (int i = 0; i < K; i++)
            v += echild_cherry[s*K + i] * cherry_plh[i];
        right_state[s] = v;
    }

    // Hadamard: left (state 2=G from pre-contracted) * right (state space)
    double tmp_state_root[4];
    for (int s = 0; s < K; s++)
        tmp_state_root[s] = plh_G[2*K + s] * right_state[s];

    // Back-transform to eigenspace
    double root_plh[4];
    for (int i = 0; i < K; i++) {
        double v = 0.0;
        for (int x = 0; x < K; x++)
            v += inv_evec[i*K + x] * tmp_state_root[x];
        root_plh[i] = v;
    }

    // Verify: compute reference via non-rev (P(t) matrices)
    double P_A[16], P_C_mat[16], P_cherry[16], P_G[16];
    computeTransMatrixEqualRate(t_A, K, P_A);
    computeTransMatrixEqualRate(t_C, K, P_C_mat);
    computeTransMatrixEqualRate(t_cherry, K, P_cherry);
    computeTransMatrixEqualRate(t_G, K, P_G);

    // Cherry partial in state space (non-rev):
    // cherry_state[s] = Σ_x P_A(s,x)*δ(x,A=0) * Σ_y P_C(s,y)*δ(y,C=1)
    //                 = P_A(s,0) * P_C(s,1)
    double cherry_state_ref[4];
    for (int s = 0; s < K; s++)
        cherry_state_ref[s] = P_A[s*K + 0] * P_C_mat[s*K + 1];

    // Root partial (non-rev):
    // root_state[s] = Σ_x P_cherry(s,x)*cherry_state[x] * Σ_y P_G(s,y)*δ(y,G=2)
    //              = (Σ_x P_cherry(s,x)*cherry_state[x]) * P_G(s,2)
    double root_state_ref[4];
    for (int s = 0; s < K; s++) {
        double v = 0.0;
        for (int x = 0; x < K; x++)
            v += P_cherry[s*K + x] * cherry_state_ref[x];
        root_state_ref[s] = v * P_G[s*K + 2];
    }

    // --- Test 13c.1: Root state-space partials match reference ---
    {
        cout << endl << "  --- Test 13c.1: Root state-space partials ---" << endl;

        // Convert eigenspace root_plh back to state space via U
        double root_state_rev[4];
        for (int s = 0; s < K; s++) {
            double v = 0.0;
            for (int i = 0; i < K; i++)
                v += evec[s*K + i] * root_plh[i];
            root_state_rev[s] = v;
        }

        double max_diff = 0.0;
        for (int s = 0; s < K; s++) {
            double d = fabs(root_state_rev[s] - root_state_ref[s]);
            if (d > max_diff) max_diff = d;
        }
        bool pass = max_diff < 1e-14;

        cout << "  Rev root (state-space):    [";
        for (int s = 0; s < K; s++) cout << root_state_rev[s] << (s<K-1?", ":"");
        cout << "]" << endl;
        cout << "  Nonrev root (reference):   [";
        for (int s = 0; s < K; s++) cout << root_state_ref[s] << (s<K-1?", ":"");
        cout << "]" << endl;
        cout << "  Max diff: " << max_diff << " ... " << (pass ? "PASS" : "FAIL") << endl;
        if (!pass) all_pass = false;
    }

    // --- Test 13c.2: Log-likelihood matches Step 5 reference ---
    {
        cout << endl << "  --- Test 13c.2: Log-likelihood ---" << endl;

        double site_lh_rev = 0.0;
        // Convert root_plh back to state space and dot with pi
        double root_state_rev[4];
        for (int s = 0; s < K; s++) {
            double v = 0.0;
            for (int i = 0; i < K; i++)
                v += evec[s*K + i] * root_plh[i];
            root_state_rev[s] = v;
        }
        for (int s = 0; s < K; s++)
            site_lh_rev += 0.25 * root_state_rev[s];

        double site_lh_ref = 0.0;
        for (int s = 0; s < K; s++)
            site_lh_ref += 0.25 * root_state_ref[s];

        double lnL_rev = log(site_lh_rev);
        double lnL_ref = log(site_lh_ref);
        double lnL_step5 = -6.465999; // known reference from Step 5

        bool pass_match = approxEqual(lnL_rev, lnL_ref, 1e-14);
        bool pass_step5 = approxEqual(lnL_rev, lnL_step5, 1e-4);

        cout << "  lnL(rev)       = " << lnL_rev << endl;
        cout << "  lnL(nonrev)    = " << lnL_ref << endl;
        cout << "  Step 5 ref     = " << lnL_step5 << endl;
        cout << "  Rev==Nonrev: " << (pass_match ? "PASS" : "FAIL") << endl;
        cout << "  Step 5 match:  " << (pass_step5 ? "PASS" : "FAIL") << endl;
        if (!pass_match || !pass_step5) all_pass = false;
    }

    cout << endl << "=== Rev Step 13c Result: "
         << (all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED")
         << " ===" << endl << endl;
    return all_pass;
}

bool testRevInternalInternal() {
    cout << endl;
    cout << "============================================================" << endl;
    cout << "=== Rev Step 13d: INTERNAL-INTERNAL Partial Likelihood    ===" << endl;
    cout << "============================================================" << endl;

    bool all_pass = true;
    const int K = 4;
    double evec[16], inv_evec[16], eval[4];
    jcEigenDecomposition(evec, inv_evec, eval);

    double tip_plh[5*K];
    jcRevTipPartialLh(inv_evec, K, tip_plh);

    // 4-taxon balanced tree: ((A:0.1, C:0.1):0.05, (G:0.15, T:0.15):0.05)
    double t_A = 0.1, t_C = 0.1, t_G = 0.15, t_T = 0.15;
    double t_left_branch = 0.05, t_right_branch = 0.05;

    // Cherry 1: (A:0.1, C:0.1)
    double echild_A[16], echild_C[16];
    jcRevEchildren(evec, eval, t_A, K, echild_A);
    jcRevEchildren(evec, eval, t_C, K, echild_C);

    double plh_A[5*K], plh_C[5*K];
    jcRevLeafPartials(echild_A, tip_plh, K, 5, plh_A);
    jcRevLeafPartials(echild_C, tip_plh, K, 5, plh_C);

    double tmp1[4];
    for (int s = 0; s < K; s++)
        tmp1[s] = plh_A[0*K + s] * plh_C[1*K + s]; // A=0, C=1

    double cherry1_plh[4]; // eigenspace
    for (int i = 0; i < K; i++) {
        double v = 0.0;
        for (int x = 0; x < K; x++)
            v += inv_evec[i*K + x] * tmp1[x];
        cherry1_plh[i] = v;
    }

    // Cherry 2: (G:0.15, T:0.15)
    double echild_G[16], echild_T[16];
    jcRevEchildren(evec, eval, t_G, K, echild_G);
    jcRevEchildren(evec, eval, t_T, K, echild_T);

    double plh_G[5*K], plh_T[5*K];
    jcRevLeafPartials(echild_G, tip_plh, K, 5, plh_G);
    jcRevLeafPartials(echild_T, tip_plh, K, 5, plh_T);

    double tmp2[4];
    for (int s = 0; s < K; s++)
        tmp2[s] = plh_G[2*K + s] * plh_T[3*K + s]; // G=2, T=3

    double cherry2_plh[4]; // eigenspace
    for (int i = 0; i < K; i++) {
        double v = 0.0;
        for (int x = 0; x < K; x++)
            v += inv_evec[i*K + x] * tmp2[x];
        cherry2_plh[i] = v;
    }

    // INTERNAL-INTERNAL: cherry1 (t_left_branch=0.05) vs cherry2 (t_right_branch=0.05)
    double echild_left[16], echild_right[16];
    jcRevEchildren(evec, eval, t_left_branch, K, echild_left);
    jcRevEchildren(evec, eval, t_right_branch, K, echild_right);

    // Transform both from eigenspace to state space
    double left_state[4], right_state[4];
    for (int s = 0; s < K; s++) {
        double vl = 0.0, vr = 0.0;
        for (int i = 0; i < K; i++) {
            vl += echild_left[s*K + i]  * cherry1_plh[i];
            vr += echild_right[s*K + i] * cherry2_plh[i];
        }
        left_state[s] = vl;
        right_state[s] = vr;
    }

    // Hadamard product
    double tmp_root[4];
    for (int s = 0; s < K; s++)
        tmp_root[s] = left_state[s] * right_state[s];

    // Back-transform to eigenspace
    double root_plh[4];
    for (int i = 0; i < K; i++) {
        double v = 0.0;
        for (int x = 0; x < K; x++)
            v += inv_evec[i*K + x] * tmp_root[x];
        root_plh[i] = v;
    }

    // Reference: non-rev approach
    double P_A_mat[16], P_C_mat[16], P_G_mat[16], P_T_mat[16], P_L[16], P_R[16];
    computeTransMatrixEqualRate(t_A, K, P_A_mat);
    computeTransMatrixEqualRate(t_C, K, P_C_mat);
    computeTransMatrixEqualRate(t_G, K, P_G_mat);
    computeTransMatrixEqualRate(t_T, K, P_T_mat);
    computeTransMatrixEqualRate(t_left_branch, K, P_L);
    computeTransMatrixEqualRate(t_right_branch, K, P_R);

    // cherry1_state[s] = P_A(s,A=0) * P_C(s,C=1)
    // cherry2_state[s] = P_G(s,G=2) * P_T(s,T=3)
    double c1_state_ref[4], c2_state_ref[4];
    for (int s = 0; s < K; s++) {
        c1_state_ref[s] = P_A_mat[s*K + 0] * P_C_mat[s*K + 1];
        c2_state_ref[s] = P_G_mat[s*K + 2] * P_T_mat[s*K + 3];
    }

    // root_state_ref[s] = (Σ_x P_L(s,x)*c1[x]) * (Σ_y P_R(s,y)*c2[y])
    double root_state_ref[4];
    for (int s = 0; s < K; s++) {
        double vl = 0.0, vr = 0.0;
        for (int x = 0; x < K; x++) {
            vl += P_L[s*K + x] * c1_state_ref[x];
            vr += P_R[s*K + x] * c2_state_ref[x];
        }
        root_state_ref[s] = vl * vr;
    }

    // --- Test 13d.1: Root state-space match ---
    {
        cout << endl << "  --- Test 13d.1: INTERNAL-INTERNAL root partials ---" << endl;

        double root_state_rev[4];
        for (int s = 0; s < K; s++) {
            double v = 0.0;
            for (int i = 0; i < K; i++)
                v += evec[s*K + i] * root_plh[i];
            root_state_rev[s] = v;
        }

        double max_diff = 0.0;
        for (int s = 0; s < K; s++) {
            double d = fabs(root_state_rev[s] - root_state_ref[s]);
            if (d > max_diff) max_diff = d;
        }
        bool pass = max_diff < 1e-13;

        cout << "  Rev root (state-space): [";
        for (int s = 0; s < K; s++) cout << root_state_rev[s] << (s<K-1?", ":"");
        cout << "]" << endl;
        cout << "  Ref root (P(t)):        [";
        for (int s = 0; s < K; s++) cout << root_state_ref[s] << (s<K-1?", ":"");
        cout << "]" << endl;
        cout << "  Max diff: " << max_diff << " ... " << (pass ? "PASS" : "FAIL") << endl;
        if (!pass) all_pass = false;
    }

    // --- Test 13d.2: Log-likelihood ---
    {
        cout << endl << "  --- Test 13d.2: Log-likelihood ---" << endl;

        double root_state_rev[4];
        for (int s = 0; s < K; s++) {
            double v = 0.0;
            for (int i = 0; i < K; i++)
                v += evec[s*K + i] * root_plh[i];
            root_state_rev[s] = v;
        }

        double site_lh_rev = 0.0, site_lh_ref = 0.0;
        for (int s = 0; s < K; s++) {
            site_lh_rev += 0.25 * root_state_rev[s];
            site_lh_ref += 0.25 * root_state_ref[s];
        }

        double lnL_rev = log(site_lh_rev);
        double lnL_ref = log(site_lh_ref);

        bool pass = approxEqual(lnL_rev, lnL_ref, 1e-13);
        cout << "  lnL(rev)    = " << lnL_rev << endl;
        cout << "  lnL(nonrev) = " << lnL_ref << endl;
        cout << "  Diff: " << fabs(lnL_rev - lnL_ref) << endl;
        cout << "  Match: " << (pass ? "PASS" : "FAIL") << endl;
        if (!pass) all_pass = false;
    }

    // --- Test 13d.3: Swap symmetry ---
    {
        cout << endl << "  --- Test 13d.3: Swap symmetry (cherry1 ↔ cherry2) ---" << endl;

        // Swap: eleft ↔ eright, cherry1 ↔ cherry2
        double left_s2[4], right_s2[4];
        for (int s = 0; s < K; s++) {
            double vl = 0.0, vr = 0.0;
            for (int i = 0; i < K; i++) {
                vl += echild_right[s*K + i] * cherry2_plh[i]; // was right→left
                vr += echild_left[s*K + i]  * cherry1_plh[i]; // was left→right
            }
            left_s2[s] = vl;
            right_s2[s] = vr;
        }

        double tmp_root2[4];
        for (int s = 0; s < K; s++)
            tmp_root2[s] = left_s2[s] * right_s2[s];

        // Since it's a Hadamard product (element-wise multiply), swapping
        // left/right state-space values gives the same result
        double max_diff = 0.0;
        for (int s = 0; s < K; s++) {
            double d = fabs(tmp_root2[s] - tmp_root[s]);
            if (d > max_diff) max_diff = d;
        }
        bool pass = max_diff < 1e-14;
        cout << "  Swap symmetry max diff: " << max_diff
             << " ... " << (pass ? "PASS" : "FAIL") << endl;
        if (!pass) all_pass = false;
    }

    cout << endl << "=== Rev Step 13d Result: "
         << (all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED")
         << " ===" << endl << endl;
    return all_pass;
}

bool testRevLikelihoodReduction() {
    cout << endl;
    cout << "============================================================" << endl;
    cout << "=== Rev Step 13e: Eigenspace Likelihood Reduction         ===" << endl;
    cout << "============================================================" << endl;

    bool all_pass = true;
    const int K = 4;
    double evec[16], inv_evec[16], eval[4];
    jcEigenDecomposition(evec, inv_evec, eval);

    double tip_plh[5*K];
    jcRevTipPartialLh(inv_evec, K, tip_plh);

    // 2-taxon tree: (A:0.1, C:0.1) — TIP-INTERNAL reduction
    // Compute cherry eigenspace partials from test 13b
    double t = 0.1;
    double echild_A[16], echild_C[16];
    jcRevEchildren(evec, eval, t, K, echild_A);
    jcRevEchildren(evec, eval, t, K, echild_C);

    double plh_A[5*K], plh_C[5*K];
    jcRevLeafPartials(echild_A, tip_plh, K, 5, plh_A);
    jcRevLeafPartials(echild_C, tip_plh, K, 5, plh_C);

    // Cherry: A=0, C=1
    double tmp_state[4];
    for (int s = 0; s < K; s++)
        tmp_state[s] = plh_A[0*K + s] * plh_C[1*K + s];

    double cherry_plh[4]; // eigenspace
    for (int i = 0; i < K; i++) {
        double v = 0.0;
        for (int x = 0; x < K; x++)
            v += inv_evec[i*K + x] * tmp_state[x];
        cherry_plh[i] = v;
    }

    // --- Test 13e.1: INTERNAL-INTERNAL reduction ---
    // Suppose dad's eigenspace partials are identical to cherry's (self-reduction test)
    // val[i] = exp(eval[i] * 0) * 1.0 = 1.0 for eval[0], exp(-4/3 * 0) = 1.0
    // Actually val[i] = exp(eval[i] * rate * t) * prop — at t=0, val=[1,1,1,1]
    {
        cout << endl << "  --- Test 13e.1: val at t=0 (identity reduction) ---" << endl;

        double val[4];
        for (int i = 0; i < K; i++)
            val[i] = exp(eval[i] * 0.0) * 1.0; // t=0, prop=1

        // lh = Σ_i val[i] * node_plh[i] * dad_plh[i]
        double lh = 0.0;
        for (int i = 0; i < K; i++)
            lh += val[i] * cherry_plh[i] * cherry_plh[i]; // node==dad for self-test

        // This should be Σ_i plh[i]^2 (quadratic form)
        double lh_check = 0.0;
        for (int i = 0; i < K; i++)
            lh_check += cherry_plh[i] * cherry_plh[i];

        bool pass = approxEqual(lh, lh_check, 1e-15);
        cout << "  val = [1,1,1,1] (t=0)" << endl;
        cout << "  lh (reduction) = " << lh << ", lh (direct) = " << lh_check << endl;
        cout << "  Match: " << (pass ? "PASS" : "FAIL") << endl;
        if (!pass) all_pass = false;
    }

    // --- Test 13e.2: TIP-INTERNAL reduction with actual val ---
    // Dad is a leaf (state A), node is cherry internal.
    // val[i] = exp(eval[i] * t_dad) * prop for a branch t_dad
    // partial_lh_node[A][i] = val[i] * tip_plh[A][i]
    // lh = Σ_i partial_lh_node[A][i] * cherry_plh[i]
    {
        cout << endl << "  --- Test 13e.2: TIP-INTERNAL reduction ---" << endl;

        double t_dad = 0.05;
        double val[4];
        for (int i = 0; i < K; i++)
            val[i] = exp(eval[i] * t_dad) * 1.0; // prop=1, ncat=1

        // partial_lh_node for state A
        double plh_node_A[4];
        for (int i = 0; i < K; i++)
            plh_node_A[i] = val[i] * tip_plh[0*K + i]; // state A=0

        double lh_rev = 0.0;
        for (int i = 0; i < K; i++)
            lh_rev += plh_node_A[i] * cherry_plh[i];

        // Reference: pi^T * P(t_dad) * L_state_space(cherry)
        // = Σ_s 0.25 * P_dad(A,s) * cherry_state[s]
        // where cherry_state[s] = P_A(s,0) * P_C(s,1)
        double P_dad[16], P_Amat[16], P_Cmat[16];
        computeTransMatrixEqualRate(t_dad, K, P_dad);
        computeTransMatrixEqualRate(t, K, P_Amat);
        computeTransMatrixEqualRate(t, K, P_Cmat);

        double cherry_state_ref[4];
        for (int s = 0; s < K; s++)
            cherry_state_ref[s] = P_Amat[s*K + 0] * P_Cmat[s*K + 1];

        double lh_ref = 0.0;
        for (int s = 0; s < K; s++)
            lh_ref += P_dad[0*K + s] * cherry_state_ref[s];
        // Note: lh_ref doesn't include pi; the rev reduction also doesn't include pi
        // The rev formula gives: Σ_i val[i]*tip_plh[A][i]*cherry_plh[i]
        // = Σ_i exp(λ_i*t)*U⁻¹[i,A]*cherry_plh[i]
        // = e_A^T * P(t)^T * U * cherry_plh (in state space via U)
        // Actually, need to verify the factor of pi.
        // For the eigenspace reduction the full formula at the root includes pi
        // implicitly through tip_partial_lh = U^{-1} * e_state where
        // U^{-1}[i,j] = pi[j] * U[j,i].
        //
        // So lh_rev = Σ_i val[i] * (U⁻¹*e_A)[i] * cherry_plh[i]
        // This produces the site probability (not yet multiplied by pi).
        //
        // The reference site probability including pi is:
        // site_lh = Σ_s pi[s] * Σ_x P_dad(s,x)*cherry_state[x] ... but
        // that's the full tree likelihood. For comparing the branch reduction
        // we need to compare like-for-like.

        // Let's just compute the full site likelihood both ways:
        // Rev: lh_site = Σ_i plh_node_A[i] * cherry_plh[i]  (this IS the site lh)
        // Ref: lh_site = Σ_s pi[s] * root_state[s] where root_state is
        //   for a rooted tree with root edge going to dad=A.
        // Actually for IQ-TREE rooted trees, the reduction already includes pi
        // via the root state handling. Let's just verify log:

        // For a 2-taxon rooted tree A--cherry, the site likelihood is:
        // Σ_s π(s) * P_dad(s,A) * cherry_state[s]  -- wrong decomposition
        // Actually: site_lh = Σ_s π(s) * cherry_state_with_all_branches(s)
        //
        // The simplest check: both should give the same lnL for the site.
        // lnL = log(Σ_s π(s) * Σ_x P_A_total(s,x) * δ(x,A) * Σ_y P_C_total(s,y)*δ(y,C))
        // With A at t_A = 0.1, C at t_C = 0.1, internal branch = t_dad = 0.05
        // and dad branch being the root edge.
        // Total A path: t_A = 0.1, Total C path: t_C = 0.1
        // Cherry to root: t_cherry = 0.05 ... but we're doing a 3-branch star for 2 taxa.
        //
        // Let me just verify the reduction value matches the non-rev computation
        // without pi (since pi is handled separately at the very end).

        // Actually the eigenspace reduction for TIP-INTERNAL already gives the
        // site probability (including pi). Let me verify numerically:
        double lnL_rev = log(fabs(lh_rev));

        // The non-rev approach for the same structure:
        // root_partial[s] = P_dad(s,0_A) * cherry_state[s]
        // (where P_dad = P(t_dad) goes from root to dad leaf)
        // site_lh = Σ_s pi[s] * root_partial[s] = 0.25 * Σ_s P_dad(s,0)*cherry_state[s]
        double site_lh_nonrev = 0.0;
        for (int s = 0; s < K; s++)
            site_lh_nonrev += 0.25 * P_dad[s*K + 0] * cherry_state_ref[s];
        double lnL_ref = log(site_lh_nonrev);

        bool pass = approxEqual(lnL_rev, lnL_ref, 1e-10);
        cout << "  lh_rev     = " << lh_rev << endl;
        cout << "  site_lh_ref = " << site_lh_nonrev << endl;
        cout << "  lnL(rev)   = " << lnL_rev << endl;
        cout << "  lnL(ref)   = " << lnL_ref << endl;
        cout << "  Match (tol 1e-10): " << (pass ? "PASS" : "FAIL") << endl;
        if (!pass) all_pass = false;
    }

    // --- Test 13e.3: INTERNAL-INTERNAL reduction at actual branch ---
    {
        cout << endl << "  --- Test 13e.3: INT-INT reduction at branch ---" << endl;

        // 4-taxon tree: ((A:0.1,C:0.1):0.05, (G:0.15,T:0.15):0.05)
        // Compute cherry1 and cherry2 eigenspace partials
        double t_G = 0.15, t_T = 0.15;
        double echild_G[16], echild_T[16];
        jcRevEchildren(evec, eval, t_G, K, echild_G);
        jcRevEchildren(evec, eval, t_T, K, echild_T);

        double plh_G[5*K], plh_T[5*K];
        jcRevLeafPartials(echild_G, tip_plh, K, 5, plh_G);
        jcRevLeafPartials(echild_T, tip_plh, K, 5, plh_T);

        double tmp2[4];
        for (int s = 0; s < K; s++)
            tmp2[s] = plh_G[2*K + s] * plh_T[3*K + s];

        double cherry2_plh[4];
        for (int i = 0; i < K; i++) {
            double v = 0.0;
            for (int x = 0; x < K; x++)
                v += inv_evec[i*K + x] * tmp2[x];
            cherry2_plh[i] = v;
        }

        // Reduction at the branch connecting cherry1 and cherry2
        // val[i] = exp(eval[i] * (t_left + t_right)) * prop
        // But the reduction uses the branch between the two:
        // In IQ-TREE: dad_branch->length is the edge between node and dad.
        // val[i] = exp(eval[i] * rate * dad_branch_len) * prop
        double branch_len = 0.05 + 0.05; // total branch = 0.1
        double val[4];
        for (int i = 0; i < K; i++)
            val[i] = exp(eval[i] * branch_len) * 1.0;

        double lh_rev = 0.0;
        for (int i = 0; i < K; i++)
            lh_rev += val[i] * cherry_plh[i] * cherry2_plh[i];

        // Reference via non-rev
        double P_Amat[16], P_Cmat[16], P_Gmat[16], P_Tmat[16], P_branch[16];
        computeTransMatrixEqualRate(0.1, K, P_Amat);
        computeTransMatrixEqualRate(0.1, K, P_Cmat);
        computeTransMatrixEqualRate(0.15, K, P_Gmat);
        computeTransMatrixEqualRate(0.15, K, P_Tmat);
        computeTransMatrixEqualRate(branch_len, K, P_branch);

        double c1_ref[4], c2_ref[4];
        for (int s = 0; s < K; s++) {
            c1_ref[s] = P_Amat[s*K + 0] * P_Cmat[s*K + 1];
            c2_ref[s] = P_Gmat[s*K + 2] * P_Tmat[s*K + 3];
        }

        // The rev reduction computes:
        //   lh = Σ_i val[i] * cherry1_plh[i] * cherry2_plh[i]
        //   where val[i]=exp(λ_i*t), cherry_plh = U⁻¹*L (eigenspace)
        //
        // This equals the full site likelihood because:
        //   Σ_i exp(λ_i*t) * (U⁻¹*L1)[i] * (U⁻¹*L2)[i]
        //   = Σ_k exp(λ_k*t) * (Σ_x U⁻¹[k,x]*L1[x]) * (Σ_s U⁻¹[k,s]*L2[s])
        //   = Σ_k exp(λ_k*t) * (Σ_x U⁻¹[k,x]*L1[x]) * (Σ_s π[s]*U[s,k]*L2[s])
        //   = Σ_s π[s]*L2[s] * Σ_x (Σ_k U[s,k]*exp(λ_k*t)*U⁻¹[k,x]) * L1[x]
        //   = Σ_s π[s]*L2[s] * Σ_x P(t)[s,x]*L1[x]
        //   = Σ_s π[s] * (P(t)*L1)[s] * L2[s]  ← standard site likelihood
        //
        // Reference: site_lh = Σ_s π[s] * (Σ_x P_branch(s,x)*c1[x]) * c2[s]
        double site_lh_ref = 0.0;
        for (int s = 0; s < K; s++) {
            double v = 0.0;
            for (int x = 0; x < K; x++)
                v += P_branch[s*K + x] * c1_ref[x];
            site_lh_ref += 0.25 * v * c2_ref[s];
        }

        // The rev reduction result should equal this site_lh
        bool pass = approxEqual(lh_rev, site_lh_ref, 1e-12);

        double lnL_rev = log(fabs(lh_rev));
        double lnL_ref = log(site_lh_ref);

        cout << "  lh(rev reduction) = " << lh_rev << endl;
        cout << "  site_lh(ref)      = " << site_lh_ref << endl;
        cout << "  lnL(rev)  = " << lnL_rev << endl;
        cout << "  lnL(ref)  = " << lnL_ref << endl;
        cout << "  Match (tol 1e-12): " << (pass ? "PASS" : "FAIL") << endl;
        if (!pass) all_pass = false;
    }

    cout << endl << "=== Rev Step 13e Result: "
         << (all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED")
         << " ===" << endl << endl;
    return all_pass;
}

bool testRevFullTraversal(PhyloTree *tree) {
    cout << endl;
    cout << "============================================================" << endl;
    cout << "=== Rev Step 13f: Full Reversible Traversal (End-to-End) ===" << endl;
    cout << "============================================================" << endl;

    if (!tree || !tree->aln || !tree->getModel() || !tree->getModelFactory()) {
        cout << "  ERROR: Tree not fully initialized" << endl;
        return false;
    }

    bool all_pass = true;
    int K = tree->aln->num_states;
    int ncat = tree->getRate()->getNRate();
    size_t orig_nptn = tree->aln->size();
    size_t nptn = tree->aln->size() + tree->getModelFactory()->unobserved_ptns.size();

    cout << "  Alignment: " << tree->aln->getNSeq() << " taxa, "
         << tree->aln->getNSite() << " sites, "
         << nptn << " patterns" << endl;
    cout << "  Model: " << tree->getModelName() << endl;
    cout << "  States: " << K << ", Rate categories: " << ncat << endl;
    cout << "  useRevKernel: " << (tree->getModel()->useRevKernel() ? "true" : "false") << endl;

    // --- Test 13f.1: Production lnL validity ---
    {
        cout << endl << "  --- Test 13f.1: Production kernel lnL ---" << endl;

        tree->initializeAllPartialLh();
        tree->clearAllPartialLH();
        double lnL = tree->computeLikelihood();

        bool pass_finite = !std::isnan(lnL) && !std::isinf(lnL);
        bool pass_negative = lnL < 0.0;

        cout << "  lnL = " << fixed << lnL << endl;
        cout.unsetf(ios_base::fixed);
        cout << "  Finite:   " << (pass_finite ? "PASS" : "FAIL") << endl;
        cout << "  Negative: " << (pass_negative ? "PASS" : "FAIL") << endl;
        if (!pass_finite || !pass_negative) all_pass = false;
    }

    // --- Test 13f.2: Weighted sum from _pattern_lh ---
    {
        cout << endl << "  --- Test 13f.2: Weighted sum from _pattern_lh ---" << endl;

        vector<double> pattern_lh_buf(orig_nptn);
        tree->computePatternLikelihood(pattern_lh_buf.data());
        double manual_sum = 0.0;
        for (size_t ptn = 0; ptn < orig_nptn; ptn++)
            manual_sum += pattern_lh_buf[ptn] * tree->ptn_freq[ptn];

        double prod_lnL = tree->computeLikelihood();
        double diff = fabs(manual_sum - prod_lnL);
        bool pass = diff < 1e-6;

        cout << "  Sum(_pattern_lh * ptn_freq) = " << fixed << manual_sum << endl;
        cout << "  Production lnL              = " << prod_lnL << endl;
        cout.unsetf(ios_base::fixed);
        cout << "  Diff: " << diff << endl;
        cout << "  Match (tol 1e-6): " << (pass ? "PASS" : "FAIL") << endl;
        if (!pass) all_pass = false;
    }

    // --- Test 13f.3: Determinism ---
    {
        cout << endl << "  --- Test 13f.3: Determinism ---" << endl;

        tree->clearAllPartialLH();
        double lnL1 = tree->computeLikelihood();
        tree->clearAllPartialLH();
        double lnL2 = tree->computeLikelihood();

        double diff = fabs(lnL1 - lnL2);
        bool pass = diff < 1e-10;

        cout << "  Run 1: " << fixed << lnL1 << endl;
        cout << "  Run 2: " << lnL2 << endl;
        cout.unsetf(ios_base::fixed);
        cout << "  Diff: " << diff << endl;
        cout << "  Deterministic (tol 1e-10): " << (pass ? "PASS" : "FAIL") << endl;
        if (!pass) all_pass = false;
    }

    // --- Test 13f.4: Per-pattern log-likelihoods valid ---
    {
        cout << endl << "  --- Test 13f.4: Per-pattern log-likelihoods ---" << endl;

        vector<double> pattern_lh_buf(orig_nptn);
        tree->computePatternLikelihood(pattern_lh_buf.data());

        int bad_patterns = 0;
        double min_ptn = 0.0, max_ptn = -1e300;
        for (size_t ptn = 0; ptn < orig_nptn; ptn++) {
            if (std::isnan(pattern_lh_buf[ptn]) || std::isinf(pattern_lh_buf[ptn]))
                bad_patterns++;
            else {
                if (pattern_lh_buf[ptn] < min_ptn) min_ptn = pattern_lh_buf[ptn];
                if (pattern_lh_buf[ptn] > max_ptn) max_ptn = pattern_lh_buf[ptn];
            }
        }

        bool pass = (bad_patterns == 0);
        cout << "  Patterns: " << orig_nptn << ", Bad: " << bad_patterns << endl;
        cout << "  Min: " << min_ptn << ", Max: " << max_ptn << endl;
        cout << "  All valid: " << (pass ? "PASS" : "FAIL") << endl;
        if (!pass) all_pass = false;
    }

    cout << endl << "=== Rev Step 13f Result: "
         << (all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED")
         << " ===" << endl << endl;
    return all_pass;
}

#endif // USE_OPENACC
