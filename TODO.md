# Code Review TODO

Reviewed 2026-08-10. The review covered the Rust implementation, Python bindings,
examples, benchmarks, packaging, and CI. Rust is not installed in the review
environment, so `cargo test --all-features` could not run. VS Code reported no
diagnostics. Small counterexamples and distribution calculations below were
checked with Python's standard library and against SciPy's reference definitions
for [Fisher's distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nchypergeom_fisher.html)
and [Wallenius' distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nchypergeom_wallenius.html).

## P0 - Correctness and data integrity

- [x] **Replace the capped-multinomial batch sampler with exact adaptive Fisher
  sampling.** Small draws now use weighted proposals conditioned on unique IDs;
  larger draws use independent exponentially tilted binomial counts conditioned
  on their sum. Conditioning cancels the tilt and gives
  `P(x) proportional to product(C(m_i, x_i) * odds_i^x_i)`. Draws over half the
  population sample the smaller complement using reciprocal weights. The
  discriminating two-low/two-high test now targets Fisher's exact probabilities
  `[1/13, 8/13, 4/13]`, and the 1000/1000 test targets the exact mean `585.829`
  rather than the old multinomial mean `666.667`. Small-draw dispatch uses an
  O(P) maximum-item-probability bound; grouped work depends on occupied bins,
  not population size. Updated release benchmarks still need to be run on a
  Rust-equipped machine.

  By contrast, repeated
  [`select_and_remove`](src/lib.rs#L901) **does implement Wallenius' sequential
  law for the quantized weights**, provided IDs are unique and aggregate
  invariants hold. A leaf containing `m` items of scaled weight `w` is selected
  with probability `m*w/total`, then an item is selected uniformly from that
  leaf, so each remaining item has probability `w/total`; removal updates that
  mass before the next draw. For the same two-plus-two counterexample its exact
  high-group probabilities are `[1/15, 3/5, 1/3]`, matching the sequential
  Wallenius process. For the repository's 1000/1000 example, exact dynamic
  programming gives mean `618.050` and standard deviation `10.927`.

- [x] **Make `select_many` return unique IDs.** `DigitBin::get_random_many` now
  samples distinct leaf positions without removal and extracts distinct IDs
  destructively when removal is requested. Full-bin removal uses linear
  iteration/draining rather than repeated rank selection. Deterministic tests
  cover no-removal uniqueness and aggregate preservation.

- [x] **Stop silently truncating public `u64` IDs in the default, Small, and
  Medium backends.** Both [`Vec<u32>`](src/lib.rs#L31) and
  [`RoaringBitmap`](src/lib.rs#L54) cast IDs with `as u32`; the default
  constructors choose one of these backends even though `add` accepts `u64`.
  IDs `1` and `2^32 + 1` therefore collide or are returned as the same ID.
  Prefer checked conversion with an error, make the public ID type backend
  specific, or use `u64` storage by default. Add boundary tests for `u32::MAX`
  and `u32::MAX + 1` for every constructor.

- [x] **Define and enforce duplicate-ID semantics before updating aggregates.**
  [`add_recurse`](src/lib.rs#L767) increments `content_count` and
  `accumulated_value` unconditionally. `RoaringBitmap` and `RoaringTreemap` are
  sets, so adding the same ID to the same bin twice stores one entry but counts
  two. After one removal the tree can report a positive count and weight while
  the bin is empty. `Vec` stores duplicates instead, making behavior differ by
  backend. Change `DigitBin::insert` to report whether insertion occurred and
  update aggregates only on success, or reject duplicate IDs at the index API.
  Test duplicate IDs with equal and different weights across all backends.

## P1 - Reliability and API behavior

- [x] **Define a validated weight contract and make quantization stable.**
  [`weight_to_digits`](src/lib.rs#L731) silently rejects `weight >= 1.0`, while
  `add` documents only non-positive values as invalid and incorrectly claims to
  return a success boolean. Binary floating-point also moves ordinary decimal
  inputs down a bin: at precision 2, `0.29 * 100` is
  `28.999999999999996`, so the cast stores `0.28`. Return `Result`/a status from
  add operations, document the accepted range, reject non-finite values
  explicitly, and expose an integer-scaled API or a documented rounding policy.
  Test `NaN`, infinities, `0`, the minimum representable bin, `0.29`, values near
  `1`, and exactly `1`.

- [x] **Eliminate saturation rejection in batch selection.** The old loop that
  repeatedly hit already-full high-weight children has been removed. Large
  draws use centered conditional-binomial proposals, and near-full draws sample
  the small excluded complement with reciprocal weights. Rejection probability
  is governed by the centered total-count variance rather than saturated weight
  mass.

- [x] **Use checked aggregate arithmetic and document capacity limits.**
  [`content_count += 1` and `accumulated_value += scaled`](src/lib.rs#L767) can
  panic in debug builds or wrap in release builds, after which random ranges and
  subtraction are unsafe. This is reachable at roughly 18.4 billion maximum
  precision-9 weights. Use `checked_add`/`checked_sub`, propagate an error, and
  test boundaries with a narrowly constructed internal node rather than adding
  billions of records. Validate precision before computing
  `10u64.pow(precision)` in
  [`with_precision_and_capacity`](src/lib.rs#L183), too.

- [x] **Return Python exceptions instead of Rust panics.**
  [`with_precision`](src/lib.rs#L1373) uses `try_into().unwrap()`, and all Python
  constructors eventually hit Rust assertions for precision outside `1..=9`.
  Return `PyResult<Self>` with `ValueError`; do the same for invalid IDs, weights,
  and overflow once those APIs report errors. Add Python tests for `-1`, `0`,
  `10`, `256`, invalid weights, and oversized IDs.

- [x] **Protect tree invariants from external mutation.** The generic type
  exposes [`root` and `precision`](src/lib.rs#L699), and `Node` exposes its
  content and aggregates. A caller can change precision without updating
  `scale`, or alter counts independently of bins. Make invariant-bearing fields
  private and provide read-only accessors or a deliberately unsafe/debug-only
  inspection API.

- [x] **Avoid reseeding in sequential Wallenius draws.** Each index now owns a
  `WyRand` seeded once, Rust exposes `select_with_rng` and
  `select_and_remove_with_rng`, and deterministic tests verify that cloned
  indexes reproduce the same stream. Fisher convenience methods still seed once
  per batch; expose their RNG-aware variants publicly if replayable Fisher draws
  are needed.

- [x] **Verify source-distribution and local Python builds enable the binding
  feature.** Wheel jobs pass `--features python-bindings` explicitly in
  [CI](.github/workflows/maturin.yml#L48), but [pyproject.toml](pyproject.toml)
  has no `[tool.maturin] features` setting and the sdist job does not pass the
  feature. Configure the feature in project metadata if needed. Acceptance test:
  build an sdist, install it into a clean Python 3.8+ environment, and import and
  instantiate `DigitBinIndex`.

## P2 - Performance and maintainability

- [x] **Optimize the Wallenius hot path without changing its law.** Selection
  now carries the scaled decimal path as `u64`, converts to `f64` only at the
  public boundary, uses a const-generic removal path, and scans children in
  descending accumulated-mass order. This order is optimal for a linear scan.
  Direct digit lookup remains O(1), while bounded local swaps restore scan order
  after the one affected child changes at each tree level. Deterministic tests
  cover order changes and exact interval mass, and Criterion includes a skewed
  low-digit-mass workload.

- [x] **Add native sequential Wallenius batching.** Rust and Python now expose
  `select_wallenius_many_and_remove`; Rust also exposes an RNG-aware variant.
  The implementation preallocates its result and stays in the inner selection
  function for the complete sequence.

- [x] **Create final-depth bins directly.** Insertion no longer allocates a
  boxed ten-child internal node only to discard it immediately when creating a
  new leaf.

- [x] **Either implement real batch insertion/removal or correct the claims.**
  Despite documentation saying inputs are grouped and each path is traversed
  once, [`add_many`](src/lib.rs#L801) and
  [`remove_many`](src/lib.rs#L873) simply call the recursive single-item path in
  a loop. Group by scaled weight/path and update ancestor aggregates once per
  group, then benchmark against the simple loop. Also document that
  `remove_many` is partially mutating and returns `true` only when every removal
  succeeds.

- [x] **Prune empty branches after successful removals.**
  [`remove_recurse`](src/lib.rs#L847) updates aggregates but never sets an empty
  child back to `None`; selection removal has the same behavior. Memory and
  traversal cost therefore follow every weight path ever observed, not the
  current population. Prune while unwinding and add a churn test that inserts
  and removes many distinct weight paths.

- [x] **Revisit backend selection for skewed weight distributions.**
  [`with_precision_and_capacity`](src/lib.rs#L183) chooses storage from average
  `capacity / 10^precision`. A million same-weight items at precision 3 still
  select `Vec<u32>` at the threshold, creating a million-element leaf with
  linear explicit removal. Consider an ID-width parameter, caller-selected
  backend, observed/adaptive leaf promotion, or at least clearer guidance and a
  skewed-bin benchmark.

- [x] **Remove recursive target vectors and repeated child scans from batch
  selection.** The exact grouped path now allocates O(B + k) storage, where B is
  occupied bins, samples one count per bin per proposal, and traverses the tree
  once for extraction. The small-draw path uses O(k) storage and O(kP) expected
  traversal work. Balanced, skewed, near-full, and precision-5 stress benchmarks
  remain in the Criterion suite for measurement on a Rust-equipped machine.

- [x] **Correct diagnostics and benchmark accounting.** `print_stats` counts
  each child node both in its parent's boxed array and again during traversal,
  and estimates every leaf as `bin_size * size_of::<u32>()`, including
  `RoaringTreemap`; see [the memory accounting](src/lib.rs#L1230). The Python
  benchmark measures the still-live input list and allocator residue as index
  memory because its baseline is taken before creating the data
  ([selection_benchmark.py](benches/selection_benchmark.py#L45)). Use backend
  allocation APIs/serialized sizes where available and isolate memory
  measurements in fresh processes.

## Test, CI, and documentation guardrails

- [x] Add a CI job for `cargo test --all-features`, including doctests and the
  exact Fisher distribution regressions.

- [x] Add CI jobs for `cargo fmt --check`, `cargo clippy --all-targets
  --all-features -- -D warnings`, and Python smoke/behavior tests against built
  wheels. The wheel jobs still build artifacts without importing the extension.

- [x] Replace broad distribution assertions with a seeded Fisher test against
  the exact small-population PMF `[1/13, 8/13, 4/13]`, update the large Fisher
  mean to `585.829`, and update the Wallenius mean to `618.050`. Add deterministic
  uniqueness, complement-removal, and aggregate-preservation tests.

- [x] Turn `test_add_many` into an invariant comparison with assertions, and add
  cross-backend tests for add/remove parity, wrong-weight removal, zero draws,
  duplicates, ID boundaries, and aggregate consistency after every mutation.

- [x] Reconcile public documentation with behavior: `add` returns `()`, not
  `bool`; `count` returns `u64`, not `u32`; accepted weights are currently
  `(0, 1)` after quantization; `remove_many` is not atomic; the blanket `O(P)`
  claim excludes linear `Vec` removal and exact Fisher batch work in B occupied
  bins; and
  [README.md](README.md#L192) says Python 3.6+ while
  [pyproject.toml](pyproject.toml#L10) requires Python 3.8+.

## Resolution notes (2026-08-10)

- Small and Medium now reject IDs above `u32::MAX`; Large supports all `u64`
  IDs. A global Roaring membership set enforces uniqueness across weight paths.
- Mutations return `Result`. Weights must be finite and in `(0, 1)` and are
  rounded to the nearest bin with positive boundary clamping. Aggregate growth
  is checked before mutation, and invariant-bearing tree fields are private.
- Python constructors validate signed precision and mutations raise typed Python
  exceptions. Both a local ABI3 wheel and an isolated sdist installation pass
  the Python behavior suite.
- `add_many` uses atomic whole-batch validation followed by per-item radix
  traversal; the unsupported grouping claim was removed. `remove_many` documents
  ordered partial mutation and returns true only when every removal succeeds.
- All removal paths prune empty branches. Backend guidance and Criterion now
  include a 100,000-item single-bin Small/Medium comparison.
- Diagnostics no longer double-count child nodes, use backend-specific bin
  estimates, and include the global ID set. Python RSS measurement runs in a
  fresh process without retaining input data.
- CI now checks formatting, all-target/all-feature Clippy, all-feature tests,
  a built native wheel, Python behavior, and an installed sdist. Rust and Python
  tests cover IDs, duplicates, invalid weights, overflow, wrong-weight removal,
  zero draws, pruning, batch parity, and aggregate invariants.
