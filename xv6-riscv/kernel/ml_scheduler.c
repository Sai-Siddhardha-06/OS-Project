#include "types.h"
#include "param.h"
#include "memlayout.h"
#include "riscv.h"
#include "spinlock.h"
#include "proc.h"
#include "defs.h"
#include "sched_config.h"
#include "model_weights.h"

static struct spinlock seed_lock;
static uint32 global_sched_seed = BASE_RNG_SEED;

void
set_global_sched_seed(uint32 seed)
{
  acquire(&seed_lock);
  global_sched_seed = seed;
  release(&seed_lock);
}

uint32
get_global_sched_seed(void)
{
  uint32 s;
  acquire(&seed_lock);
  s = global_sched_seed;
  release(&seed_lock);
  return s;
}

// Xorshift32 PRNG (per-process, deterministic)
uint32
proc_rand(struct proc *p)
{
  uint32 x = p->rng_state;
  if (x == 0)
    x = 0x6d2b79f5;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  p->rng_state = x;
  return x;
}

void
ml_init_proc(struct proc *p)
{
  uint32 base = get_global_sched_seed();
  p->rng_state = ((uint32)p->pid * 2654435761u) ^ base;
  if (p->rng_state == 0)
    p->rng_state = 0x1337beef;
  p->burst_count = 0;
  p->obs_done = 0;
  p->pred_class = -1;
  p->pred_conf = 0;
  p->init_queue = -1;
  p->infer_status = 0;
}

// Extract the 9 normalized integer features from observed process behavior
void
ml_extract_features(struct proc *p, int32 features[NUM_FEATURES])
{
  uint64 now = get_ticks();
  uint64 window_ticks = (now > p->obs_start_tick) ? (now - p->obs_start_tick) : 1;
  if (window_ticks == 0)
    window_ticks = 1;

  int n = p->burst_count;
  if (n <= 0)
    n = 1;

  uint32 burst_sum = 0;
  uint32 max_burst = 0;
  uint32 exhaust_count = 0;

  for (int i = 0; i < p->burst_count; i++) {
    burst_sum += p->burst_history[i];
    if (p->burst_history[i] > max_burst)
      max_burst = p->burst_history[i];
    if (p->burst_exhausted[i])
      exhaust_count++;
  }

  uint32 avg_burst = (burst_sum * 1000) / n;
  features[0] = (int32)avg_burst;
  features[1] = (int32)(max_burst * 1000);

  // Mean Absolute Deviation (MAD)
  uint32 mad_sum = 0;
  for (int i = 0; i < p->burst_count; i++) {
    int32 diff = (int32)(p->burst_history[i] * 1000) - (int32)avg_burst;
    if (diff < 0)
      diff = -diff;
    mad_sum += diff;
  }
  features[2] = (int32)(mad_sum / n);

  // Exhaustion fraction (0..1000)
  features[3] = (int32)((exhaust_count * 1000) / n);

  // Blocking events per tick * 1000
  features[4] = (int32)((p->blocking_count * 1000) / window_ticks);

  // Voluntary yield events per tick * 1000
  features[5] = (int32)((p->voluntary_yields * 1000) / window_ticks);

  // Sleep time ratio (0..1000)
  uint32 sleep_ratio = (uint32)((p->total_sleep_ticks * 1000) / window_ticks);
  if (sleep_ratio > 1000)
    sleep_ratio = 1000;
  features[6] = (int32)sleep_ratio;

  // CPU utilization (0..1000)
  uint32 cpu_util = (uint32)((p->total_cpu_ticks * 1000) / window_ticks);
  if (cpu_util > 1000)
    cpu_util = 1000;
  features[7] = (int32)cpu_util;

  // Number of completed bursts scaled by 1000 / OBS_MAX_BURSTS
  uint32 burst_cnt_scaled = (uint32)((p->burst_count * 1000) / OBS_MAX_BURSTS);
  if (burst_cnt_scaled > 1000)
    burst_cnt_scaled = 1000;
  features[8] = (int32)burst_cnt_scaled;
}

void
ml_record_burst(struct proc *p, uint burst_ticks, int exhausted)
{
  if (p->obs_done)
    return;

  if (p->burst_count < OBS_MAX_BURSTS) {
    p->burst_history[p->burst_count] = burst_ticks;
    p->burst_exhausted[p->burst_count] = exhausted;
    p->burst_count++;
  }

  uint64 now = get_ticks();
  int window_expired = (now >= p->obs_start_tick && (now - p->obs_start_tick) >= OBS_MAX_WINDOW_TICKS);

  if (p->burst_count >= OBS_MIN_BURSTS || window_expired || p->burst_count >= OBS_MAX_BURSTS) {
    ml_observe_and_classify(p);
  }
}

// Fixed-point exp using 256-entry lookup table for delta in [-8.0, 0.0]
static int32
fixed_point_exp(int32 delta_q16)
{
  if (delta_q16 >= 0)
    return Q_ONE;

  int32 min_neg = -(EXP_MAX_NEG << Q_SHIFT);
  if (delta_q16 <= min_neg)
    return 0;

  int idx = (int)((-(int64)delta_q16 * (EXP_TABLE_ENTRIES - 1)) / ((int64)EXP_MAX_NEG << Q_SHIFT));
  if (idx >= EXP_TABLE_ENTRIES)
    idx = EXP_TABLE_ENTRIES - 1;

  return exp_table[idx];
}

// Forward inference pass (9 -> 16 -> 8 -> 4 MLP in Q16.16)
void
ml_predict(const int32 features[NUM_FEATURES], int32 probs[OUT_DIM], int *pred_class, int *pred_conf)
{
  // 1. Normalization: x_norm = ((raw - mean) << 16) / std
  int32 x_norm[NUM_FEATURES];
  for (int i = 0; i < NUM_FEATURES; i++) {
    int64 diff = ((int64)features[i] << Q_SHIFT) - (int64)model_mean[i];
    x_norm[i] = (int32)((diff << Q_SHIFT) / (int64)model_std[i]);
  }

  // 2. Hidden Layer 1 (9 -> 16 + ReLU)
  int32 z1[H1_DIM];
  for (int j = 0; j < H1_DIM; j++) {
    int64 acc = model_b1[j];
    for (int i = 0; i < NUM_FEATURES; i++) {
      acc += ((int64)x_norm[i] * (int64)model_W1[i][j]) >> Q_SHIFT;
    }
    z1[j] = (acc > 0) ? (int32)acc : 0;
  }

  // 3. Hidden Layer 2 (16 -> 8 + ReLU)
  int32 z2[H2_DIM];
  for (int j = 0; j < H2_DIM; j++) {
    int64 acc = model_b2[j];
    for (int i = 0; i < H1_DIM; i++) {
      acc += ((int64)z1[i] * (int64)model_W2[i][j]) >> Q_SHIFT;
    }
    z2[j] = (acc > 0) ? (int32)acc : 0;
  }

  // 4. Output Layer (8 -> 4 logits)
  int32 z3[OUT_DIM];
  int32 max_z = -2147483647;
  for (int j = 0; j < OUT_DIM; j++) {
    int64 acc = model_b3[j];
    for (int i = 0; i < H2_DIM; i++) {
      acc += ((int64)z2[i] * (int64)model_W3[i][j]) >> Q_SHIFT;
    }
    z3[j] = (int32)acc;
    if (z3[j] > max_z)
      max_z = z3[j];
  }

  // 5. Softmax with Exp Table
  int64 exp_vals[OUT_DIM];
  int64 sum_exp = 0;
  for (int j = 0; j < OUT_DIM; j++) {
    int32 delta = z3[j] - max_z;
    exp_vals[j] = fixed_point_exp(delta);
    sum_exp += exp_vals[j];
  }
  if (sum_exp == 0)
    sum_exp = 1;

  int32 sum_probs = 0;
  int best_c = 0;
  int32 best_p = -1;
  for (int j = 0; j < OUT_DIM; j++) {
    probs[j] = (int32)(((int64)exp_vals[j] * 1000) / sum_exp);
    sum_probs += probs[j];
    if (probs[j] > best_p) {
      best_p = probs[j];
      best_c = j;
    }
  }

  // Fix tiny rounding discrepancy so probabilities sum exactly to 1000
  if (sum_probs != 1000) {
    probs[best_c] += (1000 - sum_probs);
    best_p = probs[best_c];
  }

  if (pred_class)
    *pred_class = best_c;
  if (pred_conf)
    *pred_conf = best_p;
}

// Assigns process to initial MLFQ queue based on inferred class or baseline control
void
ml_infer_and_assign(struct proc *p, int32 features[NUM_FEATURES])
{
  int32 probs[OUT_DIM];
  int pred_c = 0;
  int conf = 0;

  uint64 infer_start = r_time();
  ml_predict(features, probs, &pred_c, &conf);
  uint64 infer_end = r_time();
  p->infer_cycles = infer_end - infer_start;

  p->pred_class = pred_c;
  p->pred_conf = conf;
  p->infer_status = 1;

  int assigned_q = 0;

#if SCHED_MODE == SCHED_ML_PROB || SCHED_MODE == SCHED_ML_DET
  // Combined queue probability: P(Q_j) = sum_c P(class=c) * Table[c][j]
  int32 q_prob[NUM_QUEUES];
  for (int q = 0; q < NUM_QUEUES; q++) {
    int64 acc = 0;
    for (int c = 0; c < OUT_DIM; c++) {
      acc += (int64)probs[c] * (int64)class_queue_prob[c][q];
    }
    q_prob[q] = (int32)(acc / 1000);
  }
#endif

#if SCHED_MODE == SCHED_ML_PROB
  // Mode 2: Sample queue from P(Q) using per-process PRNG
  uint32 roll = proc_rand(p) % 1000;
  if (roll < (uint32)q_prob[0]) {
    assigned_q = 0;
  } else if (roll < (uint32)(q_prob[0] + q_prob[1])) {
    assigned_q = 1;
  } else {
    assigned_q = 2;
  }

#elif SCHED_MODE == SCHED_ML_DET
  // Mode 3: Deterministic assignment to argmax queue
  assigned_q = 0;
  if (q_prob[1] > q_prob[assigned_q])
    assigned_q = 1;
  if (q_prob[2] > q_prob[assigned_q])
    assigned_q = 2;

#elif SCHED_MODE == SCHED_RAND_CTRL
  // Mode 4: Random control - sample from fixed prior marginal distribution
  uint32 roll = proc_rand(p) % 1000;
  if (roll < (uint32)rand_ctrl_queue_prob[0]) {
    assigned_q = 0;
  } else if (roll < (uint32)(rand_ctrl_queue_prob[0] + rand_ctrl_queue_prob[1])) {
    assigned_q = 1;
  } else {
    assigned_q = 2;
  }

#else
  assigned_q = 0;
#endif

  p->init_queue = assigned_q;
  p->priority = assigned_q;
  p->quantum_used = 0;
}

void
ml_observe_and_classify(struct proc *p)
{
  if (p->obs_done)
    return;

  p->obs_done = 1;

  int32 feats[NUM_FEATURES];
  ml_extract_features(p, feats);

#ifdef ML_TRACE
  printk("MLTRACE pid=%d name=%s feat=%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
         p->pid, p->name,
         feats[0], feats[1], feats[2], feats[3],
         feats[4], feats[5], feats[6], feats[7], feats[8]);
#endif

#if SCHED_MODE == SCHED_ML_PROB || SCHED_MODE == SCHED_ML_DET || SCHED_MODE == SCHED_RAND_CTRL
  ml_infer_and_assign(p, feats);
#else
  p->init_queue = 0;
  p->infer_status = 0;
#endif
}

// Verification self-test: runs 24 test vectors and verifies bit-level exact match
int
ml_self_test(void)
{
  int failures = 0;
  for (int v = 0; v < NUM_SELFTEST_VECTORS; v++) {
    const struct ml_test_vector *tv = &selftest_vectors[v];
    int32 probs[OUT_DIM];
    int pred = -1;
    int conf = -1;

    ml_predict(tv->feats, probs, &pred, &conf);

    int match = 1;
    if (pred != tv->expected_pred || conf != tv->expected_conf) {
      match = 0;
    }
    for (int j = 0; j < OUT_DIM; j++) {
      if (probs[j] != tv->expected_probs[j]) {
        match = 0;
        break;
      }
    }

    if (!match) {
      printk("ml_self_test FAILED on vector %d (%s): pred=%d (exp %d), conf=%d (exp %d)\n",
             v, tv->name, pred, tv->expected_pred, conf, tv->expected_conf);
      printk("  probs=[%d,%d,%d,%d] exp=[%d,%d,%d,%d]\n",
             probs[0], probs[1], probs[2], probs[3],
             tv->expected_probs[0], tv->expected_probs[1], tv->expected_probs[2], tv->expected_probs[3]);
      failures++;
    }
  }

  if (failures == 0) {
    printk("ml_self_test: all %d vectors passed bit-level check\n", NUM_SELFTEST_VECTORS);
    return 0;
  } else {
    printk("ml_self_test: %d/%d vectors failed\n", failures, NUM_SELFTEST_VECTORS);
    return -1;
  }
}
