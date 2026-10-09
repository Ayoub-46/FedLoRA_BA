# FedLoRA_BA: federated LoRA backdoor benchmark starter

This repository is a small, runnable base for federated LoRA backdoor experiments. It keeps
the client/server training loop, task loaders, defenses, aggregation strategies,
configuration wiring, and run provenance. The quickstart uses a **text classification**
trigger attack. Image and instruction tasks run as benign baselines and have hooks for
future task-specific attacks and success metrics.

## Run a paired smoke benchmark

Install the package in an environment with Python 3.10–3.12:

```bash
pip install -e '.[dev]'
python experiments/paired_backdoor.py
```

The script runs the same synthetic task twice with the same seed and partition. Its
clean arm retains the trigger and target for ASR measurement but starts poisoning after
the last training round. It verifies matching partitions and writes
`outputs/paired_backdoor/comparison.json`. The tiny model and dataset check plumbing;
their numbers are not research results.

Pass Hydra overrides to use a different task, model, defense, or budget:

```bash
python experiments/paired_backdoor.py data=sst2 model=bert_base \
  defense=norm_bound fl.num_rounds=10 data.num_train=1000 device=cuda
```

You can run a single arm with `fedlora-run experiment=quickstart`, or select components
directly, such as `fedlora-run data=ag_news model=bert_base attack=none`.

## Wiring

1. `configs/config.yaml` composes the model, data, FL, aggregation, defense, attack,
   and experiment groups. An experiment preset changes only its required fields.
2. `runner.run_experiment` loads and partitions data, builds the model and clients,
   chooses the attack/defense/aggregation, and records the resolved config and partition.
3. `fl/client.py` trains each client's LoRA adapter. Attack hooks can alter local data,
   loss, or the submitted update.
4. `fl/server.py` broadcasts the global state, collects client updates, gets defense
   weights, aggregates adapters, evaluates, and logs every round.
5. Each run writes a manifest, `partition.json`, per-round metrics, and `summary.json`
   under `outputs/<experiment>/<config hash>_seed<seed>/`.

The defense interface returns nonnegative client weights; zero rejects a client.
`defenses/base.py` defines it, and `runner.build_defense_from_cfg` wires a named config.
Available implementations include norm bound, Multi-Krum, cosine, spectral, aggregate
norm, factor broadcast, and FLTrust. FLTrust reserves a separate trusted root shard.
Aggregation options include factor averaging (`fedit_naive`) and exact effective-update
aggregation (`exact_deltaw`), plus canonicalized and consensus-frame variants.

| Task kind | Data configs | Model configs | Current backdoor baseline |
| --- | --- | --- | --- |
| Text classification | synthetic, SST-2, AG News, Banking77 | tiny BERT, BERT, RoBERTa | Trigger poisoning and non-target ASR |
| Instruction tuning | synthetic instruct, Dolly, Spider | tiny Qwen, Qwen, Llama | Task loader and evaluation only |
| Image classification | EuroSAT | tiny CLIP, CLIP ViT-B/32 | Task loader and evaluation only |

Downloaded datasets and pretrained models require their respective caches or network
access. The synthetic quickstart needs neither.

## Extend the benchmark

- Add a task in `data/base.py`, with a loader in `data/` and a matching `configs/data/`
  file. Add its model mapping in `models/lora_wrapper.py` if needed.
- Add an attack in `attacks/`, register it, then parse its config in
  `runner.build_attack_from_cfg`. Add a task-appropriate success metric and server
  evaluation path for attacks on instruction or image tasks.
- Add a defense under `defenses/`, register it in `defenses/__init__.py`, and wire its
  config in `runner.build_defense_from_cfg`.
- Add a preset under `configs/experiment/` or a paired driver under `experiments/`.
  Keep seed, partition, model, and training budget fixed across clean/attacked arms.

The included text ASR is the fraction of **non-target** held-out examples classified
as the target after trigger insertion. The backdoor baseline does not yet poison images
or instruction responses. Runs on those tasks reject this attack explicitly.

## AG News recovery connectivity study

The AG News preset reserves 500 clean articles for server-side path fitting and 500
disjoint articles for validation before the remaining 2,000 articles are partitioned
among clients. The official labelled test split is used only for final path reporting.
The first attack is a `cf` prefix targeting class 0. It **replaces** selected non-target
examples, so the malicious shard size and aggregation weight stay fixed. This differs
from the older X4 append baseline, which remains available as `attack.poison_mode=append`.

```bash
python experiments/ag_news_recovery.py
python experiments/recovery_connectivity.py outputs/ag_news_recovery/<study_id>/pair.json
```

The first command trains matched clean and poisoned federated arms, saves their
post-aggregation global LoRA factors and classifier head at the final attack round,
and reports the observed ASR gain and clean-accuracy loss against the clean twin at
every attack round.
It then continues the poisoned model with clean-only federated training for up to
20 rounds. It selects the first recovered checkpoint whose validation ASR is no
higher than the clean twin's and whose validation accuracy is within two percentage
points. There is no attack-strength gate; inspect the reported attack effect before
interpreting a pair as backdoor recovery.
The attack trajectory is a pilot measurement, not a claim that an arbitrary ASR
gain is meaningful. Choose any later attack-strength criterion from pilot runs,
then fix it before comparing paths across fresh seeds.
If no checkpoint qualifies, `pair.json` records `no_eligible_recovery` and the path
command rejects that pair. The run uses CUDA when available and otherwise CPU.

The second command searches paths at rank `r`, rank `2r`, and with all frozen base
parameters allowed to bend. The classifier head follows the same linear interpolation
in every condition. It fits paths only on the reserved server-clean set, selects the
best candidate in each condition by validation clean-loss maximum, and reports dense
test curves of clean cross-entropy, accuracy, ASR, and triggered correctness. The
triggered metrics use articles whose original label is not class 0; the report also
includes triggered correctness conditioned on the clean prediction being correct. The
sampled `H_0` merge events in `connectivity/report.json` are estimates over evaluated
paths; an unsuccessful search is not evidence of disconnected low-loss regions.

Global checkpoints include a JSON sidecar with the model/data configuration, seed,
partition digest, and frozen-base fingerprint. The continuation loader rejects a
checkpoint whose static context does not match its run.
