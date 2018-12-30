from collections import defaultdict
import matplotlib.pyplot as plt
# % matplotlib inline

# %%

def parse_checkpoint(l):
    _, loc = l.split(": ")
    _, tmp, tune, hparams_id, model = loc.split("/")
    return hparams_id, int(model.split("-")[-1])

assert parse_checkpoint("INFO:tensorflow:Saving 'checkpoint_path' summary for global step 1738: /tmp/tune/lr_3e-05_bs_16/model.ckpt-1738") == ('lr_3e-05_bs_16', 1738)

# %%

def parse_metrics(l):
    ret = {}
    for pair in l.split(": ")[-1].split(", "):
        k, v = pair.split(" = ")
        ret[k] = float(v)
    return ret

parse_metrics("INFO:tensorflow:Saving dict for global step 1821: AGAINST_F1 = 0.78014183, AGAINST_precision = 0.7819905, AGAINST_recall = 0.7783019, FAVOR_F1 = 0.67659575, FAVOR_precision = 0.6708861, FAVOR_recall = 0.68240345, NONE_F1 = 0.6549296, NONE_precision = 0.65957445, NONE_recall = 0.6503497, eval_accuracy = 0.7275, eval_loss = 1.0763706, global_step = 1821, loss = 1.0763706, macro_avg = 0.72836876")

# %%

def metrics_by_hparams_id(fn):
    ret = defaultdict(lambda: [])
    metrics = None
    with open(fn, "r") as f:
        for l in f.readlines():
            if l.startswith("INFO:tensorflow:Saving 'checkpoint_path'"):
                hparams_id, step = parse_checkpoint(l)
                ret[hparams_id].append(metrics)
            elif l.startswith("INFO:tensorflow:Saving dict"):
                metrics = parse_metrics(l)
    return ret

metrics = metrics_by_hparams_id("cb.txt")
# %%

pluck = lambda k, model_metrics: [m[k] for m in model_metrics]
keys = sorted(metrics.keys())
max_by_hp = {hp: max(pluck("macro_avg", model_metrics)) for hp, model_metrics in metrics.items()}

plt.figure(figsize=(10, 5))
plt.ylim(0.4, 0.8)
for hp in keys:
    plt.plot(pluck("global_step", metrics[hp]), pluck("macro_avg", metrics[hp]))
plt.legend(keys, loc="lower right")
plt.savefig("tuning.png")
# plt.show()
