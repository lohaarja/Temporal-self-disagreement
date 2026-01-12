import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def generate_dataset(n_samples=5000, n_features=20, n_classes=3):
    rng = np.random.RandomState(42)

    X = rng.randn(n_samples, n_features)
    y = np.zeros(n_samples, dtype=int)


    group_A = slice(0, 5)
    group_B = slice(5, 10)
    group_C = slice(10, 15)

    for i in range(n_samples):
        choice = rng.choice([0, 1, 2])

        if choice == 0:
            X[i, group_A] += 3.0
            y[i] = 0
        elif choice == 1:
            X[i, group_B] += 3.0
            y[i] = 1
        else:
            X[i, group_C] += 3.0
            y[i] = 2

 
    trap_count = n_samples // 5
    for i in range(trap_count):
        idx = i
        X[idx, group_A] += 3.0
        X[idx, group_B] += 3.0
        y[idx] = rng.choice([0, 1])  
    X = StandardScaler().fit_transform(X)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class SelfDisagreeNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.task_head = nn.Linear(hidden_dim, num_classes)

        
        self.stability_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.truth_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.mode_head = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        z = self.encoder(x)

        logits = self.task_head(z)
        probs = F.softmax(logits, dim=-1)

        confidence, _ = probs.max(dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

        p_flip = self.stability_head(z).squeeze(-1)
        truth_conf = self.truth_head(z).squeeze(-1)
        mode = self.mode_head(z)

        return logits, probs, confidence, entropy, p_flip, truth_conf, mode


def perturb(x, std=0.05):
    return x + torch.randn_like(x) * std


@torch.no_grad()
def detect_flip(model, x):
    """
    Measures temporal self-disagreement:
    does the model change its mind under perturbation?
    """
    p1 = model(x)[0].argmax(dim=-1)
    p2 = model(perturb(x))[0].argmax(dim=-1)
    return (p1 != p2).float()


class HonestyMemory:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.H = 1.0

    def update(self, p_flip):
        
        self.H = (1 - self.alpha) * self.H + self.alpha * (1 - p_flip.mean().item())
        self.H = float(np.clip(self.H, 0.3, 1.0))

    def modulate(self, conf):
        return conf * self.H


@torch.no_grad()
def simulate_future_selves(model, x, K=6):
    """
    Monte Carlo dropout to simulate future selves.
    """
    model.train()

    confs, preds = [], []

    for _ in range(K):
        logits, _, c, _, _, _, _ = model(x)
        confs.append(c)
        preds.append(logits.argmax(dim=-1))

    return torch.stack(confs).var(0), torch.stack(preds).float().var(0)


def future_lie_risk(conf, conf_var, pred_var):
    """
    Overconfident commitment under instability.
    """
    return conf * (conf_var + pred_var)


def loss_fn(
    logits, conf, entropy, p_flip, truth_conf,
    y, flip, conf_var, pred_var, memory, force_pressure
):
    task = F.cross_entropy(logits, y)
    stab = F.binary_cross_entropy(p_flip, flip)

    if force_pressure:
        conf = torch.clamp(conf + 0.3, max=1.0)

    lie_score = conf * p_flip * truth_conf
    future = future_lie_risk(conf, conf_var, pred_var).mean()

    memory.update(lie_score)
    truth_target = (logits.argmax(dim=-1) == y).float().detach()

    truth_loss = F.binary_cross_entropy(truth_conf, truth_target)


    total = (
    task
    + stab
    + 0.5 * lie_score.mean()
    + 0.5 * future
    + 0.1 * truth_loss          
    + 0.1 * (1 - memory.H)
)


    return total, future.item(), memory.H



def train(model, X, y, epochs=40, batch=64):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    memory = HonestyMemory()

    for e in range(epochs):
        perm = torch.randperm(X.size(0))
        for i in range(0, X.size(0), batch):
            xb, yb = X[perm[i:i+batch]], y[perm[i:i+batch]]

            logits, _, conf, ent, p_flip, truth, _ = model(xb)
            flip = detect_flip(model, xb)
            cv, pv = simulate_future_selves(model, xb)

            loss, fr, trust = loss_fn(
                logits, conf, ent, p_flip, truth,
                yb, flip, cv, pv, memory,
                force_pressure=np.random.rand() < 0.3
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

        if e % 5 == 0:
            print(f"Epoch {e:03d} | Trust={trust:.3f} | FutureRisk={fr:.3f}")

    return memory


def counterfactual_explanation(conf, risk, p_flip, truth):
    """
    Explains refusal in counterfactual terms.
    """
    conditions = []

    if risk > 0.2:
        conditions.append("future-self risk were lower")
    if p_flip > 0.05:
        conditions.append("my future self agreed with me")
    if truth < 0.6:
        conditions.append("my internal truth signal were stronger")

    if not conditions:
        return "No epistemic hesitation detected."

    return "I would answer if " + " and ".join(conditions) + "."


@torch.no_grad()
def evaluate(model, memory, x):
    logits, _, conf, ent, p_flip, truth, mode = model(x)
    cv, pv = simulate_future_selves(model, x)
    risk = future_lie_risk(conf, cv, pv)

 
    memory.update(p_flip)

    failure = [
        classify_failure(conf[i], truth[i], p_flip[i])
        for i in range(len(conf))
    ]

    return {
        "pred": logits.argmax(dim=-1),
        "confidence": conf,
        "truth_conf": truth,
        "p_flip": p_flip,
        "future_risk": risk,
        "effective_conf": memory.modulate(conf),
        "mode": mode.argmax(dim=-1),
        "failure_type": failure
    }


def analyze_danger(out, conf_t=0.65, risk_t=0.15):
    conf, risk, flip, truth = (
        out["confidence"], out["future_risk"],
        out["p_flip"], out["truth_conf"]
    )

    danger = (conf > conf_t) & (risk > risk_t)

    print("\nDANGER ZONE ANALYSIS — PREEMPTIVE EPISTEMIC REFUSAL")
    print(f"Samples: {danger.sum().item()}")

    for i in torch.where(danger)[0]:
        ftype = out["failure_type"][i]
        print(
            f"[{i.item()}] class={out['pred'][i].item()} "
            f"conf={conf[i]:.3f} risk={risk[i]:.3f} "
            f"truth={truth[i]:.3f} p_flip={flip[i]:.3f}"
        )
        print(f" → Failure type: {ftype}")
        print(" → Counterfactual:",
              counterfactual_explanation(conf[i], risk[i], flip[i], truth[i]))


def classify_failure(conf, truth, p_flip,
                     conf_t=0.65,
                     truth_t=0.55,
                     flip_t=0.15):
    if conf > conf_t and truth < truth_t and p_flip < flip_t:
        return "HALLUCINATION"
    if conf > conf_t and truth >= truth_t and p_flip >= flip_t:
        return "LIE-RISK"
    if conf < conf_t:
        return "UNCERTAIN"
    return "SAFE"


@torch.no_grad()
def ambiguity_sweep(model, memory, x, steps=10):
    confs, risks, flips = [], [], []
    for i in range(steps):
        out = evaluate(
            model,
            memory,
            x + torch.randn_like(x) * (i / steps) * 0.3
        )
        confs.append(out["confidence"].mean().item())
        risks.append(out["future_risk"].mean().item())
        flips.append(out["p_flip"].mean().item())
    return confs, risks, flips

def accuracy_with_refusal(out, y_true, conf_t=0.65, risk_t=0.15):
    keep = ~((out["confidence"] > conf_t) & (out["future_risk"] > risk_t))
    return (out["pred"][keep] == y_true[keep]).float().mean().item()

def accuracy_without_refusal(out, y_true):
    return (out["pred"] == y_true).float().mean().item()


X, y = generate_dataset()
model = SelfDisagreeNet(X.shape[1], 128, len(torch.unique(y)))
memory = train(model, X, y)

sample = X[:32]
out = evaluate(model, memory, sample)


analyze_danger(out)
acc_no_refuse = accuracy_without_refusal(out, y[:32])
acc_refuse = accuracy_with_refusal(out, y[:32])

print(f"\nAccuracy without refusal: {acc_no_refuse:.3f}")
print(f"Accuracy with refusal:    {acc_refuse:.3f}")

c, r, f = ambiguity_sweep(model, memory, sample)
plt.plot(c, label="Confidence")
plt.plot(r, label="Future Risk")
plt.plot(f, label="p_flip")
plt.legend()
plt.title("Adversarial Ambiguity Sweep (ETD)")
plt.show()

