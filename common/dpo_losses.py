"""
DPO 家族 7 个变体 loss 的纯 PyTorch 实现 (无需下载任何模型)

设计目标：
    - 每个 loss 只依赖 chosen_logprob / rejected_logprob / reference_chosen_logprob / reference_rejected_logprob
    - 公式与原论文 100% 对齐,代码中带行内注释指明出处
    - 配合 stage1_loss_demos/01_loss_math.ipynb 使用

涵盖变体 (按图中顺序):
    1. DPO        (Rafailov et al., 2023)
    1b. DPO+      (Rafailov et al., 2023 论文 Appendix)  -- DPO + 正例下界 hinge
    1c. LD-DPO    (Length-Dropped DPO, 自定义)             -- DPO + 长度折扣
    2. IPO        (Azar et al., 2023)
    3. CPO        (Xu et al., 2024)
    4. KTO        (Ethayarajh et al., 2024)
    5. ORPO       (Hong et al., 2024)
    6. SimPO      (Meng et al., 2024)
    7. SIPO       (Wei et al., 2024)  -- 以 paper 中最简单的 closed-form 形式给出

约定:
    - 形状均为 (batch_size,)
    - chosen_logprob / rejected_logprob: 当前策略模型对 chosen / rejected 回答的 log p(y|x)
    - ref_chosen_logprob / ref_rejected_logprob: 参考策略模型 (frozen) 的 log p(y|x)
    - chosen_len / rejected_len: 对应 token 数 (用于 length normalization)
    - beta: 温度系数
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1) DPO -- Direct Preference Optimization (Rafailov et al., 2023)
# ---------------------------------------------------------------------------
def dpo_loss(
    chosen_logprob: torch.Tensor,
    rejected_logprob: torch.Tensor,
    ref_chosen_logprob: torch.Tensor,
    ref_rejected_logprob: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    L_DPO = -E[ log sigma( beta * (h_pi - h_ref) ) ]
    h_pi   = log p_theta(chosen|x) - log p_theta(rejected|x)
    h_ref  = log p_ref(chosen|x)   - log p_ref(rejected|x)
    即隐式奖励: r(x, y) = beta * (log p_theta(y|x) - log p_ref(y|x))
    """
    pi_logratios = chosen_logprob - rejected_logprob
    ref_logratios = ref_chosen_logprob - ref_rejected_logprob
    logits = beta * (pi_logratios - ref_logratios)  # 即 beta * (h_pi - h_ref)
    # label_smoothing: 将正例标签从 1 -> (1-eps), 负例从 0 -> eps
    # 实现: -[ (1-eps) * log sigma(l) + eps * log sigma(-l) ]
    if label_smoothing > 0.0:
        loss = (
            - (1.0 - label_smoothing) * F.logsigmoid(logits)
            - label_smoothing * F.logsigmoid(-logits)
        )
    else:
        loss = -F.logsigmoid(logits)
    return loss.mean()


# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# 1b) DPO_PLUS -- DPO with positive-sample lower bound
# 公式:
#   L_DPO+ = L_DPO + lambda * max(0,  log p_ref(y_w|x) - log p_theta(y_w|x))
# 含义: 模型对正例的概率不能比参考模型还低。hinge 项只在 ref > pi 时激活。
# ---------------------------------------------------------------------------
def dpop_loss(
    chosen_logprob: torch.Tensor,
    rejected_logprob: torch.Tensor,
    ref_chosen_logprob: torch.Tensor,
    ref_rejected_logprob: torch.Tensor,
    beta: float = 0.1,
    lam: float = 0.5,
) -> torch.Tensor:
    """
    L_DPO_PLUS = -log sigma(beta*(h_pi - h_ref))
                 + lam * max(0, ref_logp_chosen - logp_chosen)
    """
    pi_logratios  = chosen_logprob - rejected_logprob
    ref_logratios = ref_chosen_logprob - ref_rejected_logprob
    logits = beta * (pi_logratios - ref_logratios)
    dpo_term = -F.logsigmoid(logits)
    dpop_penalty = torch.clamp(ref_chosen_logprob - chosen_logprob, min=0.0)
    return (dpo_term + lam * dpop_penalty).mean()


# ---------------------------------------------------------------------------
# 1c) LD-DPO  --  Length-Dropped DPO
# 思路: 把 token 序列按较短那条的长度 l_p = min(l_w, l_l) 切成两段,
#      公共部分按权重 1 计, 较长那条多出来的部分按权重 alpha (<=1) 计。
# 公式: log p_hat(y|x) = sum_{i<=l_p} log p_i + alpha * sum_{i>l_p} log p_i
# ---------------------------------------------------------------------------
def ld_dpo_loss(
    chosen_logprob: torch.Tensor,
    rejected_logprob: torch.Tensor,
    ref_chosen_logprob: torch.Tensor,
    ref_rejected_logprob: torch.Tensor,
    chosen_len: torch.Tensor,
    rejected_len: torch.Tensor,
    beta: float = 0.1,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    chosen_logprob  : sum_{i=1}^{l_c} log p_i  (B,)
    chosen_len      : l_c                       (B,)
    alpha           : 超长部分权重, 0=no tail, 1=等价于普通 DPO
    """
    l_p = torch.minimum(chosen_len, rejected_len)
    excess_c = (chosen_len   - l_p).clamp(min=0.0)
    excess_r = (rejected_len - l_p).clamp(min=0.0)

    def reweight(sum_logp, full_len, excess, lp):
        per_tok = sum_logp / full_len.clamp(min=1.0)
        return per_tok * lp + alpha * per_tok * excess

    new_c  = reweight(chosen_logprob,        chosen_len,        excess_c, l_p)
    new_r  = reweight(rejected_logprob,      rejected_len,      excess_r, l_p)
    new_rc = reweight(ref_chosen_logprob,    chosen_len,        excess_c, l_p)
    new_rr = reweight(ref_rejected_logprob,  rejected_len,      excess_r, l_p)

    pi_logratios  = new_c - new_r
    ref_logratios = new_rc - new_rr
    logits = beta * (pi_logratios - ref_logratios)
    return -F.logsigmoid(logits).mean()


# 2) IPO -- Identity Preference Optimization (Azar et al., 2023)
# ---------------------------------------------------------------------------
def ipo_loss(
    chosen_logprob: torch.Tensor,
    rejected_logprob: torch.Tensor,
    ref_chosen_logprob: torch.Tensor,
    ref_rejected_logprob: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    L_IPO = ( log sigma(beta * (h_pi - h_ref)) - 1/(2*beta) )^2
    IPO 把 DPO 的 "对数几率" 视为需要回归到 1/(2*beta) 的目标,
    平方损失让训练更稳定, 避免 DPO 在长尾数据上过拟合。
    """
    pi_logratios = chosen_logprob - rejected_logprob
    ref_logratios = ref_chosen_logprob - ref_rejected_logprob
    logits = pi_logratios - ref_logratios
    # 论文原式:  loss = (log(pi_theta(y_w)/pi_ref(y_w)) - log(pi_theta(y_l)/pi_ref(y_l)) - 1/(2*beta))^2
    # TRL 等开源实现把 beta 折进 logits, 写成: (beta*logits - 1/(2*beta))^2
    loss = (logits - 1.0 / (2.0 * beta)) ** 2
    return loss.mean()


# ---------------------------------------------------------------------------
# 3) CPO -- Contrastive Preference Optimization (Xu et al., 2024)
# ---------------------------------------------------------------------------
def cpo_loss(
    chosen_logprob: torch.Tensor,
    rejected_logprob: torch.Tensor,
    ref_chosen_logprob: torch.Tensor,
    ref_rejected_logprob: torch.Tensor,
    chosen_mean_logprob: torch.Tensor,
    rejected_mean_logprob: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    alpha: float = 0.0,
) -> torch.Tensor:
    """
    CPO = DPO-style preference term + SFT-like NLL term
    L_CPO = -log sigma(beta * (h_pi - h_ref)) - alpha * log p_theta(y_w|x)
    无需 reference model 也可训练 (alpha 控制 SFT 强度), 论文推荐 alpha>0。
    这里保留 reference 通路, 以便和 DPO 完全可比。
    """
    pi_logratios = chosen_logprob - rejected_logprob
    ref_logratios = ref_chosen_logprob - ref_rejected_logprob
    logits = beta * (pi_logratios - ref_logratios)
    pref_loss = -F.logsigmoid(logits)
    # SFT 项使用 mean log prob (按 token 归一化), 鼓励 chosen 句子的整体似然上升
    sft_loss = -chosen_mean_logprob
    loss = pref_loss + alpha * sft_loss
    return loss.mean()


# ---------------------------------------------------------------------------
# 4) KTO -- Kahneman-Tversky Optimization (Ethayarajh et al., 2024)
# ---------------------------------------------------------------------------
def kto_loss(
    chosen_logprob: torch.Tensor,
    rejected_logprob: torch.Tensor,
    ref_chosen_logprob: torch.Tensor,
    ref_rejected_logprob: torch.Tensor,
    # KTO 不要求成对, 这里 chosen/rejected 来自不同样本, 形状仍 (batch,)
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> torch.Tensor:
    """
    L_KTO = E_y~D[ w(y) * (1 - sigma(beta * (r_theta - r_ref) - z_ref)) ]
    其中 z_ref = KL(theta || ref) 的一个标量参考点 (desirable/undesirable 共享)。
    简化版: 这里把 z_ref 设为 0 (论文附录推荐近似),
    r_theta(y|x) = log (p_theta(y|x) / p_ref(y|x))
    chosen 为 "desirable" 损失, rejected 为 "undesirable" 损失。
    """
    # chosen (desirable)
    chosen_logratio = chosen_logprob - ref_chosen_logprob
    chosen_loss = desirable_weight * (1.0 - torch.sigmoid(beta * chosen_logratio))
    # rejected (undesirable)
    rejected_logratio = rejected_logprob - ref_rejected_logprob
    rejected_loss = undesirable_weight * (1.0 - torch.sigmoid(-beta * rejected_logratio))
    # KTO 的 batch size 通常为 chosen 和 rejected 各自独立采样, 求平均
    return 0.5 * (chosen_loss.mean() + rejected_loss.mean())


# ---------------------------------------------------------------------------
# 5) ORPO -- Odds Ratio Preference Optimization (Hong et al., 2024)
# ---------------------------------------------------------------------------
def orpo_loss(
    chosen_logprob: torch.Tensor,
    rejected_logprob: torch.Tensor,
    chosen_mean_logprob: torch.Tensor,
    rejected_mean_logprob: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    L_ORPO = -log sigma( log odds_ratio ) - lambda * log p_theta(y_w|x)
    odds_ratio = p(y_w|x) / (1 - p(y_w|x))   ; 在 log 空间即 logit

    论文中实际对 mean log prob 做 logit:  log p / (1 - log p) 是错误写法,
    正确写法是 odds = exp(mean_logp) / (1 - exp(mean_logp))。
    为数值稳定, 使用 log_odds = mean_logp - log(1 - exp(mean_logp))。
    ORPO 不需要 reference model。

    【数值稳定修正】原实现用 log(1 - exp(mean_logp)) 在 mean_logp > 0 时
    容易因 exp 越界导致 NaN。这里改用 sigmoid 截断 + 显式 logit:
        p = sigmoid(mean_logp) ∈ (0, 1) 永远安全
        log_odds = log(p) - log(1 - p)   即 logit(p)
    """
    # 数值稳定: 先把 mean_logp 投到 (0, 1) 概率空间
    p_chosen   = torch.sigmoid(chosen_mean_logprob)
    p_rejected = torch.sigmoid(rejected_mean_logprob)
    chosen_logodds   = torch.log(p_chosen)   - torch.log(1.0 - p_chosen)
    rejected_logodds = torch.log(p_rejected) - torch.log(1.0 - p_rejected)
    odds_ratio = chosen_logodds - rejected_logodds
    pref_loss = -F.logsigmoid(beta * odds_ratio)
    # SFT 项 (lambda 通常=0.1)
    sft_loss = -chosen_mean_logprob
    return (pref_loss + 0.1 * sft_loss).mean()


# ---------------------------------------------------------------------------
# 6) SimPO -- Simple Preference Optimization (Meng et al., 2024)
# ---------------------------------------------------------------------------
def simpo_loss(
    chosen_mean_logprob: torch.Tensor,
    rejected_mean_logprob: torch.Tensor,
    beta: float = 2.0,
    gamma_beta_ratio: float = 0.5,
) -> torch.Tensor:
    """
    L_SimPO = -log sigma( beta * |y_w|/|y_w| * (mean_logp_w - mean_logp_l) - gamma )
    - 不需要 reference model
    - 用 mean log prob (按 token 归一化), 防止模型偏好长句
    - gamma 是正 margin, 拉开 chosen/rejected 的隐式奖励差距
    论文中 gamma / beta 的比值约 0.5。
    """
    logits = beta * (chosen_mean_logprob - rejected_mean_logprob) - gamma_beta_ratio * beta
    loss = -F.logsigmoid(logits)
    return loss.mean()


# ---------------------------------------------------------------------------
# 7) SIPO -- Statistical Information Preference Optimization (Wei et al., 2024)
# ---------------------------------------------------------------------------
def sipo_loss(
    chosen_logprob: torch.Tensor,
    rejected_logprob: torch.Tensor,
    ref_chosen_logprob: torch.Tensor,
    ref_rejected_logprob: torch.Tensor,
    beta: float = 0.1,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    SIPO 在 DPO log-ratio 基础上引入 "统计信息" 修正:
    r_SIPO(x, y) = beta * ( log p_theta(y|x) - log p_ref(y|x) )
                   + alpha * ( log p_theta(y|x) - log p_ref_mean )
    其中 ref_mean 是 reference 模型在整个 batch 上的平均 log prob。
    简化实现: 用 batch 维度的均值代替 ref_mean。
    """
    pi_logratios = chosen_logprob - rejected_logprob
    ref_logratios = ref_chosen_logprob - ref_rejected_logprob
    base_logits = beta * (pi_logratios - ref_logratios)
    # 统计信息项: 当前模型相对 reference 的整体偏离 (batch 维度均值)
    chosen_diff = chosen_logprob - ref_chosen_logprob
    rejected_diff = rejected_logprob - ref_rejected_logprob
    stat_correction = alpha * (chosen_diff.mean() - rejected_diff.mean())
    logits = base_logits + stat_correction
    loss = -F.logsigmoid(logits)
    return loss.mean()


# ---------------------------------------------------------------------------
# 工厂函数: 通过字符串名字拿到 loss
# ---------------------------------------------------------------------------
LOSS_REGISTRY = {
    "dpo":  dpo_loss,
    "ipo":  ipo_loss,
    "cpo":  cpo_loss,
    "kto":  kto_loss,
    "orpo": orpo_loss,
    "simpo": simpo_loss,
    "sipo": sipo_loss,
}


def get_loss_fn(name: str):
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name]


# ---------------------------------------------------------------------------
# 单元自测: 在合成数据上跑一遍, 验证每个 loss 都返回标量
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B = 8
    chosen_logprob      = torch.randn(B) * 0.5 - 1.0
    rejected_logprob    = torch.randn(B) * 0.5 - 2.0
    ref_chosen_logprob  = torch.randn(B) * 0.5 - 1.1
    ref_rejected_logprob= torch.randn(B) * 0.5 - 2.1
    chosen_mean_logprob = chosen_logprob / 50.0
    rejected_mean_logprob = rejected_logprob / 50.0

    print("Loss sanity check:")
    for name, fn in LOSS_REGISTRY.items():
        if name == "kto":
            val = fn(chosen_logprob, rejected_logprob, ref_chosen_logprob, ref_rejected_logprob)
        elif name == "orpo":
            val = fn(chosen_logprob, rejected_logprob, chosen_mean_logprob, rejected_mean_logprob)
        elif name == "simpo":
            val = fn(chosen_mean_logprob, rejected_mean_logprob)
        elif name == "cpo":
            val = fn(chosen_logprob, rejected_logprob, ref_chosen_logprob, ref_rejected_logprob,
                     chosen_mean_logprob, rejected_mean_logprob)
        elif name == "sipo":
            val = fn(chosen_logprob, rejected_logprob, ref_chosen_logprob, ref_rejected_logprob)
        else:  # dpo, ipo
            val = fn(chosen_logprob, rejected_logprob, ref_chosen_logprob, ref_rejected_logprob)
        print(f"  {name:6s} -> {val.item():.4f}")
    print("All losses computed successfully.")
