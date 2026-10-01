# 模型修改
## DQN / DDQN 
- 修改为使用和PPO一样的Reward公式
- 增加maskable DQN / maskable DDQN / repair DQN / repair DDQN 
[已添加，正在验证]

@2026.10.01

# 论文修改
## 叙述部分
VM -> containers [已修改]




# 奖励公式修改

奖励公式

$$ R_{\text{term}} =
\begin{cases}
M\cdot AR/AR^{*} \in (0, M], & \text{无违规} \\[2pt]
-M,(1-\text{valid}/M) \in [-M, 0), & \text{有违规}
\end{cases} $$

## ChatGPT公式研究
$$ \[
r_t=
\begin{cases}
1+\beta \Delta AR_t, & \Delta AR_t>\epsilon,\\[4pt]
\beta \Delta AR_t, & |\Delta AR_t|\leq \epsilon,\\[4pt]
-\lambda+\beta \Delta AR_t, & \Delta AR_t<-\epsilon,\\[4pt]
-C, & \mathcal{A}_{\mathrm{valid}}(s_t)=\varnothing,
\end{cases}
\qquad
\Delta AR_t = AR_{t+1}-AR_t.
\] $$

该奖励函数同时考虑资源利用率变化的方向和幅度。当当前放置使平均资源利用率提高时，给予基础正奖励 \(+1\)，并根据实际提升幅度增加 \(\beta\Delta AR_t\)；当资源利用率基本不变时，只保留变化幅度项；当资源利用率下降时，给予基础惩罚 \(-\lambda\)，并根据下降幅度进一步降低奖励。对于 Maskable PPO，违反容量或隐私约束的动作在动作选择前已通过 action mask 排除，因此 reward 不再负责惩罚单个非法动作；只有当仍有 service 未完成放置、但当前不存在任何合法动作时，才给予失败惩罚 \(-C\) 并终止当前 episode。


[等待验证]


# 杂项修改
- 针对Repair-*结果的时候，不需要记录和计算Repair的触发率

## 网站
- 前端需要能够切换分支查看数据，和后端联动吧。
- 前端UI做好手机端适配
