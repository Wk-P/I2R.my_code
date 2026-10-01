# Reward Design Summary

我们的目标是让不同 RL 方法共享同一个 **objective reward**，从而保证比较时优化目标一致；不同方法之间只改变 constraint-handling mechanism，而不是改变基础 reward。

定义平均资源利用率变化为：

$$ \[
\Delta AR_t = AR_{t+1}-AR_t
\] $$

统一 objective reward 设计为：

$$ \[
r_t^{\mathrm{obj}}=
\begin{cases}
1+\beta\Delta AR_t, & \Delta AR_t>\epsilon,\\[4pt]
\beta\Delta AR_t, & |\Delta AR_t|\leq\epsilon,\\[4pt]
-\lambda+\beta\Delta AR_t, & \Delta AR_t<-\epsilon.
\end{cases}
\] $$

这个设计同时保留两个信息。离散项 $ \(+1,0,-\lambda\) $ 提供明确的方向性学习信号，使模型能够快速区分资源利用率改善、不变和下降；连续项 $ \(\beta\Delta AR_t\) $ 则保留实际变化幅度，使得两个同样改善 AR 的动作仍然可以根据改善程度获得不同 reward。因此该 reward 属于一种 directional reward shaping，而不是单纯使用 $ \(\Delta AR_t\) $。

不再使用类似

$$ \[
-\lambda(N-t)
\] $$

的剩余 timestep 惩罚，因为这会人为加入“越早下降惩罚越大”的时间偏好。动作对未来 placement 的长期影响本身已经可以通过 DQN 中的 \(Q\)-value，或 PPO 中的 return、value 和 advantage 向前传播，因此不需要再次通过 remaining steps 重复编码这种影响。

对于完整任务成功与失败，需要与普通 step reward 分开处理。当所有 services 均成功放置时，可以增加 completion bonus：

$$ \[
r_T \leftarrow r_T+B.
\] $$

如果当前仍有未放置 service，但已经不存在任何可行 placement，则认为 episode 进入 dead end，可以给予：

$$ \[
r_t=-C
\] $$ 

并终止当前 episode。这里 $ \(B\) $ 和 $ \(C\) $ 的作用是区分完整 solution 与 partial solution，而不是直接优化 AR。

对于不同 constraint-handling 方法，基础 objective reward 保持一致。Maskable PPO、Maskable DQN 和 Maskable DDQN 通过合法动作集合

$$ \[
\mathcal A_{\mathrm{valid}}(s_t)
\] $$ 

直接排除违反 capacity 或 privacy constraint 的动作，因此训练 reward 可以直接使用：

$$ \[
r_t^{\mathrm{train}}=r_t^{\mathrm{obj}}.
\] $$

如果出现

$$ \[
\mathcal A_{\mathrm{valid}}(s_t)=\varnothing
\] $$

且任务尚未完成，则使用 failure penalty $ \(-C\) $ 。

Lagrangian PPO、DQN 和 DDQN 则另外定义 constraint cost：

$$ \[
c_t
\] $$

并在统一 objective reward 的基础上构造：

$$ \[
\tilde r_t
=
r_t^{\mathrm{obj}}
-
\eta c_t,
\] $$

其中 $ \(\eta\) $ 为 Lagrangian multiplier。这样 objective reward 和 constraint penalty 保持独立。

Heuristic PPO、DQN 和 DDQN 也继续使用相同的：

$$ \[
r_t^{\mathrm{obj}}.
\] $$

Heuristic 只负责 action filtering、correction 或辅助选择，不改变资源利用率目标本身。因此不同方法之间的比较可以统一理解为：

$$ \[
\boxed{
\text{Same Objective Reward}
+
\text{Different Constraint-Handling Mechanism}
}
\] $$

整体设计原则可以概括为：

$$ \[
\boxed{
\text{Constraint 决定动作能不能执行，Reward 决定合法决策做得好不好。}
}
\] $$ 

因此论文实验中应尽量保持 environment、objective reward 和 evaluation metric 一致，只改变 Heuristic、Lagrangian、Maskable 等 constraint-handling mechanism，从而避免 reward design 本身成为不同算法性能差异的额外混杂因素。