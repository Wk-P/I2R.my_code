# 重要问题记录
## 来自教授建议
1. reward function 的变换和解释 
- 现在先做 reward funciton，去掉 M，然后统一reward function.
    - reward function 现在面临惩罚奖励范围不相匹配，奖励项出现[-M, M]，惩罚项出现[-M, 0]。现在需要的是惩罚就是负值，奖励就要正向的结果。
- 其次需要明确一点，没有中途停止，而是全部使用终局奖励和惩罚，坚持M步走完。
2. 放置前顺序的排列问题：当前用 FFD First Fit Desicion，足够说明即可 
3. 最重要 5% 以下的 gap 差异
