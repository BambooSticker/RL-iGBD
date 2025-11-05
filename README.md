# Learning to control inexact Benders decomposition via reinforcement learning

<!-- ## Abstract -->
- **Goal:** Benders decomposition (BD), along with its generalized version (GBD), is a widely used algorithm for solvinglarge-scale mixed-integer optimization problems that arise in the operation of process systems. However, theoff-the-shelf application to online settings can be computationally inefficient due to the repeated solution of the master problem. An approach to reduce the solution time is to solve the master problem to local optimality. However, identifying the level of suboptimality at each iteration that minimizes the total solution time is nontrivial.

- **Approach:** In this work, we train a reinforcement learning agent to determine the best optimality gap schedule that minimizes the total solution time, balancing the solution time per iteration with optimality gap improvement. 

- **Paper Link:** https://www.sciencedirect.com/science/article/pii/S0098135425004648

## Experiments & Reproduction
- `notebooks/GBD.ipynb`: [vanilla GBD implementation for the integration of scheduling and dynamic optimization problem]
- `notebooks/RL_training.ipynb`: [agent training]
- `notebooks/RL_test_episode.ipynb`: [evaluate the agent on a single instance, comparing per-iterate solution time]
- `notebooks/RL_test_stats.ipynb`: [evaluate the agent over 50 randomly generated instances]

## Repository Layout
```text
.
├── src/                         
│   ├── GurobiParamEnv.py        
│   ├── dynamic_opt.py           
│   ├── schedule_opt.py          
│   ├── policies.py              
│   ├── GurobiParamEnv_test.py   
│   └── tools.py                  
├── notebooks/                   
│   ├── GBD.ipynb                
│   ├── RL_training.ipynb        
│   ├── RL_test_episode.ipynb    
│   └── RL_test_stats.ipynb      
├── data/                        
│   ├── test_episode_data.csv    
│   └── test_stats_data.csv      
├── models/                      
│   ├── ppo_benders_model.zip
│   └── vecnormalize_benders.pkl
├── LICENSE                      
└── README.md                    
```

## Citation
If you use this repository in your research, please cite:
```bibtex
@article{li2025learning,
  title={Learning to control inexact Benders decomposition via reinforcement learning},
  author={Li, Zhe and Agyeman, Bernard T and Mitrai, Ilias and Daoutidis, Prodromos},
  journal={Computers \& Chemical Engineering},
  pages={109461},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledgements
- The partial financial support of NSF CBET (award number 2313289)is gratefully acknowledged. 
- IM would like to acknowledge financialsupport from the McKetta Department of Chemical Engineering.

## Contact
(c) 2025 Daoutidis Lab

For questions, feel free to open an issue or reach out directly.
