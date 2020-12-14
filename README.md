# Implementation of Reinforcement Learning Algorithm 
PyTorch를 기반한 Reinforcement Learning (RL) 알고리즘 튜토리얼

## Project Structure
본 프로젝트의 구조는 moemen95's GitHub [Pytorch-Project-Template](https://github.com/moemen95/Pytorch-Project-Template)를 기반으로 작성되어 있습니다. 
```
├── agents
|  └── dqn.py # the main training agent for the dqn
├── graphs
|  └── models
|  |  └── dqn.py
|  |  └── actor-crtic.py
|  |  └── REINFORCE.py
|  |  └── ....
|  └── losses
|  |  └── huber_loss.py # contains huber loss definition
├── datasets  # contains all dataloaders for the project
├── utils # utilities folder containing input extraction, replay memory, config parsing, etc
|  └── assets
|  └── replay_memory.py
|  └── env_utils.py
├── main.py
└── run.sh
```

![](https://github.com/moemen95/Pytorch-Project-Template/raw/master/utils/assets/class_diagram.png)

---

## Environments
아래와 같은 환경에서만 테스트하였습니다. 환경은 추후 추가될 예정입니다.
- CartPole
- GridWorld
- ...

---

## Usage

```
python3 main.py
```

---

## Requirements
- Python: 3.7
- PyTorch: 
- 

---

## REFERENCE
### Deep Learning Template
- victoresque's GitHub [pytorch-template](https://github.com/victoresque/pytorch-template) repo
- FrancescoSaverioZuppichini's GitHub [PyTorch-Deep-Learning-Template](https://github.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Template) repo
- moemen95's GitHub [Pytorch-Project-Template](https://github.com/moemen95/Pytorch-Project-Template) repo
### Reinforcement Learning Algorithm
- seungeunrho's GitHub [minimalRL](https://github.com/seungeunrho/minimalRL) repo
- seungeunrho's GitHub [RLfrombasics](https://github.com/seungeunrho/RLfrombasics) repo

---

## License
<img align="right" src="http://opensource.org/trademarks/opensource/OSI-Approved-License-100x137.png">

The class is licensed under the [MIT License](http://opensource.org/licenses/MIT):

Copyright (c) 2020 **Cognitive Systems Lab**

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

