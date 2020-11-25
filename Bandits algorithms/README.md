# Bandits algorithms, Practical session 3
In this second practical, you are asked to put what you just learnt
about bandits to good use. You are provided with the `main.py` file,
a bandits test bed. Use `python main.py -h` to check how you are
supposed to use this file.

You will implement:
* Epsilon-greedy bandit
* UCB1 https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf
* Besa https://hal.archives-ouvertes.fr/hal-01025651v1/document
* softmax https://www.cs.mcgill.ca/~vkules/bandits.pdf
* Thompson sampling Agent https://en.wikipedia.org/wiki/Thompson_sampling
* KL UCB https://hal.archives-ouvertes.fr/hal-00738209v2 (optional)


![image not found:](multiarmedbandit.jpg "Bandits")



## How do I complete these files ?
Remove the expection raising part, and
complete the two blank methods for each Agent.

In `__init__`, build the buffers your agent requires.
It might be interesting, for instance, to store the
number of time each action has been selected.

In `choose`, prescribe how the agent selects its
actions (interact must return an action, that is
an index in [0, ..., 9]).

Finally, in update, implement how the agent updates
its buffers, using the newly observed `action` and `reward`.


## D) How do I proceed to be evaluated ?

You will be noted:
* Implementation of the 6 agents (but KL-UCB optional) in the `agents.py` file. Bonus points will be given to clean, scalable code.
* Answering this question -> for each implemented agent, give 1 pros and 1 cons ?


Send `agents.py` and answers to heri(at)lri(dot)fr before November, 25th 2020 at 23:59.
