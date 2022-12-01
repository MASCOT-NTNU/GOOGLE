# Notes for Simulation study for long horizon path planning given different cost fields.

---

## RRTStar convergence study

---

The following parameters can be adjusted in the simulation study.

- **step size**
- **number of iterations**
- **goal sampling rates**
- **neighbour radius**
- **home radius**
- **distance between start and end locations**

The total distance reduction can be used for the convergence rate study.
- **trajectory distance reduction rate**

Two cases to study
- no cost field.
- add cost field.

---


Result discussion

#### Stepsize:
- increasing step size does not necessarily guarantee the convergence.
- the distance does not reduce sometimes.
- computation time does not change significantly.

#### Maximum expansion iterations:
- dod odo
