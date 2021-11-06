### Practical example for quantum PCA

> Ref: Quantum Algorithm Implementations for Beginners. arXiv:1804.03719.

Suppose the following two features, number of bedrooms and square footage, corresponding to each of the 15 houses on sale

$$
\begin{aligned}
&X_{1}=(4,3,4,4,3,3,3,3,4,4,4,5,4,3,4) \\
&X_{2}=(3028,1365,2726,2538,1318,1693,1412,1632,2875,3564,4412,4444,4278,3064,3857)
\end{aligned}
$$

1. Compute the covariance matrix.
2. Design a quantum circuit to perform quantum PCA.


(1) Dividing the square footage by 1000 and subtracting off the mean of both features, we have
$$
\begin{aligned}
&X_1 \leftarrow X_1-\mathbb{E}[X_1],\\
&X_2 \leftarrow \frac{X_2-\mathbb{E}[X_2]}{1000}.
\end{aligned}
$$


```python
import numpy as np

X_1 = [4,3,4,4,3,3,3,3,4,4,4,5,4,3,4]
X_2 = [3028,1365,2726,2538,1318,1693,1412,1632,2875,3564,4412,4444,4278,3064,3857]
X_1 = X_1 - np.average(X_1)
X_2 = (X_2 - np.average(X_2)) / 1000
print('The rescaled feature vectors are')
print('X_1 = ', X_1)
print('X_2 = ', X_2)
```

    The rescaled feature vectors are
    X_1 =  [ 0.33333333 -0.66666667  0.33333333  0.33333333 -0.66666667 -0.66666667
     -0.66666667 -0.66666667  0.33333333  0.33333333  0.33333333  1.33333333
      0.33333333 -0.66666667  0.33333333]
    X_2 =  [ 0.21426667 -1.44873333 -0.08773333 -0.27573333 -1.49573333 -1.12073333
     -1.40173333 -1.18173333  0.06126667  0.75026667  1.59826667  1.63026667
      1.46426667  0.25026667  1.04326667]


Thus, the covariance matrix can be written as
$$
M=\left(\begin{array}{ll}
\mathbb{E}\left[X_{1} X_{1}\right] & \mathbb{E}\left[X_{1} X_{2}\right] \\
\mathbb{E}\left[X_{2} X_{1}\right] & \mathbb{E}\left[X_{2} X_{2}\right]
\end{array}\right)
=\frac{1}{15-1}\left(\begin{array}{ll}
X_1^T X_1 & X_1^T X_2 \\
X_2^T X_1 & X_2^T X_2
\end{array}\right).
$$


```python
M=np.array([[np.dot(X_1,X_1),np.dot(X_1,X_2)],[np.dot(X_2,X_1),np.dot(X_2,X_2)]]) / (15-1)
print('The covariance matrix is', 'M = \n', M)
```

    The covariance matrix is M = 
     [[0.38095238 0.57347619]
     [0.57347619 1.29693364]]


Note that we exploit the convention of $(n-1)$ rather than $n$ on the definition of the covariance matrix here.


(2) To diagonalize $M$ using quantum PCA, we exploit a specific workflow as follows.

1. Classical pre-processing. Normalize into density matrix.
2. State preparation. Purify the density matrix into a pure state of enlarged system.
3. Purity measurement. Hadamard test with the controlled-SWAP gate on two copies.
4. Classical post-processing. Transform the purity to eigenvalues.


Step 1. Normalization.
$$
\rho=\frac{M}{\operatorname{Tr}(M)}.
$$


```python
rho = M / np.trace(M)
print('The density matrix is \n', rho)
```

    The density matrix is 
     [[0.22704306 0.34178495]
     [0.34178495 0.77295694]]


Step 2. Purification. Rigorously, we should design a state preparation circuit for this purification. But here, we directly calculate it and focus on the quantum PCA process.


```python
rho_eig_val, rho_eig_vec = np.linalg.eig(rho)
p_vec = np.concatenate((np.sqrt(rho_eig_val), np.sqrt(rho_eig_val)), axis=0)
U_vec = rho_eig_vec.reshape((4))
psi = p_vec * U_vec
print('The purified state is \n', psi)
rho_partial_trace = np.dot(psi.reshape((2,2)),psi.reshape((2,2)).transpose())
print('Verify the reduction to the original mixed state \n', rho_partial_trace)
```

    The purified state is 
     [-0.22545283 -0.41977861  0.10847494 -0.8724621 ]
    Verify the reduction to the original mixed state 
     [[0.22704306 0.34178495]
     [0.34178495 0.77295694]]


Step 3. Purity measurement. Since the matrix is just $2\times 2$, we can exploit a more efficient algorithm instead of the original exponential SWAP method, using only 5 qubits in total. 

Note that the degree of freedom of determining the eigenvalues of the single qubit density matrix is equal to 1, due to the unit trace condition. Thus, as long as we obtain the purity of the mixed state
$$
P=\operatorname{Tr}(\rho^2),
$$
we can determine the two eigenvalues $p_1,p_2$ by
$$
\left\{\begin{aligned}
&p_1^2+p_2^2= P \\
&p_1 + p_2 = 1
\end{aligned}\right\}
\Leftrightarrow 
\left\{\begin{aligned}
&p_1p_2= (1-P)/2 \\
&p_1 + p_2 = 1
\end{aligned}\right\}
\Leftrightarrow 
p_{1,2}=\frac{1\pm\sqrt{2P-1}}{2}.
$$


The purity is equal to the expectation value of the SWAP gate between two purification copies. This can be verified directly as follows.
$$
\begin{aligned}
& \rho=\sum_i p_i \left| a_i\right\rangle \left\langle a_i \right|, 
\left| \psi \right\rangle=\sum_i \sqrt{p_i} \left|a_i\right\rangle \left| b_i \right\rangle \\
 \Rightarrow& P=\operatorname{Tr}(\rho^2)=\sum_i p_i^2, \\
& \left\langle\psi\right| \left\langle\psi\right| \operatorname{SWAP_a} \left| \psi \right\rangle \left| \psi \right\rangle
= \sum_{ij} \left\langle b_j\right| \left\langle a_j\right|  \left\langle b_i\right| \left\langle a_i\right|
\sqrt{p_ip_j}\sqrt{p_ip_j} 
\left|a_j\right\rangle \left| b_i \right\rangle \left|a_i\right\rangle \left| b_j \right\rangle
= \sum_i p_i^2 \\
\Rightarrow & \operatorname{Tr}(\rho^2)=\left\langle\psi\right| \left\langle\psi\right| \operatorname{SWAP_a} \left| \psi \right\rangle \left| \psi \right\rangle.
\end{aligned}
$$

Therefore, the purity can be measured using the Hadamard test on a controlled-SWAP gate. The corresponding circuit is shown below. 

> Quantum circuit of the 5-qubit quantum PCA. This picture is reprinted from arXiv:1804.03719.


![Quantum circuit for the 5-qubit quantum PCA.](https://raw.githubusercontent.com/Haokai-Zhang/ExampleQPCA/master/qPCA_5qubits.png)


We can use Qiskit to simulate this circuit.


```python
from qiskit import QuantumCircuit, execute, Aer, assemble

circ = QuantumCircuit(5, 1)
circ.initialize([1,0], (0,))
circ.initialize(psi, (1,2))
circ.initialize(psi, (3,4))
circ.h(0)
circ.cswap(0,1,3)
circ.h(0)
circ.measure(0,0)
circ.draw()
```




<pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">                     ┌─────────────────┐                ┌───┐   ┌───┐┌─┐
q_0: ────────────────┤ Initialize(1,0) ├────────────────┤ H ├─■─┤ H ├┤M├
     ┌───────────────┴─────────────────┴───────────────┐└───┘ │ └───┘└╥┘
q_1: ┤0                                                ├──────X───────╫─
     │  Initialize(-0.22545,-0.41978,0.10847,-0.87246) │      │       ║ 
q_2: ┤1                                                ├──────┼───────╫─
     ├─────────────────────────────────────────────────┤      │       ║ 
q_3: ┤0                                                ├──────X───────╫─
     │  Initialize(-0.22545,-0.41978,0.10847,-0.87246) │              ║ 
q_4: ┤1                                                ├──────────────╫─
     └─────────────────────────────────────────────────┘              ║ 
c: 1/═════════════════════════════════════════════════════════════════╩═
                                                                      0 </pre>
```python
# Tell Qiskit how to simulate our circuit
sim = Aer.get_backend('aer_simulator')
# Tell simulator to save statevector
circ.save_statevector()
# Create a Qobj from the circuit for the simulator to run
qobj = assemble(circ)
counts = sim.run(qobj, shots=1e+7).result().get_counts()
print(counts)
```

    {'0': 9412783, '1': 587217}


Step 4. Transform the purity to eigenvalues.


```python
purity = (counts['0'] - counts['1']) / (counts['0'] + counts['1'])
m_1 = (1 + np.sqrt(2 * purity - 1)) / 2 * np.trace(M)
m_2 = (1 - np.sqrt(2 * purity - 1)) / 2 * np.trace(M)
print('The eigenvalues obtained by the quantum PCA are \n', [m_1, m_2])
```

    The eigenvalues obtained by the quantum PCA are 
     [1.572772746964541, 0.10511327208307823]

```python
m, vec = np.linalg.eig(M)
idx = m.argsort()[::-1]
m = m[idx]
vec = vec[:,idx]
print('The eigenvalues obtained by classical diagonalization are \n', m)
```

    The eigenvalues obtained by classical diagonalization are 
     [1.57285742 0.1050286 ]


Therefore, we can see that within the allowed margin of random error, the quantum circuit simulation gives the same result as the classical diagonalization.
