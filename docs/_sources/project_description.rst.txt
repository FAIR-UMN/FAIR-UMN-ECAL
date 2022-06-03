Project Description
==================

Introduction
----------------------------------

The Compact Muon Solenoid (CMS) detector is a general purpose detector designed to detect particles produced in the proton-proton collisions. The proton beams are provided by the Large Hadron Collider (LHC) and the collision occur at a center-of-mass energy as high as $\sqrt{s}=14$ TeV. The CMS detector (Figure $ $) is 21.6 m in length, has a radius of 7.5 m and weighs around 14,600 T. It consists of several layers of sub-detectors modules, each module designed to perform specific measurement. The inner most layer consists of tracker- pixels and silicon strips. The next two layers are the Calorimeters- electromagnetic (ECAL) and hadronic (HCAL). These layers are embedded within a solenoid capable of generating a magnetic field of 3.8 T. The outermost layers are the Muon systems, which consists of several layers of gas detectors. The information from all the sub-detectors is used for particle identification and measurement of their physical properties. The calorimeters play a vital role in the measurement of the energies of the particles produced in the collisions such as electrons, photons and hadrons. Neutrinos produced during the collisions travel through the entire assembly without being detected. The presence of neutrinos can be characterized by missing component of the energy in transverse direction, called Missing Transverse Energy (MET). The calorimeters provide a hermatic coverage for the measurement of MET. 

.. image:: images/cms_experiment.png
   :width: 400
   :align: center

.. centered::
	*An overview of the CMS detector. The figure shows the arrangement of different sub-detector components inside the assembly.*

The CMS Coordinate System
------------------------------------

The point of collision is considered to be the center of the coordinate system. The Y axis points vertically upward whereas the X axis points to the center of the LHC ring and the Z axis points along the direction of the beam. For practical purposes, it is useful to use the radial coordinates R. radial distance from the beam line, :math:`\theta`, the polar angle with respect to the Z axis and $\phi$, the azimuthal angle. Particles produced in the collisions tend to be boosted along the Z direction. Hence, one can define rapidity y as:

.. math::
	
	y = \dfrac{1}{2}\ln{\bigg{(}\dfrac{E+p_{z}c}{E-p_{z}c}\bigg{)}}

Where E is the energy of the particle and pz is the component of the particle momentum along the z axis.The advantage of using rapidity instead of the polar angle is that the rapidity differences are invariant under the boost along the Z axis. In the ultra relativistic limit, the rapidity of the particle can be approximated to pseudo rapidity $\eta$

.. math::
	
	\eta = -\ln{\tan{\bigg{(}\dfrac{\theta}{2}\bigg{)}}}


The pseudorapidity doesn’t require precise measurement of either E or the momentum as opposed to the rapidity. Hence, the four momentum of the particle can be defined 

.. math::
	
	(E,p_{x},p_{y},p_{z}) = (m_{T}\cosh{\eta}, p_{T}\cos{\phi}, p_{T}\sin{\phi}, m_{T}\sinh{\eta})


Where the transverse mass mT is defined as $\sqrt{p_{T}^2+m^2}$ and $p_{T}=\sqrt{p_{x}^2+p_{y}^2}$

.. image:: images/cms_coordinate_system.png
   :width: 400
   :align: center

.. centered::
	*The CMS coordinate system*


Luminosity
------------------

Luminosity is defined as the rate of particles passing through a cross section area. In the context of the LHC, the luminosity, also known and instantaneous luminosity, is given by

.. math::
	
	\mathcal{L} = \dfrac{N_{1}N_{2}fN_{b}}{4\pi\sigma_{x}\sigma_{y}}


Where $N_{1}$ and $N_{2}$ are the number of particles per bunches, f is the revolution frequency and $N_{b}$ is the number of bunches in the ring. The sizeof the bunches along x and y can be characterized by the standard deviation of the gaussian distribution of their positions- $\sigma_{x}$ and $\sigma_{y}$ respectively. One can define integrated luminosity as the integral of instantaneous luminosity over a given period of time:

.. math::
	
	L = \int_{0}^{T}\mathcal{L}dt


Hence for a given physical process, the number of its occurrences in the proton-proton collision can be given by

.. math::
	
	N_{events} = L\sigma


Where $\sigma$ is the cross section of the physics process.

The luminosity at the CMS is calibrated using separation scans called "van der Meer scans" which allow for determination of the absolute luminosity as a function of beam parameters which in turn allows for the calibration of the individual luminometers. These scans are performed once per calibration system and year, and the correction factors and uncertainties corresponding to the luminosity measurements are calculated for every year.

The LHC was designed to operate at a center-of-mass energy of 14 TeV energy with an instantaneous luminosity peaking at $10^{34}$ cm$^{-2}$s$^{-1}$. During the Run II era, the maximum luminosity recorded was $2.06\times10^{34}$ at 13 TeV.

A LHC beam consists of around 2500 bunches of protons, each bunch containing ~O($10^11$) protons at an energy of 6.5 TeV\cite{lhc_lumi_report}. Two beam circulating in opposite directions cross at various interaction points around the LHC- one of them being at the site of the CMS detector. A CMS event corresponds to the data recorded in crossing of one bunch. A luminosity block corresponds to the a collection of temporarily consecutive events, which is roughly 22 s. A CMS Run consists of a collection of several luminosity blocks. The instantaneous luminosity peaks at one point during the LHC beam cycle and gradually decreases with an exponential trend until the cycle ends. A CMS Runs starts and ends during such LHC cycle and typically lasts for several hours. 







Key Features
------------------
GRANSO is among the first optimization packages that can handle general nonsmooth NCVX problems with nonsmooth constraints (Curtis et al., 2017):

.. math::

   \min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x}), \text{ s.t. } c_i(\mathbf{x}) \leq 0, \forall i \in \mathcal{I};\ c_i(\mathbf{x}) = 0, \forall i \in \mathcal{E}

Here, the objective f and constraint functions ci’s are only required to be almost everywhere continuously differentiable. GRANSO is based on quasi-Newton updating with sequential quadratic programming (BFGS-SQP), and has the following advantages:

	#. Unified Treatment of NCVX problems: no need to distinguish CVX vs NCVX and smooth vs nonsmooth problems, similar to typical nonlinear programming packages; 

	#. Reliable Step Size Rule: specialized methods for nonsmooth problems, such as subgradient and proximal methods, often entail tricky step-size tuning and require the expertise to recognize the structures. By contrast, GRANSO chooses step sizes adaptively via gold standard line search; 
	
	#. Principled Stopping Criterion: GRANSO stops its iteration by	checking a theory-grounded stationarity condition for nonsmooth problems, whereas specialized methods are usually stopped when reaching ad-hoc iteration caps.

However, GRANSO users must derive gradients analytically2 and then provide code for
these computations, a process which may require some expert knowledge, is often error-prone, and in machine learning, is generally impractical, e.g., for the training of large neural networks. Furthermore, as part of the MATLAB software ecosystem, GRANSO is generally
hard for practitioners to integrate it with existing popular machine learning frameworks—
mostly in Python and R—and users’ own existing toolchains. To overcome these issues and
facilitate both high performance and ease of use in machine and deep learning, we introduce
a new software package called NCVX, whose initial release contains the solver PyGRANSO, a
PyTorch-enabled port of GRANSO with several new key features: 

	#. auto-differentiation of all gradients, which is a main feature to make PyGRANSO user-friendly, 

	#. support for both CPU and GPU computations for improved hardware acceleration and massive parallelism,

	#. support for general tensor variables including vectors and matrices, as opposed to the single vector of concatenated optimization variables that GRANSO uses, 

	#. integrated support for OSQP (Stellato et al., 2020) and other QP solvers for respectively computing search directions and the stationarity measure on each iteration. OSQP generally outperforms commercial QP solvers in terms of scalability and speed. 

All of these enhancements are
crucial for machine learning researchers and practitioners to solve large-scale problems.
NCVX, licensed under the AGPL version 3, is built entirely on freely available and widely
used open-source frameworks; see https://ncvx.org for documentation and examples

Road Map
----------------------------------

Although NCVX already has many powerful features, we plan to further improve it by adding
several major components:

	#. Symmetric Rank One (SR1): SR1, another major type of quasi-Newton methods, allows less stringent step size search and tends to help escape from saddle points faster by taking advantage of negative curvature directions;
	
	#. Stochastic Algorithms: in machine learning, computing with large-scale datasets often involves finite sums with huge number of terms, calling for (mini-batch) stochastic algorithms for reduced per-iteration cost and better scalability; 
	
	#. Conic Programming (CP): semidefinite programming and second-order cone programming, special cases of CP, are abundant in machine learning, e.g., kernel machines;
	
	#. MiniMax Optimization (MMO): MMO is an emerging technique in modern machine learning, e.g., generative adversarial networks (GANs) and multi-agent reinforcement learning.


References
-----------------

*[1] Buyun Liang, Tim Mitchell, and Ju Sun, NCVX: A User-Friendly and Scalable Package for Nonconvex Optimization in Machine Learning, arXiv preprint arXiv:2111.13984 (2021).* Available at https://arxiv.org/abs/2111.13984

*[2] Frank E. Curtis, Tim Mitchell, and Michael L. Overton, A BFGS-SQP method for nonsmooth, nonconvex, constrained optimization and its evaluation using relative minimization profiles, Optimization Methods and Software, 32(1):148-181, 2017.* Available at https://dx.doi.org/10.1080/10556788.2016.1208749

