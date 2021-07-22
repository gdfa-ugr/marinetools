# MarineTools

**MarineTools** is a framework that integrates software that can be used in the search for solutions to real engineering and marine problems. Like Python and most of the packages developed by the scientific community, *MarineTools.temporal* is an open-source software. In this work we present the *temporal* package aimed at providing users with a friendly, general code to statistically characterize a vector random process (RP) to obtain realizations of it. It is implemented in Python - an interpreted, high-level, object-oriented programming language widely used in the scientific community - and it makes the most of the Python packages ecosystem. Among the existing Python packages, it uses Numpy, which is the fundamental package for scientific computing in Python [["1"]](#1), SciPy, which offers a wide range of optimization and statistics routines [["2"]](#2), Matplotlib [["3"]](#3), that includes routines to obtain high-quality graphics, and Pandas [["4"]](#4) to analyse and manipulate data.

The tools implemented in the package named *temporal* allow to capture the statistical properties of a **non stationary (NS) vector RP** by using **compound or piecewise parametric PMs** to properly describe all the range of values and to **simulate uni- or multivariate time series** with the same random behavior. The statistical parameters of the distributions are assumed to depend on time and are expanded into a Generalized Fourier Series (GFS) [["5"]](#5) in order to reproduce their NS behavior. The applicability of the present approach has been illustrated in several works with different purposes, among others: (i) the observed wave climate variability in the preceding century and expected changes in projections under a climate change scenario [["6"]](#6); (ii) the optimal design and management of an oscillating water column system [["7"]](#7) [["8"]](#8), (iii) the planning of maintenance strategies of coastal structures [["9"]](#9), (iv) the analysis of monthly Wolf sunspot number over a 22 year period [["5"]](#5), and (v) the simulation of estuarine water conditions for the management of the estuary [["10"]](#10).

In the **example folder** can be found 7 Jupyter Notebooks. Each one described how to set-up the environment to run the code and how to use the main functions included in *MarineTools.temporal*.

The **Environmental Fluid Dynamics** team of the University of Granada whishes a good experience in learning process. Enjoy it!


## References
<a id="1">[1]</a> 
Harris, Charles R. and Millman, K. Jarrod and
    van der Walt, Stéfan J and Gommers, Ralf and
    Virtanen, Pauli and Cournapeau, David and
    Wieser, Eric and Taylor, Julian and Berg, Sebastian and
    Smith, Nathaniel J. and Kern, Robert and Picus, Matti and
    Hoyer, Stephan and van Kerkwijk, Marten H. and
    Brett, Matthew and Haldane, Allan and
    Fernández del Río, Jaime and Wiebe, Mark and
    Peterson, Pearu and Gérard-Marchant, Pierre and
    Sheppard, Kevin and Reddy, Tyler and Weckesser, Warren and
    Abbasi, Hameer and Gohlke, Christoph and
    Oliphant, Travis E. (2020). 
Array programming with {NumPy}.
Nature.

<a id="2">[2]</a> 
Virtanen, Pauli and Gommers, Ralf and Oliphant, Travis E. and
  Haberland, Matt and Reddy, Tyler and Cournapeau, David and
  Burovski, Evgeni and Peterson, Pearu and Weckesser, Warren and
  Bright, Jonathan and {van der Walt}, Stéfan J. and
  Brett, Matthew and Wilson, Joshua and Millman, K. Jarrod and
  Mayorov, Nikolay and Nelson, Andrew R. J. and Jones, Eric and
  Kern, Robert and Larson, Eric and Carey, C J and
  Polat, Ilhan and Feng, Yu and Moore, Eric W. and
  {VanderPlas}, Jake and Laxalde, Denis and Perktold, Josef and
  Cimrman, Robert and Henriksen, Ian and Quintero, E. A. and
  Harris, Charles R. and Archibald, Anne M. and
  Ribeiro, Antonio H. and Pedregosa, Fabian and
  {van Mulbregt}, Paul and {SciPy 1.0 Contributors} (2020).
{{SciPy} 1.0: Fundamental Algorithms for Scientific
Computing in Python}.
Nature Methods.
  
<a id="3">[3]</a> 
John D. Hunter.
Matplotlib: A 2D Graphics Environment.
Computing in Science & Engineering.

<a id="4">[4]</a> 
McKinney, Wes and others (2010).
Data structures for statistical computing in python.
Proceedings of the 9th Python in Science Conference.

<a id="5">[5]</a> 
Cobos, M. and Otíñar, P. and Magaña, P. and Baquerizo, A. (2021)
A method to characterize and simulate climate, Earth or environmental vector random processes.
Submitted to Probabilistic Engineering and Mechanics.

<a id="6">[6]</a> 
Lira-Loarca, Andrea Lira and Cobos, Manuel and Besio, Giovanni and Baquerizo, Asunción (2021).
Projected wave climate temporal variability due to climate change.
Stochastic Environmental Research and Risk Assessment.

<a id="7">[7]</a> 
Jalón, María L and Baquerizo, Asunción and Losada, Miguel A (2016).
Optimization at different time scales for the design and management of an oscillating water column system.
Energy.

<a id="8">[8]</a> 
López-Ruiz, Alejandro and Bergillos, Rafael J and Lira-Loarca, Andrea and Ortega-Sánchez, Miguel (2018).
A methodology for the long-term simulation and uncertainty analysis of the operational lifetime performance of wave energy converter arrays.
Energy.

<a id="9">[9]</a> 
Lira-Loarca, Andrea and Cobos, Manuel and Losada, Miguel Ángel and Baquerizo, Asunción (2020).
Storm characterization and simulation for damage evolution models of maritime structures.
Coastal Engineering.

<a id="10">[10]</a> 
Cobos, Manuel (2020).
A model to study the consequences of human actions in the Guadalquivir River Estuary.
Tesis Univ. Granada.
