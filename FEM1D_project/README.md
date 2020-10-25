# IN5270

## Project2 : Solving a 1D Poisson equation with the finite element method (P2 elements)

### Overview of files

In this directory you encounter three files.

* mainFEM.py
* p2_finite_elem.py
* FEM1D_project.ipynb

The Jupyter-notebook file contains a report summarizing the work done and the main results of the project. All the code produced is also available in the notebook, so strictly speaking the two python files in this directory are redundant. They are included for readability and reproducibility purposes.

The p2_finite_elem.py file contains the implemented class solver and the mainFEM.py file is used to test the implemented solver for different number of elements and boundary conditions.

### Reproducing results

If you want to reproduce results for any of the tasks in the project, simply write in your terminal

```console
python3 mainFEM.py number_of_elements
```

where number_of_elements is an integer number Ne > 3.

If you want to look at the convergence rate of the solution as a function of Number of elements, simply write "L2" instead of number_of_element like this:

```console
python3 mainFEM.py L2
```
If you wish to change either the values of the boundary conditions, C and D, or the number of gridpoints you wish to evaluate your approximation in, it can be done by simply changing their values in the top of the mainFEM.py file.
