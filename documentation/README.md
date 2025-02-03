Draft README:

Workflow: 

- One function monte_carlo_simulation
- One pricer per exotic option category, with a paramter call / put 
    --> we prices all the possible exotic options on the same monte carlo simulation, so we can compare the options (because they are relative to the same 'stock'), and we dont ahve to re run expensive computation 
- One function to plot the simulation paths
- One plotter for each option categore, that add specificities to the basic option plotter
    --> ex: for asian, we add in addition to the simulation, the avg of each simulation 
    --> ex: for lookback, we add in addition to the simulation, the max / min of each simulation 
    ?. Maybe always make the plot for a high number of simulation (ex 100), then on Dash make a slider that allows user to choose between 0 and 100

--> With this logic, it is very easy to add new types of options


# Project Documentation

![Architecture Overview](architecture.png)