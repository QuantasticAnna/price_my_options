# Price My Options
A brief description or tagline about your project.

## Description
This project is designed to [solve a specific problem / provide a specific service]. Key features include:
- Feature 1
- Feature 2


## Installation

1. Clone the repository
   ```
   git clone https://github.com/QuantasticAnna/price_my_options.git
   ```

2. Navigate to the project directory
   ```
   cd price_my_options
   ```

3. Create a virtual environment
   ```
   python -m venv venv
   ```

4. Activate the virtual environment
    - On Windows:
    ```
    venv\Scripts\activate
    ```
    - On macOS/Linux:
    ```
    source venv/bin/activate
    ```

5. Install the project in editable mode
    ```
    pip install -e .
    ```

## Usage

Run the following command:
```
python main.py
```

Then go to [http://127.0.0.1:8050](http://127.0.0.1:8050) on your browser.

## Project Documentation

![Architecture Overview](architecture.PNG)



## Folder Structure

```
price_my_options/
├── app_folder/
│   ├── __init__.py
│   ├── app.py
│   ├── components_model_div.py
│   ├── components.py
│
├── documentation/
│   ├── architecture.PNG
│   ├── info_msg.py
│   ├── README_models_option_pricing.md
│   ├── README_models_stock_price.md
│   ├── README_option_types.md
│   └── README.md
│
├── greeks/
│   ├── __init__.py
│   ├── delta.py
│   ├── gamma.py
│   ├── greeks_functions.py
│   ├── greeks_map.py
│   ├── rho.py
│   ├── theta.py
│   └── vega.py
│
├── precomputed_data/
│   ├── __init__.py
│   ├── data_precomputed.joblib
│   ├── precompute_data.py
│   ├── precomputed_greeks_vs_stock_price_results.joblib
│   ├── precomputed_greeks_vs_strike_price_results.joblib
│   └── precomputed_greeks_vs_ttm_results.joblib
│
├── pricer_plotter/
│   ├── __init__.py
│   ├── asian.py
│   ├── barrier.py
│   ├── binary.py
│   ├── cliquet.py
│   ├── european.py
│   ├── lookback.py
│   ├── monte_carlo.py
│   ├── range.py
│
├── .gitignore
├── constants.py
├── custom_templates.py
├── requirements.txt
└── setup.py
```



## Draft, to delete 

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
