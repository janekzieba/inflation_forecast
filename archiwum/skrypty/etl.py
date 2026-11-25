from graphviz import Digraph


def create_etl_diagram():
    # Ustawienia ogólne grafu
    dot = Digraph('ETL_Architecture', comment='Inflation Prediction ETL')
    dot.attr(rankdir='LR', size='12,8', ratio='fill')

    # Style węzłów
    dot.attr('node', shape='box', style='filled', fontname='Helvetica', fontsize='10')

    # --- 1. ŹRÓDŁO DANYCH (EXTRACT) ---
    with dot.subgraph(name='cluster_source') as c:
        c.attr(label='Źródło Danych (Extract)', color='lightgrey', style='dashed')
        c.node('NBP', 'NBP API / Portal\n(Statystyka)', shape='ellipse', fillcolor='#e1f5fe', color='#0277bd')
        c.node('RawXLS', 'Plik Źródłowy\n(bazowa.xlsx)', shape='note', fillcolor='#fff9c4')

    # --- 2. PRZETWARZANIE (TRANSFORM) ---
    with dot.subgraph(name='cluster_processing') as c:
        c.attr(label='Przetwarzanie i Modelowanie (Transform)', color='grey')
        c.node('DataSource', 'data_source.py\n(Czyszczenie & ETL)', shape='component', fillcolor='#ffe0b2',
               color='#ef6c00')
        c.node('FeatureEng', 'Inżynieria Cech\n(Lags, Rolling Mean)', shape='component', fillcolor='#ffe0b2',
               color='#ef6c00')

        # Modele
        c.node('ARIMA', 'arima.py\n(AutoARIMA)', shape='box', style='rounded,filled', fillcolor='#c8e6c9',
               color='#2e7d32')
        c.node('ML', 'ml_forecast_10y.py\n(XGBoost / RF)', shape='box', style='rounded,filled', fillcolor='#c8e6c9',
               color='#2e7d32')

    # --- 3. MAGAZYN DANYCH (LOAD) ---
    with dot.subgraph(name='cluster_storage') as c:
        c.attr(label='Warstwa Danych (Load)', color='grey')
        c.node('CleanCSV', 'cpi_data.csv\n(Dane Historyczne)', shape='folder', fillcolor='#f0f4c3')
        c.node('ForecastCSV', 'forecast_*.csv\n(Wyniki Prognoz)', shape='folder', fillcolor='#f0f4c3')
        c.node('ValCSV', 'validation_data.csv\n(Metryki Błędów)', shape='folder', fillcolor='#f0f4c3')

    # --- 4. PREZENTACJA (APP) ---
    with dot.subgraph(name='cluster_app') as c:
        c.attr(label='Warstwa Prezentacji (Application)', color='lightgrey')
        c.node('App', 'app.py\n(Streamlit Dashboard)', shape='tab', fillcolor='#d1c4e9', color='#4527a0')
        c.node('User', 'Użytkownik Końcowy\n(Analiza/Symulacja)', shape='ellipse', fillcolor='white')

    # --- RELACJE (KRAWĘDZIE) ---

    # Extract
    dot.edge('NBP', 'RawXLS', label=' Pobranie')
    dot.edge('RawXLS', 'DataSource', label=' Odczyt')

    # Transform
    dot.edge('DataSource', 'CleanCSV', label=' Zapis oczyszczonych')
    dot.edge('CleanCSV', 'FeatureEng', label=' Ładowanie')
    dot.edge('FeatureEng', 'ARIMA')
    dot.edge('FeatureEng', 'ML')

    # Load (Model Outputs)
    dot.edge('ARIMA', 'ForecastCSV')
    dot.edge('ML', 'ForecastCSV')
    dot.edge('ML', 'ValCSV', style='dashed', label=' Backtesting')

    # Consume
    dot.edge('CleanCSV', 'App')
    dot.edge('ForecastCSV', 'App')
    dot.edge('ValCSV', 'App')

    # User interaction
    dot.edge('App', 'User', label=' Wizualizacja')

    # Renderowanie
    output_path = 'diagram_przeplywu_etl'
    dot.render(output_path, format='png', view=False)
    print(f"Diagram zapisano jako: {output_path}.png")


if __name__ == '__main__':
    create_etl_diagram()