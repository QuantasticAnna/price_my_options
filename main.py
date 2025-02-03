from app_folder.app import app  

def run():
    """
    Entry point for running the application.
    """
    app.run_server(debug=False, host="0.0.0.0", port=8050)

if __name__ == "__main__":
    run()