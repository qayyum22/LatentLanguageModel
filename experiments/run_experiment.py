# Script to run experiments


from training.train import main

def run_experiment():
    print("Starting Experiment...")
    main()
    print("Experiment Finished.")

if __name__ == "__main__":
    run_experiment()