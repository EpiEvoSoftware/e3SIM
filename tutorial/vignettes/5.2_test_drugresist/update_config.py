import json, os

def update_dir():
    """
    Update the test.json for testing installation.
    """
    # config
    wkdir = os.getcwd()
    config_path = os.path.join(wkdir, "test_config_drugresist.json")
    config = open(config_path, "r")
    config_dict = json.loads(config.read())
    # wkdir
    config_dict["BasicRunConfiguration"]["cwdir"] = wkdir
    # data path for config json
    data_path = os.path.join(os.path.dirname(wkdir), "../e3SIM-main/test_installation/data", "COVID", "EPI_ISL_402124.fasta")
    config_dict["GenomeElement"]["ref_path"] = data_path

    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

def main():
    update_dir()

if __name__ == "__main__":
    main()


    

