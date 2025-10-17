import json, os

def update_dir():
    """
    Update the test.json for testing installation.
    """
    # config
    wkdir = os.getcwd()
    config_path = os.path.join(wkdir, "test_config.json")
    config = open(config_path, "r")
    config_dict = json.loads(config.read())
    # wkdir
    config_dict["BasicRunConfiguration"]["cwdir"] = wkdir
    # data path for config json
    data_path = os.path.join(os.path.dirname(wkdir), "data", "TB", "GCF_000195955.2_ASM19595v2_genomic.fna")
    config_dict["GenomeElement"]["ref_path"] = data_path
    seed_path = os.path.join(wkdir, "slimseeds.csv")

    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

def main():
    update_dir()

if __name__ == "__main__":
    main()


    

