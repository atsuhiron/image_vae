import json


class Key:
    def __init__(self):
        with open("keys.json", "r") as f:
            key_dict = json.load(f)

        self.ds_url = key_dict["ds_url"]
        self.ds_service_name = key_dict["ds_service_name"]
        self.ds_password = key_dict["ds_password"]
        self.ds_remote_name = key_dict["ds_remote_name"]
        self.ds_user_name = key_dict["ds_user_name"]


key = Key()
