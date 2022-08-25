#!/usr/bin/python

import json

import xmltodict


def get_dict_from_xml(xml_path):
    with open(xml_path, "r") as file:
        # prefixes @/_ lead to issues with pydantic parsing! so simply use no prefix
        xml_dict = xmltodict.parse(file.read(), attr_prefix="")
        return xml_dict


def get_dict_from_json(path):
    with open(path, "r") as file:
        return json.load(file)


if __name__ == "__main__":
    """convert XML to JSON"""
    import sys

    assert len(sys.argv) == 2
    xml_path = sys.argv[1]
    json_path = xml_path[:-3] + "json"

    print(f"converting {xml_path} to {json_path}")

    xml_dict = get_dict_from_xml(xml_path)
    with open(json_path, "w") as outfile:
        json.dump(xml_dict, outfile)
