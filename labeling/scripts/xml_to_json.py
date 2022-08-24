#!/usr/bin/python

import xmltodict
import json


def get_dict_from_xml(xml_path):
    with open(xml_path, "r") as file:
        xml_dict = xmltodict.parse(file.read(),attr_prefix="_")
        return xml_dict


if __name__ == "__main__":
    import sys
    assert len(sys.argv) == 2
    xml_path = sys.argv[1]
    json_path = xml_path[:-3] + 'json'

    print(f"converting {xml_path} to {json_path}")

    xml_dict = get_dict_from_xml(xml_path)
    with open(json_path, "w") as outfile:
        json.dump(xml_dict,outfile)


    