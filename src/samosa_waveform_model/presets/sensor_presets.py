# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

import yaml
from typing import Tuple, Dict
from samosa_waveform_model import __RESOURCE_DIR__
from samosa_waveform_model.datamodels import SensorParameters, SARParameters


class SensorPresets(object):
    """
    This class provides preset sensor parameters for the SAMOSA waveform model.
    It includes predefined configurations for different sensor types, such as typical SAR sensors.
    """
    def __init__(self, presets: Dict[Tuple[str, str], Tuple[SensorParameters, SARParameters]]) -> None:
        self.presets = presets


    @classmethod
    def from_package(cls):
        """

        :return:
        """
        # Get list of preset files in the resource directory
        preset_files = list(__RESOURCE_DIR__.glob("sensor_presets/*.yaml"))
        presets = {}
        for preset_file in preset_files:

            # Read the content of the preset file and parse it into a SensorPresetFile object
            with open(preset_file, 'r') as f:
                preset_dict = yaml.safe_load(f.read())

            for radar_mode in preset_dict['radar_modes']:

                # Include additional information about the platform, sensor, and mode in the sensor parameters
                platform_mode_dict = dict(
                    platform=preset_dict['platform'],
                    sensor=preset_dict['sensor'],
                    mode=radar_mode['name']
                )
                radar_mode["sensor_parameter"].update(**platform_mode_dict)
                radar_mode.update(dict(platform=preset_dict['platform'], sensor=preset_dict['sensor']))

                # Store the sensor and SAR parameters in the presets dictionary, keyed by (platform, radar_mode)
                presets[(preset_dict['platform'], radar_mode['name'])] = (
                    SensorParameters(**radar_mode['sensor_parameter']),
                    SARParameters(**radar_mode['sar_parameter'])
                )
        return cls(presets=presets)

    def get_preset(self, sensor_type: str, sensor_name: str) -> Tuple[SensorParameters, SARParameters]:
        """
        Retrieve the preset sensor parameters for a given sensor type and name.

        Args:
            sensor_type (str): The type of the sensor (e.g., "SAR").
            sensor_name (str): The name of the sensor (e.g., "Sentinel-1").

        Returns:
            Tuple[SensorParameters, SARParameters]: The corresponding preset parameters.
        """
        return self.presets.get((sensor_type, sensor_name), None)


SENSORS_PRESETS = SensorPresets.from_package()
breakpoint()
