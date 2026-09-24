# Source of CryoSat-2 lookup tables

The lookup tables here originally come from [SAMPy](https://github.com/cls-obsnadir-dev/SAMPy)

## Renamed Tables

The orginal files have been renamed and reformatted for dynamic file loading in this project.

The original filenames are listed below:


| New Filename | Original Filename |
|-------------------|--------------|
| `alpha_power_cryosat2_hamming.csv` | `alphap_table_DX3000_ZP20_SWH20_10_Sept_2019(CS2_HAMMING).txt` |
| `alpha_power_cryosat2_nohamming.csv` | `alphap_table_DX3000_ZP20_SWH20_10_Sept_2019(CS2_NO_HAMMING).txt` |


## Removed Tables

The file `alphaPower_table_CONSTANT_SWH20_10_Feb_2020(CS2_NOHAMMING).txt` has been removed 
as it only contained a constant value, which can now be found in the sensor configuration
class. 
