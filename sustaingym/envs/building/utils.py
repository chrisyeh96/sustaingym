"""
This module primarily implements the `ParameterGenerator()` function which
generates the parameters dict for `BuildingEnv`.

All of the building layouts, U-factor values (thermal transmittance, in
W/m2-K), ground temperatures, and weather data come from the Building Energy
Codes Program: https://www.energycodes.gov/prototype-building-models.

Buildings models

- HTM files were extracted from the "Individual Standard 90.1 Prototype Building
  Models" (version 90.1-2019)
- We associate each building type with a list of 7 U-factor values (thermal
  transmittance, in W/m2-K) for different surfaces in the building in the
  order: [intwall, floor, outwall, roof, ceiling, groundfloor, window]
  These values are manually compiled from the HTM files, with missing
  values manually filled in from similar building types. For example,
  values missing from OfficeSmall were filled in from OfficeMedium.

Monthly ground temperature values come from the
"Site:GroundTemperature:FCfactorMethod" table in the building HTM files.

Weather data come from EnergyPlus TMY3 Weather Files (in ``*.epw`` format)
also provided by the Building Energy Codes Program.
"""
from __future__ import annotations

from collections.abc import Sequence
from collections import defaultdict
import io
import os
from typing import Any, NamedTuple

import numpy as np
import pvlib
from scipy import interpolate

from sustaingym.data.utils import read_to_stringio


class Ufactor(NamedTuple):
    """Thermal transmittance (in W/m2-K) of different surfaces in a building"""
    intwall: float      # 0
    floor: float        # 1
    outwall: float      # 2
    roof: float         # 3
    ceiling: float      # 4
    groundfloor: float  # 5
    window: float       # 6


BUILDINGS = {
    "ApartmentHighRise": (
        "ASHRAE901_ApartmentHighRise_STD2019_Tucson.table.htm",
        Ufactor(6.299, 3.285, 0.384, 0.228, 3.839, 0.287, 2.786),
    ),
    "ApartmentMidRise": (
        "ASHRAE901_ApartmentMidRise_STD2019_Tucson.table.htm",
        Ufactor(6.299, 3.285, 0.384, 0.228, 3.839, 0.287, 2.786),
    ),
    "Hospital": (
        "ASHRAE901_Hospital_STD2019_Tucson.table.htm",
        Ufactor(6.299, 3.839, 0.984, 0.228, 3.839, 3.285, 2.615),
    ),
    "HotelLarge": (
        "ASHRAE901_HotelLarge_STD2019_Tucson.table.htm",
        Ufactor(6.299, 0.228, 0.984, 0.228, 0.228, 2.705, 2.615),
    ),
    "HotelSmall": (
        "ASHRAE901_HotelSmall_STD2019_Tucson.table.htm",
        Ufactor(6.299, 3.839, 0.514, 0.228, 3.839, 0.1573, 2.615),
    ),
    "OfficeLarge": (
        "ASHRAE901_OfficeLarge_STD2019_Tucson.table.htm",
        Ufactor(6.299, 3.839, 0.984, 0.228, 4.488, 3.839, 2.615),
    ),
    "OfficeMedium": (
        "ASHRAE901_OfficeMedium_STD2019_Tucson.table.htm",
        Ufactor(6.299, 3.839, 0.514, 0.228, 4.488, 0.319, 2.615),
    ),
    "OfficeSmall": (
        "ASHRAE901_OfficeSmall_STD2019_Tucson.table.htm",
        Ufactor(6.299, 3.839, 0.514, 0.228, 4.488, 0.319, 2.615),
    ),
    "OutPatientHealthCare": (
        "ASHRAE901_OutPatientHealthCare_STD2019_Tucson.table.htm",
        Ufactor(6.299, 3.839, 0.514, 0.228, 3.839, 0.5650e-02, 2.615),
    ),
    "RestaurantFastFood": (
        "ASHRAE901_RestaurantFastFood_STD2019_Tucson.table.htm",
        Ufactor(6.299, 0.158, 0.547, 4.706, 0.158, 0.350, 2.557),
    ),
    "RestaurantSitDown": (
        "ASHRAE901_RestaurantSitDown_STD2019_Tucson.table.htm",
        Ufactor(6.299, 0.158, 0.514, 4.706, 0.158, 0.194, 2.557),
    ),
    "RetailStandalone": (
        "ASHRAE901_RetailStandalone_STD2019_Tucson.table.htm",
        Ufactor(6.299, 0.047, 0.984, 0.228, 0.228, 0.047, 3.695),
    ),
    "RetailStripmall": (
        "ASHRAE901_RetailStripmall_STD2019_Tucson.table.htm",
        Ufactor(6.299, 0.1125, 0.514, 0.228, 0.228, 0.1125, 3.695),
    ),
    "SchoolPrimary": (
        "ASHRAE901_SchoolPrimary_STD2019_Tucson.table.htm",
        Ufactor(6.299, 0.144, 0.514, 0.228, 0.228, 0.144, 2.672),
    ),
    "SchoolSecondary": (
        "ASHRAE901_SchoolSecondary_STD2019_Tucson.table.htm",
        Ufactor(6.299, 3.839, 0.514, 0.228, 3.839, 0.144, 2.672),
    ),
    "Warehouse": (
        "ASHRAE901_Warehouse_STD2019_Tucson.table.htm",
        Ufactor(0.774, 0.1926, 1.044, 0.5892, 10.06, 0.1926, 2.557),
    ),
}

GROUND_TEMP = {
    "Albuquerque": [13.7, 7.0, 2.1, 2.6, 4.3, 8.8, 13.9, 17.8, 23.2, 25.6, 24.1, 20.5],
    "Atlanta": [16.0, 11.9, 7.7, 4.0, 7.9, 13.8, 17.2, 20.8, 24.8, 26.1, 26.5, 22.5],
    "Buffalo": [9.7, 6.0, -2.2, -3.4, -4.2, 2.7, 7.5, 13.7, 18.6, 22.0, 20.7, 16.5],
    "Denver": [7.1, 3.0, -1.0, 0.8, -0.2, 4.8, 6.1, 13.7, 22.2, 22.7, 21.7, 18.5],
    "Dubai": [29.5, 25.5, 21.1, 19.2, 20.8, 23.1, 26.5, 31.4, 33.0, 35.1, 35.3, 32.5],
    "ElPaso": [18.3, 11.2, 6.8, 8.1, 10.3, 12.5, 19.2, 23.8, 27.9, 27.5, 26.3, 23.4],
    "Fairbanks": [-3.1, 17.7, 19.3, 17.6, 15.4, 10.3, 0.7, 10.6, 16.0, 16.9, 14.2, 6.7],
    "GreatFalls": [8.6, 2.8, 4.1, 8.8, 2.2, 0.3, 6.7, 10.1, 16.5, 20.6, 19.2, 14.7],
    "HoChiMinh": [26.9, 26.7, 26.0, 26.4, 27.5, 28.3, 29.2, 29.0, 28.9, 27.2, 27.5, 27.6],
    "Honolulu": [26.2, 24.8, 23.7, 22.5, 22.8, 23.2, 23.8, 25.2, 25.9, 26.9, 27.1, 26.9],
    "InternationalFalls": [5.4, 2.0, 14.6, 16.9, 11.5, 6.2, 4.0, 13.4, 18.0, 19.7, 17.9, 12.3],
    "NewDelhi": [25.1, 19.6, 14.5, 13.4, 17.0, 22.4, 29.1, 33.0, 33.6, 31.7, 30.0, 28.7],
    "NewYork": [14.0, 7.3, 3.3, 1.2, -0.2, 5.6, 10.9, 16.1, 21.7, 25.0, 24.8, 19.9],
    "PortAngeles": [9.3, 6.7, 4.1, 4.2, 4.2, 5.9, 9.0, 10.0, 13.3, 15.0, 15.7, 13.4],
    "Rochester": [7.4, 0.0, 7.6, 12.6, 7.7, 0.3, 7.0, 14.2, 19.2, 20.9, 20.0, 15.4],
    "SanDiego": [18.8, 14.3, 13.6, 13.2, 13.3, 12.6, 15.3, 15.6, 17.7, 19.4, 19.7, 18.5],
    "Seattle": [11.4, 8.1, 5.4, 4.5, 5.8, 8.3, 10.9, 13.0, 15.6, 17.7, 18.8, 15.1],
    "Tampa": [24.2, 18.9, 15.7, 13.6, 15.5, 17.1, 21.2, 26.9, 27.6, 27.9, 27.4, 26.2],
    "Tucson": [20.9, 15.4, 11.9, 14.8, 12.7, 15.4, 23.3, 26.3, 31.2, 30.4, 29.8, 27.8],
}

WEATHER = {
    "Very_Hot_Humid": "USA_HI_Honolulu.Intl.AP.911820_TMY3.epw",
    "Hot_Humid": "USA_FL_Tampa-MacDill.AFB.747880_TMY3.epw",
    "Hot_Dry": "USA_AZ_Tucson-Davis-Monthan.AFB.722745_TMY3.epw",
    "Warm_Humid": "USA_GA_Atlanta-Hartsfield.Jackson.Intl.AP.722190_TMY3.epw",
    "Warm_Dry": "USA_TX_El.Paso.Intl.AP.722700_TMY3.epw",
    "Warm_Marine": "USA_CA_San.Deigo-Brown.Field.Muni.AP.722904_TMY3.epw",
    "Mixed_Humid": "USA_NY_New.York-John.F.Kennedy.Intl.AP.744860_TMY3.epw",
    "Mixed_Dry": "USA_NM_Albuquerque.Intl.Sunport.723650_TMY3.epw",
    "Mixed_Marine": "USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3.epw",
    "Cool_Humid": "USA_NY_Buffalo.Niagara.Intl.AP.725280_TMY3.epw",
    "Cool_Dry": "USA_CO_Denver-Aurora-Buckley.AFB.724695_TMY3.epw",
    "Cool_Marine": "USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw",
    "Cold_Humid": "USA_MN_Rochester.Intl.AP.726440_TMY3.epw",
    "Cold_Dry": "USA_MT_Great.Falls.Intl.AP.727750_TMY3.epw",
    "Very_Cold": "USA_MN_International.Falls.Intl.AP.727470_TMY3.epw",
    "Subarctic/Arctic": "USA_AK_Fairbanks.Intl.AP.702610_TMY3.epw",
}


class Zone(NamedTuple):
    name: str                   # 0
    Zaxis: float                # 1
    Xmin: float                 # 2
    Xmax: float                 # 3
    Ymin: float                 # 4
    Ymax: float                 # 5
    Zmin: float                 # 6
    Zmax: float                 # 7
    FloorArea: float            # 8, in m^2
    ExteriorGrossArea: float    # 9, in m^2
    ExteriorWindowArea: float   # 10, in m^2
    ind: int                    # 11, can't use name "index" because of tuple.index()


def get_zones(
    path_or_file: str | io.TextIOBase,
) -> tuple[list[list[Zone]], int, list[Zone]]:
    """Parses information from the HTM file and sorts each zone by layer.

    Args:
        path_or_file: path to building HTM file, or a file-like object

    Returns:
        layers: Zones grouped by Zaxis, layers[i] is a list of Zones in layer i
        n: number of zones
        all_zones: list of n Zones
    """
    # Initialize lists for storing zone information
    cord: list[str | float] = []
    cordall: list[list[str | float]] = []

    # Read all lines of the html file
    if isinstance(path_or_file, str):
        with open(path_or_file, 'r') as f:
            htmllines = f.readlines()
    elif isinstance(path_or_file, io.TextIOBase):
        htmllines = path_or_file.readlines()
    else:
        raise ValueError(f'Unsupported type for {path_or_file}')

    # Initialize count and printflag variables
    count = 0
    printflag = False

    # Iterate through each line in the html file
    for line in htmllines:
        count += 1

        # Turn off the printflag after the 'Zone info' chart
        if 'Zone Internal Gains Nominal' in line:
            printflag = False

        # Extract information when the printflag is True
        if printflag:
            # Zone_name
            if (count - 35) % 32 == 0 and count != 3:
                cord.append(line[22:-6])
            # Zaxis
            if (count - 42) % 32 == 0 and count != 10:
                cord.append(float(line[22:-6]))
            # Xmin
            if (count - 46) % 32 == 0 and count != 14:
                cord.append(float(line[22:-6]))
            # Xmax
            if (count - 47) % 32 == 0 and count != 15:
                cord.append(float(line[22:-6]))
            # Ymin
            if (count - 48) % 32 == 0 and count != 16:
                cord.append(float(line[22:-6]))
            # Ymax
            if (count - 49) % 32 == 0 and count != 17:
                cord.append(float(line[22:-6]))
            # Zmin
            if (count - 50) % 32 == 0 and count != 18:
                cord.append(float(line[22:-6]))
            # Zmax
            if (count - 51) % 32 == 0 and count != 19:
                cord.append(float(line[22:-6]))
            # FloorArea
            if (count - 56) % 32 == 0 and count != 24:
                cord.append(float(line[22:-6]))
            # ExteriorNetArea
            if (count - 58) % 32 == 0 and count != 26:
                cord.append(float(line[22:-6]))
            # ExteriorWindowArea
            if (count - 59) % 32 == 0 and count != 27:
                cord.append(float(line[22:-6]))

                # Append the current cord to cordall and reset cord
                cordall.append(cord)
                cord = []

        # Set printflag to True when 'Zone Information' is encountered in the line
        if 'Zone Information' in line:
            printflag = True
            count = 0

    # Calculate the total number of zones
    n = len(cordall)

    # Sort cordall by Zaxis
    cordall.sort(key=lambda x: x[1])

    # Convert cordall to Zone NamedTuples
    # see github.com/python/mypy/issues/6799
    all_zones = [Zone(*cord, ind=int(i)) for i, cord in enumerate(cordall)]  # type: ignore

    layers = []
    current_layer = []
    current_zaxis = all_zones[0].Zaxis

    for i, zone in enumerate(all_zones):
        if zone.Zaxis == current_zaxis:
            # If Zaxis remains the same, add the zone to the current_layer
            current_layer.append(zone)
        else:
            # If the Zaxis value changes, add the current_layer to layers
            # and reset the current_layer and current_zaxis variables
            layers.append(current_layer)
            current_layer = [zone]
            current_zaxis = zone.Zaxis

        # If this is the last zone, append the current_layer to layers
        if i == len(all_zones) - 1:
            layers.append(current_layer)

    return layers, n, all_zones


def checkconnect(z1: Zone, z2: Zone) -> bool:
    """Checks whether zones in the same layer are connected."""
    z1_min_in_z2 = (
        z2.Xmin <= z1.Xmin <= z2.Xmax
        and z2.Ymin <= z1.Ymin <= z2.Ymax
    )
    z1_max_in_z2 = (
        z2.Xmin <= z1.Xmax <= z2.Xmax
        and z2.Ymin <= z1.Ymax <= z2.Ymax
    )
    return z1_min_in_z2 or z1_max_in_z2


def checkconnect_layer(z1: Zone, z2: Zone) -> bool:
    """Checks whether zones in different layers are connected."""
    z1_min_in_z2 = (
        z2.Xmin <= z1.Xmin < z2.Xmax
        and z2.Ymin <= z1.Ymin < z2.Ymax
    )
    z1_max_in_z2 = (
        z2.Xmin < z1.Xmax <= z2.Xmax
        and z2.Ymin < z1.Ymax <= z2.Ymax
    )
    return z1_min_in_z2 or z1_max_in_z2


def Nfind_neighbor(
    n: int,
    layers: Sequence[Sequence[Zone]],
    ufactor: Ufactor,
    SpecificHeat_avg: float,
) -> tuple[dict[str, list[int]], np.ndarray, np.ndarray, np.ndarray]:
    """
    This function is for the building model.

    Args:
        n: number of rooms
        layers: sorted layer list
        ufactor: list of 7 U-values (thermal transmittance) for different
            surfaces in the building in the order
            [intwall, floor, outwall, roof, ceiling, groundfloor, window].
        SpecificHeat_avg: specific heat of air (in J/kg-K)

    Returns:
        neighbors: maps zone name to a list of neighboring zone indices
        Rtable: shape [n, n+1], thermal conductance between rooms (in W/K)
        Ctable: shape [n], heat capacity of each zone (in J/K)
        Windowtable: shape [n], exterior window area of each zone (in m^2)
    """
    # Initialize Rtable, Ctable, and Windowtable
    Rtable = np.zeros((n, n + 1))
    Ctable = np.zeros(n)
    Windowtable = np.zeros(n)

    # Set air density value
    Air = 1.225  # kg/m^3

    # Initialize the dictionary for room neighbors
    neighbors = defaultdict[str, list[int]](list)
    outind = n

    # Iterate through each layer in layers
    num_layers = len(layers)
    for k, layer in enumerate(layers):
        FloorRoom_num = len(layer)

        # Check for neighbors in the layer above
        if k + 1 < num_layers:
            for z1 in layer:
                for z2 in layers[k+1]:
                    # Check if zones are connected between layers
                    if (checkconnect_layer(z1, z2) or checkconnect_layer(z2, z1)):
                        # Calculate cross-sectional area
                        x_overlap = min(z1.Xmax, z2.Xmax) - max(z1.Xmin, z2.Xmin)
                        y_overlap = min(z1.Ymax, z2.Ymax) - max(z1.Ymin, z1.Ymin)
                        crossarea = x_overlap * y_overlap

                        # Calculate heat transfer coefficient (U) for connected zones
                        # - floor and ceiling are in series
                        U = crossarea * (ufactor.floor * ufactor.ceiling / (ufactor.floor + ufactor.ceiling))

                        # Update Rtable for connected zones
                        Rtable[z2.ind, z1.ind] = U
                        Rtable[z1.ind, z2.ind] = U

                        # Update the dictionary with connected zones
                        neighbors[z1.name].append(z2.ind)
                        neighbors[z2.name].append(z1.ind)

        # Calculate heat capacity (C) and window area for each zone in current layer
        for i, z1 in enumerate(layer):
            height = z1.Zmax - z1.Zmin
            xleng = z1.Xmax - z1.Xmin
            yleng = z1.Ymax - z1.Ymin

            # Compute heat capacity (C) for the current z1
            C_room = SpecificHeat_avg * height * xleng * yleng * Air
            Ctable[z1.ind] = C_room

            # Update Windowtable for the current z1
            Windowtable[z1.ind] = z1.ExteriorWindowArea

            # Update Rtable for exterior zones
            if z1.ExteriorGrossArea > 0 or (i == len(layer) - 1):
                if i == len(layer) - 1:
                    Rtable[z1.ind, -1] = (
                        z1.ExteriorGrossArea * ufactor.outwall
                        + xleng * yleng * ufactor.roof
                        + z1.ExteriorWindowArea * ufactor.window
                    )

                else:
                    Rtable[z1.ind, -1] = (
                        z1.ExteriorGrossArea * ufactor.outwall
                        + z1.ExteriorWindowArea * ufactor.window
                    )

                # Update the dictionary
                neighbors[z1.name].append(outind)

            # Check for neighbors within the same layer
            for j in range(i + 1, FloorRoom_num):
                z2 = layer[j]

                # Check if zones are connected within the same layer
                if checkconnect(z1, z2) or checkconnect(z2, z1):
                    # Calculate the length of the shared wall
                    x_overlap = min(z1.Xmax, z2.Xmax) - max(z1.Xmin, z2.Xmin)
                    y_overlap = min(z1.Ymax, z2.Ymax) - max(z1.Ymin, z1.Ymin)
                    length = np.sqrt(x_overlap**2 + y_overlap**2)

                    # Calculate heat transfer coefficient (U) for connected zones
                    U = height * length * ufactor.intwall

                    # Update Rtable for connected zones
                    Rtable[z2.ind, z1.ind] = U
                    Rtable[z1.ind, z2.ind] = U

                    # Update the dictionary with connected zones
                    neighbors[z1.name].append(z2.ind)
                    neighbors[z2.name].append(z1.ind)

    return neighbors, Rtable, Ctable, Windowtable


def ParameterGenerator(
    building: str,
    weather: str,
    location: str,
    U_Wall: Ufactor = (0,) * 7,
    ground_temp: Sequence[float] = (0,) * 12,
    shgc: float = 0.252,
    shgc_weight: float = 0.01,
    ground_weight: float = 0.5,
    full_occ: np.ndarray | float = 0,
    max_power: np.ndarray | int = 8000,
    ac_map: np.ndarray | int = 1,
    time_res: int = 3600,
    reward_beta: float = 0.999,
    reward_pnorm: float = 2,
    target: np.ndarray | float = 22,
    activity_sch: np.ndarray | float = 120,
    temp_range: tuple[float, float] = (-40, 40),
    is_continuous_action: bool = True,
    root: str = ''
) -> dict[str, Any]:
    """Generates parameters from the selected building and temperature file for the env.

    Args:
        building: name of a building in `BUILDINGS`, or path (relative to ``root``)
            to a htm file for building idf
        weather: name of a weather condition in `WEATHER`, or path to an epw file.
        location: name of a location in `GROUND_TEMP`
        U_Wall: list of 7 U-values (thermal transmittance) for different
            surfaces in the building in the order
            [intwall, floor, outwall, roof, ceiling, groundfloor, window].
            Only used if ``building`` cannot be found in `BUILDINGS`
        ground_temp: monthly ground temperature (in Celsius) when ``location``
            is not in `GROUND_TEMP`
        shgc: Solar Heat Gain Coefficient for windows (unitless)
        shgc_weight: Weight factor for extra loss of solar irradiance (ghi)
        ground_weight: Weight factor for extra loss of heat from ground
        full_occ: max number of people that can occupy each room, either an
            array of shape (n,) specifying maximum for each room, or a scalar
            maximum that applies to all rooms
        max_power: max power output of a single HVAC unit (in W)
        ac_map: binary indicator of presence (1) or absence (0) of AC, either a
            boolean array of shape (n,) to specify AC presence in individual
            rooms, or a scalar indicating AC presence in all rooms
        time_res: length of 1 timestep in seconds. Default is 3600 (1 hour).
        reward_beta: temperature error penalty weight for reward function
        reward_pnorm: p to use for norm in reward function
        target: target temperature setpoints (in Celsius), either an array
            specifying individual setpoints for each zone, or a scalar
            setpoint for all zones
        activity_sch: metabolic rate of people in the building (in W), either
            an array of shape (T,) to specify metabolic rate at every time
            step, or a scalar rate for all time steps
        temp_range: (min temperature, max temperature) in Celsius, defining
            the possible temperature in the building
        is_continuous_action: whether to use continuous action space (as opposed
            to MultiDiscrete)
        root: root directory for building and weather data files, only used when
            ``building`` and ``weather`` do not correspond to keys in `BUILDINGS`
            and `WEATHER`

    Returns:
        parameters: Contains all parameters needed for environment initialization.
    """
    # check if location is in GROUND_TEMP, otherwise use ground_temp
    monthly_ground_temp = GROUND_TEMP.get(location, ground_temp)

    # Calculate ground temperature for each month
    all_ground_temp = np.concatenate([
        np.ones(31 * 24 * 3600 // time_res) * monthly_ground_temp[0],
        np.ones(28 * 24 * 3600 // time_res) * monthly_ground_temp[1],
        np.ones(31 * 24 * 3600 // time_res) * monthly_ground_temp[2],
        np.ones(30 * 24 * 3600 // time_res) * monthly_ground_temp[3],
        np.ones(31 * 24 * 3600 // time_res) * monthly_ground_temp[4],
        np.ones(30 * 24 * 3600 // time_res) * monthly_ground_temp[5],
        np.ones(31 * 24 * 3600 // time_res) * monthly_ground_temp[6],
        np.ones(31 * 24 * 3600 // time_res) * monthly_ground_temp[7],
        np.ones(30 * 24 * 3600 // time_res) * monthly_ground_temp[8],
        np.ones(31 * 24 * 3600 // time_res) * monthly_ground_temp[9],
        np.ones(30 * 24 * 3600 // time_res) * monthly_ground_temp[10],
        np.ones(31 * 24 * 3600 // time_res) * monthly_ground_temp[11],
    ])

    # Check if building is in BUILDINGS, otherwise use building as building_file
    building_file: str | io.StringIO
    if building in BUILDINGS:
        internal_path = os.path.join('data', 'building', BUILDINGS[building][0])
        building_file = read_to_stringio(internal_path)
        U_Wall = BUILDINGS[building][1]
    else:
        building_file = os.path.join(root, building)

    # Get room information from the building file
    layers, n, all_zones = get_zones(building_file)
    print("###############All Zones from Ground############")
    for zone in all_zones:
        print(zone.name, " [Zone index]: ", zone.ind)
    print("###################################################")

    # Check if weather is in WEATHER, otherwise use weather as weather_file
    if weather in WEATHER:
        internal_path = os.path.join('data', 'building', WEATHER[weather])
        weather_file = read_to_stringio(internal_path)
        weather_df, weather_metadata = pvlib.iotools.parse_epw(weather_file)
    else:
        weather_path = os.path.join(root, weather)
        weather_df, weather_metadata = pvlib.iotools.read_epw(weather_path)

    # Read the weather data and interpolate temperature values
    oneyear = weather_df["temp_air"]
    num_datapoint = len(oneyear)
    x = np.arange(num_datapoint)
    y = np.array(oneyear)

    f = interpolate.interp1d(x, y)
    xnew = np.arange(0, num_datapoint - 1, 1 / 3600 * time_res)
    outtempdatanew = f(xnew)

    # Read the weather data and interpolate GHI values to the new time resolution
    oneyearrad = weather_df["ghi"]  # in Wh/m^2
    x = np.arange(num_datapoint)
    y = np.array(oneyearrad)

    f = interpolate.interp1d(x, y)
    xnew = np.arange(0, num_datapoint - 1, 1 / 3600 * time_res)
    solardatanew = f(xnew)

    # Define constants and calculate SHGC
    SpecificHeat_avg = 1000  # specific heat of indoor air, in J/kg-K
    SHGC = shgc * shgc_weight * (max(weather_df["ghi"]) / (1 / 3600 * time_res))  # GHI change from Wh to W
    # Find neighboring rooms, resistance and capacitance tables, and window properties
    neighbors, Rtable, Ctable, Windowtable = Nfind_neighbor(
        n, layers, U_Wall, SpecificHeat_avg
    )
    RCtable = Rtable / np.array([Ctable]).T

    # Initialize binary connectivity matrix
    connectmap = np.zeros((n, n + 1))
    for i, zone in enumerate(all_zones):
        connectmap[i, neighbors[zone.name]] = 1

    # calculate thermal conductance between each zone and the ground (in W/K)
    # the first layer connects to the ground
    ground_connectlist = np.zeros((n, 1))
    for room in layers[0]:
        ground_connectlist[room.ind] = (
            room.FloorArea * U_Wall.groundfloor * ground_weight
        )  # for those rooms, assign 1/R table by floor area and u factor

    # Calculate occupancy, AC weight, weighted connection map, and non-linear term
    people_full = (np.zeros(n) + full_occ).reshape(n, 1)  # shape [n, 1]
    ACweight = np.diag(np.zeros(n) + ac_map) * max_power  # shape [n, n]
    weightcmap = (
        np.concatenate(
            (
                people_full,
                ground_connectlist,
                np.zeros((n, 1)),  # this gets filled in in construct_BD_matrix
                ACweight,
                (Windowtable * SHGC).reshape(n, 1),
            ),
            axis=-1,
        )
        / Ctable[:, None]
    )

    # Construct A,B,and D matrix
    # - the occupancy coefficient comes from page 1299 of
    #   https://energyplus.net/assets/nrel_custom/pdfs/pdfs_v23.1.0/EngineeringReference.pdf.
    #   This is the only term that is linear (in temperature). It corresponds
    #   to the coefficient c4 in the BEAR paper (Zhang et al., 2023)
    OCCU_COEF = 7.139322  # in units (W/C)
    A = construct_A_matrix(RCtable, weightcmap, connectmap, OCCU_COEF, n)
    B, D = construct_BD_matrix(weightcmap, connectmap, RCtable)

    # Store parameters in a dictionary for the simulation
    parameters: dict[str, Any] = {}
    parameters['n'] = n
    parameters['zones'] = all_zones
    parameters['target'] = np.zeros(n) + target
    parameters['out_temp'] = outtempdatanew
    parameters['ground_temp'] = all_ground_temp
    parameters['ghi'] = (
        solardatanew
        / (1 / 3600 * time_res)
        / (max(weather_df['ghi']) / (1 / 3600 * time_res))
    )
    parameters['metabolism'] = activity_sch * np.ones(len(outtempdatanew))
    parameters['reward_beta'] = reward_beta
    parameters['reward_pnorm'] = reward_pnorm
    parameters['ac_map'] = np.zeros(n) + ac_map
    parameters['max_power'] = max_power
    parameters['temp_range'] = temp_range
    parameters['is_continuous_action'] = is_continuous_action
    parameters['time_resolution'] = time_res
    parameters['A'] = A
    parameters['B'] = B
    parameters['D'] = D
    return parameters


def construct_A_matrix(
    RCtable: np.ndarray,
    weightcmap: np.ndarray,
    connectmap: np.ndarray,
    occu_coef: float,
    n: int
) -> np.ndarray:
    """
    Constructs the A matrix for the building environment.

    Args:
        RCtable: shape (n, n+1), represents 1/resistance-capacitance values for each room.
            The last column represents the connection to the outside.
        weightcmap: shape (n, n+4), represents weighted connections for each room.
            Columns represent [people, ground, outside, AC, window, solar gain].
        connectmap: shape (n, n+1), represents connectivity between rooms.
            A value of 1 indicates a connection, 0 indicates no connection. The last
            column represents the connection to the outside.
        occu_coef: Occupancy linear coefficient. Represents the effect of occupancy
            on the room's temperature.
        n (int): Number of rooms or zones in the building.

    Returns:
        A: shape (n, n)
    """
    # Calculate the diagonal values for the A matrix. The ground is also considered as a node.
    ground = weightcmap[:, 1]
    diagvalue = -np.diag(RCtable @ connectmap.T) - ground

    # Copy the 1/RC table excluding the last column, which is the connection map to outside.
    A = RCtable[:, :-1].copy()

    # Replace the diagonal of A with the calculated diagonal values
    np.fill_diagonal(A, diagvalue)

    # Adjust the values of A based on numberofpeople/C(the first column in weightcmap) and n
    A += weightcmap[:, 0] * occu_coef / n
    return A


def construct_BD_matrix(
    weightcmap: np.ndarray,
    connectmap: np.ndarray,
    RCtable: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Constructs the B matrix for the building environment.

    Args:
        weightcmap: shape (n, n+4), represents weighted connections for each room.
            Columns represent [people, ground, outside, AC (n cols), solar gain].
        connectmap: shape (n, n+1), represents connectivity between rooms.
            A value of 1 indicates a connection, 0 indicates no connection. The last
            column represents the connection to the outside.
        RCtable: shape (n, n+1), represents resistance-capacitance values for each room.
            The last column represents the connection to the outside.

    Returns:
        B: B matrix of shape (n, n+3)
        D: D vector of shape (n,)
    """
    BD = weightcmap.copy()

    # Fill in the outside temperature column. Address the RC effect with outdoor temperature.
    connection_to_out = connectmap[:, -1]
    RCout = RCtable[:, -1]
    BD[:, 2] = connection_to_out * RCout

    # Construct D vector with first column of weightcmap, which is numberofpeople/C
    B = BD[:, 1:]
    D = BD[:, 0]
    return B, D
